from langchain_ollama import ChatOllama
from qwen_tts import Qwen3TTSModel

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def load_chat_model(
    model_name: str = "llama3.1:8b-instruct-q4_K_M",
) -> ChatOllama:
    """
    Load and configure the ChatOllama language model.

    Args:
        model_name: The Ollama model identifier.

    Returns:
        Configured ChatOllama instance.
    """
    return ChatOllama(
        model=model_name,
        temperature=0.8,
        top_p=0.85,
        num_predict=650,
        keep_alive=False,
    )


def load_tts_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
) -> Qwen3TTSModel:
    """
    Load the text-to-speech (TTS) model.

    Args:
        model_name: The name of the TTS model to load.

    Returns:
        Configured Qwen3TTSModel instance.
    """
    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )


def load_diffusion_model(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    lora_paths: list[str] | None = None,
) -> StableDiffusionPipeline:
    """
    Load SD 1.5 optimized for niche agents with optional LoRAs.

    Fits comfortably in 4-6GB VRAM. Applies attention slicing and VAE
    slicing for memory efficiency.

    Args:
        model_name: The Stable Diffusion model identifier.
        lora_paths: Optional list of paths to LoRA weights to load.

    Returns:
        Configured StableDiffusionPipeline instance.
    """

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()

    # Load LoRAs per agent
    if lora_paths:
        for lora_path in lora_paths:
            pipe.load_lora_weights(lora_path)

    return pipe


def _build_clip_chunks(
    pipe: StableDiffusionPipeline,
    prompt: str,
) -> torch.Tensor:
    """
    Tokenize a prompt into CLIP-compatible chunks.

    Each chunk is exactly ``model_max_length`` tokens structured as
    ``[BOS] <content tokens> [EOS] [PAD...]``, matching the format
    CLIP was trained on.

    Args:
        pipe: Pipeline whose tokenizer and device are used.
        prompt: Raw text prompt (may exceed the token limit).

    Returns:
        Tensor of shape ``(1, num_chunks * max_length)`` on CUDA,
        ready to be split and encoded chunk-by-chunk.
    """
    tokenizer = pipe.tokenizer
    max_length = tokenizer.model_max_length
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    chunk_size = max_length - 2  # room for BOS + EOS

    # Tokenize without truncation / padding to get raw IDs.
    raw_ids = (
        tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )

    # Split content tokens into chunks of ``chunk_size``.
    chunks: list[list[int]] = [
        raw_ids[i : i + chunk_size]
        for i in range(0, max(len(raw_ids), 1), chunk_size)
    ]

    # Wrap every chunk: [BOS] + content + [EOS] + padding → max_length
    built: list[list[int]] = []
    for chunk in chunks:
        padded = chunk + [eos] * (chunk_size - len(chunk))
        built.append([bos] + padded + [eos])

    flat = [tok for c in built for tok in c]
    return torch.tensor([flat], dtype=torch.long).to(pipe.device)


def generate_image_long_prompt(
    pipe: StableDiffusionPipeline,
    prompt: str,
) -> Image:
    """
    Generate an image from a long prompt using CLIP-optimized chunking.

    Every chunk sent to the text encoder is wrapped with proper BOS/EOS
    tokens so each slice looks like a well-formed CLIP input::

        [BOS] <up to 75 content tokens> [EOS] [PAD...]

    This preserves the sentence-boundary signals CLIP was trained on and
    produces higher-quality embeddings compared to naive slicing.

    See: https://github.com/huggingface/diffusers/issues/2136

    Args:
        pipe: Configured StableDiffusionPipeline instance.
        prompt: The text prompt for image generation.

    Returns:
        The generated PIL Image object.
    """
    max_length = pipe.tokenizer.model_max_length

    # --- positive prompt ---
    pos_ids = _build_clip_chunks(pipe, prompt)
    num_chunks = pos_ids.shape[-1] // max_length

    # --- negative prompt (empty string → single well-formed chunk) ---
    neg_ids = _build_clip_chunks(pipe, "")
    # Repeat the single negative chunk to match the positive count.
    if neg_ids.shape[-1] < pos_ids.shape[-1]:
        repeats = num_chunks
        neg_ids = neg_ids.repeat(1, repeats)[:, : pos_ids.shape[-1]]

    # --- encode each chunk and concatenate ---
    concat_embeds = []
    neg_embeds = []
    for i in range(num_chunks):
        start = i * max_length
        end = start + max_length
        concat_embeds.append(pipe.text_encoder(pos_ids[:, start:end])[0])
        neg_embeds.append(pipe.text_encoder(neg_ids[:, start:end])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    return pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
