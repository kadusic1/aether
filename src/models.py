from langchain_ollama import ChatOllama
from qwen_tts import Qwen3TTSModel

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import gc
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dataclasses import dataclass, field
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any, Callable
from langchain_core.runnables import Runnable


@dataclass(frozen=True)
class ProviderConfig:
    """Holds provider-specific configuration for a chat model.

    Attributes:
        provider: The name of the provider (e.g. "google", "ollama").
        model: The underlying LangChain chat model instance.
        reasoning_kwarg: The provider's kwarg name for reasoning control.
        Callers always pass ``reasoning=True/False`` to the adapter,
        which translates it to the provider-specific kwarg.
        content_extractor: Function to extract the text content from the model's
        raw response. It is needed because different providers return responses
        in different formats
        reasoning_encoder: Converts a plain bool to the value the provider
        expects (e.g. True -> "high" for Google).
        is_structured: Whether the model is wrapped for structured output.
        supports_reasoning: Whether the model supports reasoning. When false
        the adapter strips the reasoning kwarg.
    """

    provider: str
    model: BaseChatModel | Runnable
    reasoning_kwarg: str
    content_extractor: Callable[[Any], str] = field(default=lambda r: r.content)
    reasoning_encoder: Callable[[bool], Any] = field(default=lambda v: v)
    is_structured: bool = False
    supports_reasoning: bool = True


class ChatModelAdapter(Runnable):
    """Normalizes provider differences for LangChain chat models.

    Callers always pass ``reasoning=``, and the adapter translates
    it to the provider-specific kwarg transparently. Also normalizes
    invoke method response content to a plain string if the provider returns
    something more complex (e.g. list of content blocks).

    Args:
        config: Provider configuration containing the model and
            its reasoning kwarg name.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._provider = config.provider
        self._model = config.model
        self._reasoning_kwarg = config.reasoning_kwarg
        self._is_structured = config.is_structured
        self._content_extractor = config.content_extractor
        self._reasoning_encoder = config.reasoning_encoder
        self._supports_reasoning = config.supports_reasoning

    def _translate(self, kwargs) -> dict:
        """Handles different reasoning/thinking kwargs across providers.
        E.g. Gemini uses ``thinking_level`` instead of ``reasoning``.
        The _translate method allows all providers to activate reasoning/thinking
        with the ``reasoning`` kwarg.

        Returns:
            A new dict with ``reasoning`` replaced by the
            provider-specific kwarg name, if present.
        """
        if "reasoning" in kwargs:
            if not isinstance(kwargs["reasoning"], bool):
                raise ValueError(
                    f"Invalid reasoning value: {kwargs['reasoning']}. "
                    "Expected a boolean.",
                )
            # Use copy to avoid mutating the original kwargs dict passed by the caller.
            kwargs = kwargs.copy()
            if self._supports_reasoning:
                # reasoning_encoder allows providers to have custom mappings
                # (e.g. Gemini's "thinking_level").
                kwargs[self._reasoning_kwarg] = self._reasoning_encoder(
                    kwargs.pop("reasoning")
                )
            else:
                # If the provider doesn't support reasoning, we remove the reasoning
                # kwarg.
                kwargs.pop("reasoning")

        return kwargs

    def _normalize(self, response) -> str:
        """Normalize the model invoke method response content to a string if it's
        something else (e.g. list). Avoids normalization on structured output responses.

        Args:
            response: Raw response from the underlying model invoke method.

        Returns:
            Normalized response with content as a string.
        """
        if self._is_structured:
            return response

        return self._content_extractor(response)

    def invoke(self, *args, **kwargs) -> str:
        """Normalized and translated invoke method for the underlying model.
        See the docstrings of ``_translate`` and ``_normalize`` for more
        details.

        Returns:
            Normalized and translated invoke response.
        """
        return self._normalize(
            self._model.invoke(*args, **self._translate(kwargs))
        )

    async def ainvoke(self, *args, **kwargs) -> str:
        """Normalized and translated ainvoke method for the underlying model.
        See the docstrings of ``_translate`` and ``_normalize`` for more
        details.

        Returns:
            Normalized and translated invoke response.
        """
        return self._normalize(
            await self._model.ainvoke(*args, **self._translate(kwargs))
        )

    def with_structured_output(self, *args, **kwargs) -> "ChatModelAdapter":
        """Wraps structured output to obtain provider properties.

        Returns:
            A new ``ChatModelAdapter`` wrapping the structured-output
            runnable, with the same provider config preserved.
        """
        return ChatModelAdapter(
            ProviderConfig(
                provider=self._provider,
                model=self._model.with_structured_output(*args, **kwargs),
                reasoning_kwarg=self._reasoning_kwarg,
                is_structured=True,
                content_extractor=self._content_extractor,
                reasoning_encoder=self._reasoning_encoder,
                supports_reasoning=self._supports_reasoning,
            )
        )

    def stream(self, input, config=None, **kwargs):
        """Explicitly wrap stream method."""
        for chunk in self._model.stream(
            input, config=config, **self._translate(kwargs)
        ):
            yield self._normalize(chunk)

    async def astream(self, input, config=None, **kwargs):
        """Explicitly wrap astream method."""
        async for chunk in self._model.astream(
            input, config=config, **self._translate(kwargs)
        ):
            yield self._normalize(chunk)

    def batch(self, inputs, config=None, **kwargs) -> list[str]:
        """Explicitly wrap batch method."""
        responses = self._model.batch(
            inputs, config=config, **self._translate(kwargs)
        )
        return [self._normalize(r) for r in responses]

    async def abatch(self, inputs, config=None, **kwargs) -> list[str]:
        """Explicitly wrap abatch method."""
        responses = await self._model.abatch(
            inputs, config=config, **self._translate(kwargs)
        )
        return [self._normalize(r) for r in responses]

    def __getattr__(self, name: str):
        """Proxies attribute access to the underlying model.

        Args:
            name: Attribute name to look up on the wrapped model.

        Returns:
            The attribute value from the underlying model.
        """
        return getattr(self._model, name)


def load_chat_model(
    provider: str = "google",
    model: str | None = None,
    temperature: float | None = None,
) -> ChatModelAdapter:
    """Loads a provider-specific chat model, chosen by environment.

    Available providers:
        - **Google**: gemini-3.1-flash-lite-preview
        - **Ollama**: qwen3:8b-q4_k_m
    Args:
        provider: The name of the provider to load. Supported values are
            "google" and "ollama". Defaults to "google".
        temperature: Sampling temperature for the model. Higher values produce more
            random outputs, while lower values produce more deterministic outputs.
            Accepted values are [0.0, 1.0] for Ollama and [0.0, 2.0]
            for Google Gemini.

    Returns:
        A configured ``ChatModelAdapter`` for the detected provider.
    """
    if provider not in {"google", "ollama"}:
        raise ValueError(f"Unsupported provider: {provider}")
    if (
        provider == "google"
        and temperature is not None
        and not (0.0 <= temperature <= 2.0)
    ):
        raise ValueError(
            "Temperature for Google Gemini must be between 0.0 and 2.0"
        )
    if (
        provider in {"ollama"}
        and temperature is not None
        and not (0.0 <= temperature <= 1.0)
    ):
        raise ValueError(
            f"Temperature for {provider.title()} must be between 0.0 and 1.0"
        )

    if provider == "google" and os.getenv("GOOGLE_API_KEY"):
        model_kwargs = {
            "model": model or "gemini-3.1-flash-lite-preview",
            "max_retries": 6,
            # "top_p": 0.85,
        }
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        return ChatModelAdapter(
            ProviderConfig(
                provider="google",
                model=ChatGoogleGenerativeAI(**model_kwargs),
                reasoning_kwarg="thinking_level",
                content_extractor=lambda r: r.text,
                reasoning_encoder=lambda v: "high" if v else "medium",
            )
        )
    else:
        model_kwargs = {
            "model": model or "qwen3:8b-q4_k_m",
            "keep_alive": False,
            # "top_p": 0.85,
        }
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        return ChatModelAdapter(
            ProviderConfig(
                provider="ollama",
                model=ChatOllama(**model_kwargs),
                reasoning_kwarg="reasoning",
            )
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
    Example:
        import soundfile as sf

        tts = load_tts_model()
        char = {
            "name": "Wise Old Wizard",
            "text": "Young one, the path to wisdom is paved with patience and perseverance.",
            "instruct": "Elderly male voice, deep and gravelly, speaking slowly with wisdom and authority.",
            "language": "English"
        }
        wavs, sr = tts.generate_voice_design(
            text=char['text'],
            language=char['language'],
            instruct=char['instruct'],
        )
        sf.write("test.wav", wavs[0], sr)
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


def unload_from_gpu(*models: object) -> None:
    """
    Move model(s) off GPU and release VRAM.
    Handles both nn.Module instances and wrapper objects
    like Qwen3TTSModel (which store the real model in
    .model attribute).
    Args:
        *models: Model objects to unload and delete.
    """
    for model in models:
        # Unwrap: Qwen3TTSModel stores real model in .model
        inner = getattr(model, "model", model)
        if hasattr(inner, "to"):
            inner.to("cpu")
        del inner
    del models
    gc.collect()
    torch.cuda.empty_cache()
