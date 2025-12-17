from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_model(model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Load a pretrained causal language model with 4-bit quantization.

    Loads a model from Hugging Face Hub with 4-bit quantization to
    reduce memory usage. Also loads the corresponding tokenizer for
    text processing.

    Args:
        model_id: Model identifier on Hugging Face Hub. Defaults to
                  meta-llama/Llama-3.1-8B-Instruct

    Returns:
        Tuple containing the loaded model and tokenizer.
    """
    # 4-bit quantization config - reduces model size and memory usage by
    # converting weights to 4-bit precision
    # load_in_4bit=True: Enable 4-bit quantization to reduce memory
    # bnb_4bit_quant_type="nf4": Use normalized float 4-bit quantization
    # type (better precision-efficiency tradeoff)
    # bnb_4bit_compute_dtype=torch.bfloat16: Use bfloat16 for computations
    # to maintain numerical stability
    # bnb_4bit_use_double_quant=True: Apply double quantization to reduce
    # memory even further
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer from pretrained model
    # model_id: Identifier for the pretrained model on Hugging Face Hub
    # use_fast=True: Use the faster Rust-based tokenizer if available
    # local_files_only=True: Only use cached files, don't download
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=True,
    )

    # Load pretrained causal language model with 4-bit quantization
    # model_id: Identifier for the pretrained model on Hugging Face Hub
    # quantization_config=bnb_config: Apply the 4-bit quantization config
    # device_map="cuda": Load model onto GPU (CUDA device) for inference
    # local_files_only=True: Only use cached files, don't download
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        local_files_only=True,
    )

    # Set model to evaluation mode to disable dropout and batch
    # normalization for inference. This ensures consistent predictions
    # without randomness from training-specific layers
    model.eval()

    return model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt_text: str,
    max_tokens: int = 650,
):
    """
    Generate text using a pretrained language model with chat prompts.

    Takes system and user prompts, formats them for the model, and
    generates a response using sampling-based decoding with temperature
    and nucleus sampling for natural output.

    Params:
        model: Pretrained causal language model.
        tokenizer: Tokenizer matching the model.
        system_prompt: System prompt that sets model behavior.
        user_prompt_text: User input to generate response for.
        max_tokens: Maximum tokens to generate. Defaults to 650.

    Returns:
        Generated text response from the model.
    """
    # Create message list with system and user prompts for chat template
    # role="system": System message that sets the behavior and context
    # role="user": User input message that the model should respond to
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_text},
    ]

    # Apply chat template and tokenize messages into model input format
    # apply_chat_template(): Format messages according to model's chat
    # add_generation_prompt=True: Append token to signal generation start
    # tokenize=True: Convert text to token IDs the model understands
    # return_dict=True: Return output as dictionary with 'input_ids'
    # return_tensors="pt": Return PyTorch tensors instead of lists
    # .to(model.device): Move tensors to same device as model (GPU/CPU)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate text with specified parameters in no-gradient context
    # torch.no_grad(): Disable gradient computation for speed/memory
    # **inputs: Unpack input_ids and attention_mask tensors
    # max_new_tokens: Maximum number of new tokens to generate (650 default)
    # temperature=0.8: Controls randomness (lower=deterministic, high=creative)
    # top_p=0.85: Nucleus sampling - tokens with cumulative prob up to 0.85
    # do_sample=True: Use sampling instead of greedy for natural outputs
    # repetition_penalty=1.1: Penalize repeated tokens to reduce repetition
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1,
        )

    # Decode generated tokens back to text
    # outputs[0]: Get first batch item from generated output
    # [inputs["input_ids"].shape[-1]:]: Slice for newly generated tokens
    # .strip(): Remove leading/trailing whitespace
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]).strip()
    return text
