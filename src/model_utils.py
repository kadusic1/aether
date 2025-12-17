from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_model(model_id="meta-llama/Llama-3.1-8B-Instruct"):
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=True,
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        local_files_only=True,
    )

    # Eval mode for inference
    model.eval()

    return model, tokenizer


def generate_text(model, tokenizer, system_prompt, user_prompt_text, max_tokens=650):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1,
        )

    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]).strip()
    return text
