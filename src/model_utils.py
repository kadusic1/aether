from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
import torch


def load_model(model_id: str = "meta-llama/Llama-3.1-8B-Instruct") -> ChatHuggingFace:
    """
    Load a pretrained language model with 4-bit quantization wrapped in LangChain.

    Loads a model from Hugging Face Hub with 4-bit quantization to
    reduce memory usage. Also loads the corresponding tokenizer for
    text processing. Returns a LangChain ChatHuggingFace wrapper that
    integrates the model and tokenizer for chat-based text generation.

    Args:
        model_id: Model identifier on Hugging Face Hub. Defaults to
                  meta-llama/Llama-3.1-8B-Instruct

    Returns:
        ChatHuggingFace: A LangChain ChatHuggingFace object that wraps
                        the quantized model and tokenizer for text
                        generation in chat format.
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

    # Initialize the Transformers pipeline for text generation.
    # This acts as the backend engine that manages the model and tokenizer.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # Controls randomness: 0.1 is focused/stable, 0.8+ is creative/diverse.
        temperature=0.8,
        # Nucleus sampling: only considers the top 85% of most likely tokens.
        top_p=0.85,
        # The maximum number of new tokens to generate (excluding input).
        max_new_tokens=650,
        # Must be True to enable the sampling-based decoding (temp/top_p).
        do_sample=True,
        # Discourages the model from repeating the same phrases or words.
        repetition_penalty=1.1,
        # Only return the newly generated text, without the original prompt.
        return_full_text=False,
    )

    # Wrap the native Transformers pipeline into a LangChain-compatible object.
    # This enables use within LCEL (LangChain Expression Language) chains.
    llm = HuggingFacePipeline(pipeline=pipe)

    # This specifically links the tokenizer to the chat formatting logic
    return ChatHuggingFace(llm=llm, tokenizer=tokenizer)


# Create a chat prompt template with system and user message roles.
# This follows the standard chat structure where the system message
# provides context and the user message is the actual query.
_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system}"),
        ("user", "{user}"),
    ]
)


def setup_chain(chat_model: ChatHuggingFace) -> Runnable:
    """
    Set up a LangChain Expression Language (LCEL) chain for chat-based text generation.

    Creates a processing chain that takes system and user prompts,
    formats them using a chat prompt template, passes them through
    the provided chat model, and parses the output into a clean string.

    Args:
        chat_model: A LangChain ChatHuggingFace model instance for
                   generating text responses.
    Returns:
        chain: An LCEL chain that processes prompts and generates
               text responses.
    """
    # Create the LCEL chain: Template -> Model -> String Output
    # The chat_model automatically applies the tokenizer's chat template
    # formatting to ensure proper message structure for the model.
    # StrOutputParser converts the model output to a clean string.
    return _PROMPT_TEMPLATE | chat_model | StrOutputParser()


def generate_text(
    chain: Runnable,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Generate text using a pre-built LCEL chain with system and user prompts.

    Invokes the provided chain with system and user messages to generate
    a text response. The chain should be created once using setup_chain()
    and reused across multiple calls for better performance.

    Args:
        chain: A pre-built LCEL Runnable chain created from setup_chain().
        system_prompt: The system prompt that sets the behavior and
                      context for the model.
        user_prompt: The user input prompt to generate a response for.

    Returns:
        str: The generated text response from the model, stripped of
             leading/trailing whitespace.
    """

    # Invoke the chain with the provided prompts to generate a response.
    # The template fills in the {system} and {user} placeholders with
    # the actual prompt values.
    return chain.invoke({"system": system_prompt, "user": user_prompt}).strip()
