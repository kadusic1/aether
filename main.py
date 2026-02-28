from src.llm import load_model, setup_chain, generate_text


def main():
    # Load the pretrained language model with 4-bit quantization and
    # tokenizer from Hugging Face Hub, wrapped in LangChain (ChatHuggingFace)
    llm = load_model()

    # Set up the LCEL chain once for performance optimization.
    # Reuse this chain across multiple generate_text calls.
    chain = setup_chain(llm)

    example = generate_text(
        chain,
        system_prompt="You are a helpful assistant.",
        user_prompt="What is Aether?",
    )
    print(example)


if __name__ == "__main__":
    main()
