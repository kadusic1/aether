from src.llm import load_model


def main():
    # Load the pretrained language model with 4-bit quantization and
    # tokenizer from Hugging Face Hub, wrapped in LangChain (ChatHuggingFace)
    llm = load_model()

    example = llm.invoke("What is 2+2?")
    print(example)


if __name__ == "__main__":
    main()
