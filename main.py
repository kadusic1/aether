from src.models import load_chat_model


def main():
    # Load the pretrained language model with 4-bit quantization and
    # tokenizer from Hugging Face Hub, wrapped in LangChain (ChatHuggingFace)
    llm = load_chat_model()

    example = llm.invoke("What is 2+2?")
    print(example)


if __name__ == "__main__":
    main()
