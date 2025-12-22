from src.utils import load_model, setup_chain
from src.generators import NoVideoSimpleContentGenerator


def main():
    # Load the pretrained language model with 4-bit quantization and
    # tokenizer from Hugging Face Hub, wrapped in LangChain (ChatHuggingFace)
    llm = load_model()

    # Set up the LCEL chain once for performance optimization.
    # Reuse this chain across multiple generate_text calls.
    chain = setup_chain(llm)

    simple_generator = NoVideoSimpleContentGenerator(chain)
    results = simple_generator.generate()
    print(results)


if __name__ == "__main__":
    main()
