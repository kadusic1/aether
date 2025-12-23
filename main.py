from src.utils import load_model, setup_chain
from src.engines.factory import ContentEngineFactory
from src.constants import EngineType


def main():
    # Load the pretrained language model with 4-bit quantization and
    # tokenizer from Hugging Face Hub, wrapped in LangChain (ChatHuggingFace)
    llm = load_model()

    # Set up the LCEL chain once for performance optimization.
    # Reuse this chain across multiple generate_text calls.
    chain = setup_chain(llm)

    simple_generator = ContentEngineFactory.create(
        EngineType.PSYCHOLOGY,
        chain,
    )

    results = simple_generator.generate()
    print(results)


if __name__ == "__main__":
    main()
