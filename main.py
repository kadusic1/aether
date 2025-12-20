from src import (
    load_model,
    generate_text,
    system_prompt_headline,
    system_prompt_content,
    system_prompt_review,
    setup_chain,
)


def main():
    # Load the pretrained language model with 4-bit quantization and
    # tokenizer from Hugging Face Hub, wrapped in LangChain (ChatHuggingFace)
    llm = load_model()

    # Set up the LCEL chain once for performance optimization.
    # Reuse this chain across multiple generate_text calls.
    chain = setup_chain(llm)

    # Step 1: Generate headline using the model
    # system_prompt_headline: Sets model behavior for headline generation
    # User prompt: Specifies the task of creating viral headlines
    headline = generate_text(
        chain,
        system_prompt_headline,
        "Generate one viral headline for a psychology/manipulation channel.",
    )
    print("Headline:\n", headline)

    # Step 2: Generate content based on the generated headline
    # Create a prompt that includes the headline for consistency
    # system_prompt_content: Sets model behavior for content generation
    # User prompt: Specifies creating 6-8 short-form items
    content_prompt = (
        f"Create 6-8 viral short-form items with the headline: '{headline}'"
    )
    content = generate_text(chain, system_prompt_content, content_prompt)
    print("\nGenerated Content:\n", content)

    # Step 3: Review and refine the generated content
    # system_prompt_review: Sets model behavior for content improvement
    # User prompt: Includes the content and requests improvement
    review_prompt = (
        f"Here is the content:\n{content}\nImprove it according to the rules."
    )
    refined_content = generate_text(chain, system_prompt_review, review_prompt)
    print("\nRefined Content:\n", refined_content)


if __name__ == "__main__":
    main()
