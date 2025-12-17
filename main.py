from src import (
    load_model,
    generate_text,
    system_prompt_headline,
    system_prompt_content,
    system_prompt_review,
)


def main():
    model, tokenizer = load_model()

    # Step 1: Generate headline
    headline = generate_text(
        model,
        tokenizer,
        system_prompt_headline,
        "Generate one viral headline for a psychology/manipulation channel.",
    )
    print("Headline:\n", headline)

    # Step 2: Generate content based on headline
    content_prompt = (
        f"Create 6-8 viral short-form items with the headline: '{headline}'"
    )
    content = generate_text(model, tokenizer, system_prompt_content, content_prompt)
    print("\nGenerated Content:\n", content)

    # Step 3: Review and refine content
    review_prompt = (
        f"Here is the content:\n{content}\nImprove it according to the rules."
    )
    refined_content = generate_text(
        model, tokenizer, system_prompt_review, review_prompt
    )
    print("\nRefined Content:\n", refined_content)


if __name__ == "__main__":
    main()
