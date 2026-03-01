from langchain_ollama import ChatOllama


def load_model():
    """
    Loads and configures the ChatOllama language model.

    Returns:
        ChatOllama: An instance of the ChatOllama model configured with specified parameters.
    """
    model_name = "llama3.1:8b-instruct-q4_K_M"
    llm = ChatOllama(
        model=model_name,
        temperature=0.8,
        top_p=0.85,
        num_predict=650,
        keep_alive=False,
    )
    return llm
