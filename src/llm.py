from langchain_ollama import ChatOllama
from qwen_tts import Qwen3TTSModel
import torch


def load_chat_model(model_name: str = "llama3.1:8b-instruct-q4_K_M") -> ChatOllama:
    """
    Loads and configures the ChatOllama language model.

    Args:
        model_name (str): The name of the model to load. Default is "llama3.1:8b-instruct-q4_K_M".
    Returns:
        ChatOllama: An instance of the ChatOllama model configured with specified parameters.
    """
    
    return ChatOllama(
        model=model_name,
        temperature=0.8,
        top_p=0.85,
        num_predict=650,
        keep_alive=False,
    )


def load_tts_model(model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") -> Qwen3TTSModel:
    """
    Loads the text-to-speech (TTS) model.
    
    Args:
        model_name (str): The name of the TTS model to load."
    Returns:
        Qwen3TTSModel: An instance of the Qwen3TTSModel.
    """
    return Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
