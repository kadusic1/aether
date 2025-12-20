# src/__init__.py
from .model_utils import generate_text, load_model, setup_chain
from .prompt_utils import (
    system_prompt_headline,
    system_prompt_content,
    system_prompt_review,
)

__all__ = [
    "generate_text",
    "load_model",
    "system_prompt_headline",
    "system_prompt_content",
    "system_prompt_review",
    "setup_chain",
]
