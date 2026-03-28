> **Note:** This project is actively under development. Documentation is continuously updated.

# Aether

Short-form content generation engine.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-fast-magenta.svg)](https://github.com/astral-sh/uv)
[![Framework: LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange.svg)](https://python.langchain.com/docs/langgraph)
[![Status: WIP](https://img.shields.io/badge/Status-Work_in_Progress-yellow.svg)]()
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

## Overview

Aether is a system designed to research, script, narrate, and generate short-form video content (TikTok, YouTube Shorts, Instagram Reels).

It features a hybrid cloud/local architecture. By default, it leverages cloud LLMs (Google Gemini, Mistral). Its modular adapter system allows switching to local hardware inference (via Ollama, Stable Diffusion, and local TTS).

## Features & Architecture

- **Orchestration:** Built on LangGraph, utilizing cyclical graphs and typed state management (`TypedDict`) to handle workflows like ideation, web research, self-reflection, and script review.
- **LLM Abstraction:** A custom `ChatModelAdapter` normalizes interactions across different providers (Google GenAI, Mistral, Ollama), handling structured outputs and reasoning arguments.
- **Multimodal Generation Pipeline:** Integration with PyTorch and HuggingFace Diffusers (Stable Diffusion) for visual generation, alongside Qwen TTS for voiceovers.
- **Automated Research & Scraping:** Uses Crawl4AI for web scraping and yt-dlp with transcript APIs for YouTube research and data extraction.
- **Stack:** Written in Python 3.13, fully typed with Pydantic, and managed by `uv`.
- **Domain-Driven Design:** Separation of logic (`src/nodes/`) from niche configurations (`niche_config/`), allowing scaling into new content categories (e.g., Psychology, Finance, Tech).

## Tech Stack

- **Core & Tooling:** Python 3.13, `uv`, Ruff, Pytest
- **Orchestration:** LangGraph, LangChain Core
- **LLM Providers:** Google Generative AI, MistralAI, Ollama
- **Multimodal & ML:** PyTorch, Diffusers, Qwen-TTS
- **Data & Research:** Pydantic, Crawl4AI, YouTube Transcript API, yt-dlp

## Project Status

Aether is currently under active development. The core LangGraph state machine, provider abstractions, and research nodes are implemented. Active work is focused on optimizing the multi-step render pipeline and expanding the local-only fallback mechanisms.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.