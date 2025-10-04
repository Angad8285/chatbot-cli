# Local Command-Line Chatbot (Work in Progress)

This project implements a local CLI chatbot using a Hugging Face text generation model.

Current status: Model loader implemented for `openai-community/openai-gpt`.

## Features (Planned)
- Hugging Face pipeline based generation
- Sliding window memory buffer
- Interactive CLI with commands (/exit, /reset, /save, /config)
- Configurable generation parameters

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model_loader.py  # smoke test
```

## Model Loader
The loader sets up a `text-generation` pipeline with safe defaults and auto device detection (CUDA > MPS > CPU). It also ensures a `pad_token_id` is defined for GPT-style models.

