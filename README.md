# Local Command-Line Chatbot (Work in Progress)

This project implements a local CLI chatbot using a Hugging Face text generation model.

Current status: Default model set to `microsoft/DialoGPT-small` (a small dialogue-tuned model). You can override with `--model <other-id>`.

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
The loader sets up a `text-generation` pipeline with safe defaults and auto device detection (CUDA > MPS > CPU). It also ensures a `pad_token_id` is defined if missing.

Default: `microsoft/DialoGPT-small` (dialogue-optimized). Example override:

```bash
python main.py --model distilgpt2
```

