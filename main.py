# Entry point for the local CLI chatbot.
from __future__ import annotations

import argparse
from interface import run_cli

DEFAULT_MODEL = "microsoft/DialoGPT-small"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local Hugging Face Chatbot CLI")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id to load")
    p.add_argument("--window", type=int, default=4, help="Number of previous turns to retain")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top-p", dest="top_p", type=float, default=0.9, help="Nucleus sampling p")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=80, help="Max new tokens per reply")
    p.add_argument("--device", default=None, help="Force device: 'cpu', 'mps', 'cuda:0'")
    p.add_argument("--system-prompt", dest="system_prompt", default="You are a concise, factual assistant.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (set -1 to disable)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_arg = None if args.seed == -1 else args.seed
    run_cli(
        model_name=args.model,
        window=args.window,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        system_prompt=args.system_prompt,
        seed=seed_arg,
    )


if __name__ == "__main__":
    main()
