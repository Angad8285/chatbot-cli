# Entry point for the local CLI chatbot.
from __future__ import annotations

import argparse
from interface import run_cli

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local Hugging Face Chatbot CLI")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id to load")
    p.add_argument("--window", type=int, default=4, help="Number of previous turns to retain")
    p.add_argument("--temperature", type=float, default=0.55, help="Sampling temperature (creativity)")
    p.add_argument("--top-p", dest="top_p", type=float, default=0.9, help="Nucleus sampling cumulative prob cutoff")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=160, help="Max new tokens per reply")
    p.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.18, help="Penalty >1.0 discourages repeating")
    p.add_argument("--no-repeat-ngram", dest="no_repeat_ngram_size", type=int, default=3, help="Block exact ngram repetition (n)")
    p.add_argument("--min-new-tokens", dest="min_new_tokens", type=int, default=10, help="Force a minimum reply length")
    p.add_argument("--top-k", dest="top_k", type=int, default=50, help="Top-k sampling cutoff (combined with top-p)")
    p.add_argument("--torch-dtype", dest="torch_dtype", default=None, help="Force dtype (e.g. bfloat16,float16)")
    p.add_argument("--device", default=None, help="Force device: 'cpu', 'mps', 'cuda:0'")
    p.add_argument("--profile", choices=["balanced","focused","creative"], help="Parameter preset profile")
    p.add_argument("--system-prompt", dest="system_prompt", default="You are a concise, factual assistant.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (set -1 to disable)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_arg = None if args.seed == -1 else args.seed
    # Apply profile overrides if chosen
    if args.profile:
        if args.profile == "balanced":
            args.temperature, args.top_p, args.max_new_tokens = 0.6, 0.92, 55
        elif args.profile == "focused":
            args.temperature, args.top_p, args.max_new_tokens = 0.55, 0.9, 50
        elif args.profile == "creative":
            args.temperature, args.top_p, args.max_new_tokens = 0.7, 0.95, 60

    run_cli(
        model_name=args.model,
        window=args.window,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        system_prompt=args.system_prompt,
        seed=seed_arg,
        profile=args.profile,
        top_k=args.top_k,
        torch_dtype=args.torch_dtype,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=args.min_new_tokens,
    )


if __name__ == "__main__":
    main()
