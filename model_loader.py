from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import (
    pipeline,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import transformers

DEFAULT_MODEL_NAME = "microsoft/DialoGPT-small"


@dataclass
class ModelBundle:
    pipe: Any
    model_name: str
    pad_token_id: int
    default_gen_kwargs: Dict[str, Any]


def detect_device(preferred: Optional[str] = None) -> int | str | None:
    if preferred:
        return preferred
    if torch.cuda.is_available():  # GPU
        return 0
    # MPS (Apple Silicon) support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None  # CPU


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    seed: Optional[int] = 42,
    temperature: float = 0.6,
    top_p: float = 0.92,
    max_new_tokens: int = 45,
    do_sample: bool = True,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    min_new_tokens: int | None = None,
) -> ModelBundle:
    """Args:
        model_name: Hugging Face model identifier.
        device: Explicit device override ('cpu', 'mps', 'cuda:0'); if None auto-detect.
        seed: Random seed for reproducibility (None to skip seeding).
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_new_tokens: Maximum newly generated tokens per call.
        do_sample: Whether to enable sampling (True recommended for creativity).
    """
    # Basic version guard (avoid very old transformers).
    try:
        ver_tuple = tuple(int(x) for x in transformers.__version__.split(".")[:3])
        min_req = (4, 30, 0)
        if ver_tuple < min_req:
            raise RuntimeError(
                f"transformers>={' '.join(map(str,min_req))} required; found {transformers.__version__}"
            )
    except Exception:  # pragma: no cover - defensive
        pass

    if seed is not None:
        set_seed(seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer for '{model_name}': {e}") from e
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    # Ensure pad_token_id is set (GPT-style models often lack it)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    resolved_device = detect_device(device)

    # Build pipeline with fallback if MPS selected but fails (common edge case).
    try:
        text_gen_pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
        )
    except Exception as e:
        if resolved_device == "mps":
            print("[WARN] MPS initialization failed; retrying on CPU.")
            text_gen_pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                device=None,
            )
        else:
            raise RuntimeError(f"Failed to create generation pipeline: {e}") from e

    default_gen = dict(
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    if min_new_tokens is not None:
        default_gen["min_new_tokens"] = min_new_tokens

    return ModelBundle(
        pipe=text_gen_pipe,
        model_name=model_name,
        pad_token_id=tokenizer.pad_token_id,
        default_gen_kwargs=default_gen,
    )

# Simple manual smoke test when run directly.
if __name__ == "__main__":  
    bundle = load_model()
    test_prompt = "User: Hello\nAssistant:"
    print("Loaded model:", bundle.model_name)
    out = bundle.pipe(test_prompt, **bundle.default_gen_kwargs)[0]["generated_text"]
    print(out)
