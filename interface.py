from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from typing import Optional, Dict, Any
import re

from model_loader import load_model, ModelBundle
from chat_memory import ChatMemory

COMMAND_PREFIX = "/"


# Generation helpers

def extract_bot_reply(full: str) -> str:
    # Find last "Bot:" marker; else take whole.
    pos = full.rfind("Bot:")
    segment = full[pos + 4:] if pos != -1 else full
    # Stop if model starts next turn prematurely
    for marker in ("User:", "USER:"):
        cut = segment.find(marker)
        if cut != -1:
            segment = segment[:cut]
    # Remove stray bracketed artifacts if any
    segment = re.sub(r"\[[^\]]+\]", "", segment)
    segment = re.sub(r"\s+", " ", segment)
    return segment.strip()

LOW_INFO_PATTERN = re.compile(r"^(i'?m (a )?(simple )?bot\.?|i did\.?|i don't know\.?){1,2}$", re.I)

QUESTION_PATTERN = re.compile(r"\b(what|who|when|where|why|how)\b|capital of|meaning of", re.I)

def is_question(text: str) -> bool:
    return bool(QUESTION_PATTERN.search(text)) or text.strip().endswith("?")

def is_low_info(reply: str) -> bool:
    return len(reply) < 12 or LOW_INFO_PATTERN.match(reply.strip()) is not None

def generate_reply(bundle: ModelBundle, prompt: str, user_input: str) -> str:
    gen_kwargs = bundle.default_gen_kwargs.copy()
    # Add anti-repetition if not present
    gen_kwargs.setdefault("repetition_penalty", 1.15)
    gen_kwargs.setdefault("no_repeat_ngram_size", 3)
    gen_kwargs.setdefault("max_new_tokens", min(gen_kwargs.get("max_new_tokens", 60), 60))

    if is_question(user_input):
        # Make answers crisper
        gen_kwargs["temperature"] = min(gen_kwargs.get("temperature", 0.7), 0.5)
        gen_kwargs["top_p"] = min(gen_kwargs.get("top_p", 0.9), 0.88)
        gen_kwargs["max_new_tokens"] = min(gen_kwargs.get("max_new_tokens", 60), 50)

    result = bundle.pipe(prompt, **gen_kwargs)[0]["generated_text"]
    reply = extract_bot_reply(result)
    if is_low_info(reply):
        # Retry once with slightly adjusted parameters
        regen = gen_kwargs.copy()
        regen["temperature"] = max(0.35, gen_kwargs["temperature"] * 0.9)
        regen["top_p"] = max(0.8, gen_kwargs.get("top_p", 0.85))
        result2 = bundle.pipe(prompt, **regen)[0]["generated_text"]
        cand = extract_bot_reply(result2)
        if not is_low_info(cand):
            reply = cand
    return reply


# Command handling
def save_transcript(memory: ChatMemory, path: Path) -> Path:
    data = [ {"user": t.user, "bot": t.bot} for t in memory.iter_turns() ]
    path.write_text(json.dumps(data, indent=2))
    return path

def handle_command(cmd: str, memory: ChatMemory, bundle: ModelBundle) -> bool:
    parts = cmd.split()
    base = parts[0].lower()

    if base == "/exit":
        print("Exiting chatbot. Goodbye!")
        return False
    if base == "/help":
        print("Commands: /exit /help /reset /save [file] /config")
        return True
    if base == "/reset":
        memory.reset()
        print("[OK] Memory cleared.")
        return True
    if base == "/save":
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = parts[1] if len(parts) > 1 else f"transcript_{ts}.json"
        path = save_transcript(memory, Path(fname))
        print(f"[OK] Saved transcript to {path}")
        return True
    if base == "/config":
        cfg = bundle.default_gen_kwargs
        print(
            f"Model={bundle.model_name} temperature={cfg['temperature']} top_p={cfg['top_p']} "
            f"max_new_tokens={cfg['max_new_tokens']} window={memory.window_size}"
        )
        return True

    print("[WARN] Unknown command. Type /help for list.")
    return True


# Main loop
def run_cli(
    model_name: str,
    window: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: Optional[str],
    system_prompt: Optional[str],
    seed: Optional[int],
):
    print(f"[INFO] Loading model '{model_name}' ...")
    bundle = load_model(
        model_name=model_name,
        device=device,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    memory = ChatMemory(window_size=window, system_prompt=system_prompt)
    print("[INFO] Chatbot ready. Type /help for commands. /exit to quit.")

    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting chatbot. Goodbye!")
            break

        if not user:
            continue
        if user.startswith(COMMAND_PREFIX):
            if not handle_command(user, memory, bundle):
                break
            continue

        # Build prompt
        prompt = memory.build_context(user)

        try:
            bot_reply = generate_reply(bundle, prompt, user_input=user)
        except KeyboardInterrupt:
            print("\n[ABORTED] Generation interrupted.")
            continue
        except Exception as e:  # surface but continue session
            print(f"[ERROR] Generation failed: {e}")
            continue

        print(f"Bot: {bot_reply}")
        memory.add_turn(user, bot_reply)