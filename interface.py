from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from typing import Optional, Dict, Any

from model_loader import load_model, ModelBundle
from chat_memory import ChatMemory

COMMAND_PREFIX = "/"


# Generation helpers

def extract_bot_reply(full: str) -> str:
    if "[Bot]:" in full:
        segment = full.split("[Bot]:")[-1]
    else:
        segment = full
    # Truncate at potential new user markers
    for marker in ("[User]:", "User:", "Human:"):
        pos = segment.find(marker)
        if pos != -1:
            segment = segment[:pos]
    # Basic cleanup of bracket noise (heuristic)
    cleaned = segment.strip()
    return cleaned

def generate_reply(bundle: ModelBundle, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> str:
    gen_kwargs = bundle.default_gen_kwargs.copy()
    if overrides:
        gen_kwargs.update(overrides)
    result = bundle.pipe(prompt, **gen_kwargs)[0]["generated_text"]
    return extract_bot_reply(result)


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
            bot_reply = generate_reply(bundle, prompt)
        except KeyboardInterrupt:
            print("\n[ABORTED] Generation interrupted.")
            continue
        except Exception as e:  # surface but continue session
            print(f"[ERROR] Generation failed: {e}")
            continue

        print(f"Bot: {bot_reply}")
        memory.add_turn(user, bot_reply)