from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
import re

from model_loader import load_model, ModelBundle
from chat_memory import ChatMemory, DEFAULT_PERSONA

COMMAND_PREFIX = "/"


# Generation helpers

def extract_bot_reply(full: str) -> str:
    """Robustly extract the last Bot turn.

    Strategy:
      1. Take substring after LAST occurrence of 'Bot:' (avoids earlier context copies).
      2. Truncate at the first newline that begins a new role marker ('User:' or 'Bot:').
      3. Sanitize artifacts (role echoes, bracketed tags, odd tokenization like 'u lt 3').
      4. Collapse spaces and trim.
    """
    pos = full.rfind("Bot:")
    segment = full[pos + 4:] if pos != -1 else full
    # Truncate at next role marker (allow optional space before colon)
    m = re.search(r"\n(?:User|Bot)\s*:", segment, flags=re.I)
    if m:
        segment = segment[:m.start()]
    # Remove bracketed artifacts
    segment = re.sub(r"\[[^\]]+\]", "", segment)
    # Remove stray duplicated role tokens embedded mid-string
    # Remove any inline role markers variants
    segment = re.sub(r"\b(User|Bot)\s*:", "", segment, flags=re.I)
    # Heart artifact normalization (u lt 3 / lt 3)
    segment = re.sub(r"\b(u\s+)?lt\s*3\b", "<3", segment, flags=re.I)
    # Collapse whitespace
    segment = re.sub(r"\s+", " ", segment)
    cleaned = segment.strip()
    return cleaned

LOW_INFO_PATTERN = re.compile(r"^(i'?m (a )?(simple )?bot\.?|i did\.?|i don't know\.?){1,2}$", re.I)

# Trivial single or short tokens to treat as non-informative
TRIVIAL_SET = {"hi","hey","hello","ok","k","yeah","yep","oh","cool"}

PUNCT_ONLY = re.compile(r"^[\W_]+$")

QUESTION_PATTERN = re.compile(r"\b(what|who|when|where|why|how)\b|capital of|meaning of", re.I)

def is_question(text: str) -> bool:
    return bool(QUESTION_PATTERN.search(text)) or text.strip().endswith("?")

def is_low_info(reply: str) -> bool:
    base = reply.strip().lower()
    if not base:
        return True
    if len(base) < 3:
        return True
    if base in TRIVIAL_SET:
        return True
    if PUNCT_ONLY.match(base):
        return True
    if LOW_INFO_PATTERN.match(base):
        return True
    return False

def jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def is_corrupted(text: str) -> bool:
    # Indicators of role/tag leakage or garbled structure
    if re.search(r"\b(User|Bot)\s*:\s*(User|Bot)?", text, flags=re.I):
        return True
    # Excess punctuation clusters without words
    if len(re.findall(r"[A-Za-z]", text)) < 3 and len(text) > 10:
        return True
    # Multiple disjoint role markers inside reply
    if len(re.findall(r"\b(User|Bot)\s*:", text, flags=re.I)) > 0:
        return True
    return False

def generate_reply(bundle: ModelBundle, prompt: str, user_input: str, recent_replies: List[str]) -> str:
    gen_kwargs = bundle.default_gen_kwargs.copy()
    # Strengthen length bounds
    gen_kwargs.setdefault("min_new_tokens", 8)
    gen_kwargs["max_new_tokens"] = max(gen_kwargs.get("max_new_tokens", 55), 55)
    gen_kwargs.setdefault("repetition_penalty", 1.18 if is_question(user_input) else 1.15)
    gen_kwargs.setdefault("no_repeat_ngram_size", 3)

    if is_question(user_input):
        gen_kwargs["temperature"] = min(gen_kwargs.get("temperature", 0.6), 0.55)
        gen_kwargs["top_p"] = min(gen_kwargs.get("top_p", 0.92), 0.9)

    result = bundle.pipe(prompt, **gen_kwargs)[0]["generated_text"]
    reply = extract_bot_reply(result)

    # Quick corruption cleanup & basic normalization
    reply = re.sub(r"\b(User|Bot)\s*:", "", reply, flags=re.I).strip()

    def needs_retry(r: str) -> bool:
        if is_low_info(r):
            return True
        # Duplicate damping against last 5 replies
        for prev in recent_replies[-5:]:
            if jaccard_similarity(prev, r) > 0.8:
                return True
        return False

    if needs_retry(reply) or is_corrupted(reply):
        regen = gen_kwargs.copy()
        regen["temperature"] = max(0.45, gen_kwargs.get("temperature", 0.55) - 0.05)
        regen["top_p"] = min(gen_kwargs.get("top_p", 0.9), 0.9)
        regen["repetition_penalty"] = max(1.2, gen_kwargs.get("repetition_penalty", 1.15))
        regen["min_new_tokens"] = max(8, regen.get("min_new_tokens", 8))
        result2 = bundle.pipe(prompt, **regen)[0]["generated_text"]
        cand = extract_bot_reply(result2)
        cand = re.sub(r"\b(User|Bot)\s*:", "", cand, flags=re.I).strip()
        if not needs_retry(cand):
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
    profile: Optional[str] = None,
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

    # Few-shot priming examples (polite, concise style)
    few_shot = [
        ("Hi!", "Hello! How can I help you today?"),
        ("Can you give me a quick focus tip?", "Pick one small task, silence notifications, and set a short timer."),
    ]

    # Optional system primer (short to avoid dilution)
    user_supplied_prompt = system_prompt is not None and system_prompt != ""
    persona = system_prompt if user_supplied_prompt else DEFAULT_PERSONA
    memory = ChatMemory(window_size=window, system_prompt=persona, few_shot_examples=few_shot, use_default_persona=False)
    recent_replies: List[str] = []
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
            bot_reply = generate_reply(bundle, prompt, user_input=user, recent_replies=recent_replies)
        except KeyboardInterrupt:
            print("\n[ABORTED] Generation interrupted.")
            continue
        except Exception as e:  # surface but continue session
            print(f"[ERROR] Generation failed: {e}")
            continue

        print(f"Bot: {bot_reply}")
        # Only store if not low-info to avoid reinforcing dull style
        if not is_low_info(bot_reply):
            memory.add_turn(user, bot_reply)
            recent_replies.append(bot_reply)
            if len(recent_replies) > 50:
                recent_replies.pop(0)