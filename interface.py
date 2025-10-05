from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
import re

from model_loader import load_model, ModelBundle
from chat_memory import ChatMemory, DEFAULT_PERSONA

COMMAND_PREFIX = "/"


###############################
# Dual-mode generation (legacy + chat-template)
###############################

LOW_INFO_PATTERN = re.compile(r"^(i'?m (a )?(simple )?bot\.?.|i did\.?.|i don't know\.?.){1,2}$", re.I)
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

def extract_bot_reply(full: str) -> str:
    pos = full.rfind("Bot:")
    segment = full[pos + 4:] if pos != -1 else full
    m = re.search(r"\n(?:User|Bot)\s*:", segment, flags=re.I)
    if m:
        segment = segment[:m.start()]
    segment = re.sub(r"\[[^\]]+\]", "", segment)
    segment = re.sub(r"\b(User|Bot)\s*:", "", segment, flags=re.I)
    segment = re.sub(r"\b(u\s+)?lt\s*3\b", "<3", segment, flags=re.I)
    segment = re.sub(r"\s+", " ", segment)
    return segment.strip()

def jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def is_corrupted(text: str) -> bool:
    if re.search(r"\b(User|Bot)\s*:\s*(User|Bot)?", text, flags=re.I):
        return True
    if len(re.findall(r"[A-Za-z]", text)) < 3 and len(text) > 10:
        return True
    if len(re.findall(r"\b(User|Bot)\s*:", text, flags=re.I)) > 0:
        return True
    return False

def generate_reply(bundle: ModelBundle, prompt: str, user_input: str, recent_replies: List[str]) -> str:
    gen_kwargs = bundle.default_gen_kwargs.copy()
    gen_kwargs.setdefault("min_new_tokens", 8)
    gen_kwargs["max_new_tokens"] = max(gen_kwargs.get("max_new_tokens", 55), 55)
    gen_kwargs.setdefault("repetition_penalty", 1.18 if is_question(user_input) else 1.15)
    gen_kwargs.setdefault("no_repeat_ngram_size", 3)
    if is_question(user_input):
        gen_kwargs["temperature"] = min(gen_kwargs.get("temperature", 0.6), 0.55)
        gen_kwargs["top_p"] = min(gen_kwargs.get("top_p", 0.92), 0.9)
    result = bundle.pipe(prompt, **gen_kwargs)[0]["generated_text"]
    reply = extract_bot_reply(result)
    reply = re.sub(r"\b(User|Bot)\s*:", "", reply, flags=re.I).strip()
    def needs_retry(r: str) -> bool:
        if is_low_info(r):
            return True
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

def is_chat_template_model(bundle: ModelBundle) -> bool:
    tok = getattr(bundle.pipe, "tokenizer", None)
    if tok is None:
        return False
    if hasattr(tok, "chat_template") and tok.chat_template:
        return True
    return False


def generate_reply_chat(bundle: ModelBundle, messages: list, recent_replies: List[str]) -> str:
    """Generate reply for chat-template capable models (e.g., TinyLlama)."""
    tok = bundle.pipe.tokenizer
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gen_kwargs = bundle.default_gen_kwargs.copy()
    # Chat models often like a bit more length by default
    gen_kwargs.setdefault("max_new_tokens", 128)
    # Ensure sampling defaults reasonable
    gen_kwargs.setdefault("temperature", 0.7)
    gen_kwargs.setdefault("top_p", 0.95)
    gen_kwargs.setdefault("top_k", 50)
    result = bundle.pipe(prompt, **gen_kwargs)[0]["generated_text"]
    # Extract new text after prompt
    if result.startswith(prompt):
        new_part = result[len(prompt):]
    else:
        # Fallback: attempt to split at assistant tag
        parts = re.split(r"<\|assistant\|>", result)
        new_part = parts[-1] if parts else result
    # Truncate at next role tag
    cut = re.search(r"<\|(user|system)\|>", new_part)
    if cut:
        new_part = new_part[:cut.start()]
    # Basic cleanup
    new_part = re.sub(r"</s>", "", new_part)
    new_part = re.sub(r"\s+", " ", new_part).strip()
    # Minimal duplicate / low-info filter reuse
    if is_low_info(new_part) and recent_replies:
        # one retry with slightly adjusted sampling
        regen = gen_kwargs.copy()
        regen["temperature"] = max(0.55, gen_kwargs.get("temperature", 0.7) - 0.1)
        regen["top_p"] = min(0.9, gen_kwargs.get("top_p", 0.95))
        regen["repetition_penalty"] = max(1.15, gen_kwargs.get("repetition_penalty", 1.1))
        result2 = bundle.pipe(prompt, **regen)[0]["generated_text"]
        if result2.startswith(prompt):
            cand = result2[len(prompt):]
        else:
            parts = re.split(r"<\|assistant\|>", result2)
            cand = parts[-1] if parts else result2
        cut2 = re.search(r"<\|(user|system)\|>", cand)
        if cut2:
            cand = cand[:cut2.start()]
        cand = re.sub(r"</s>", "", cand)
        cand = re.sub(r"\s+", " ", cand).strip()
        if not is_low_info(cand):
            new_part = cand
    return new_part


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
    top_k: Optional[int] = None,
    torch_dtype: Optional[str] = None,
    repetition_penalty: float = 1.18,
    no_repeat_ngram_size: int = 3,
    min_new_tokens: Optional[int] = 10,
):
    print(f"[INFO] Loading model '{model_name}' ...")
    bundle = load_model(
        model_name=model_name,
        device=device,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        torch_dtype=torch_dtype,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_new_tokens=min_new_tokens,
    )

    # Few-shot priming examples (kept small to avoid diluting chat template semantics)
    few_shot = [
        ("Hi!", "Hi there! How can I help?"),
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

        # Lightweight preprocessing shortcuts (arithmetic & simple capitals)
        lower = user.lower()
        # Arithmetic pattern: (what is)? a op b
        m = re.match(r"^(?:what'?s|what is)?\s*(\d+)\s*([+\-*/xX])\s*(\d+)\s*\??$", lower)
        if m:
            a, op, b = m.groups(); a=int(a); b=int(b)
            try:
                if op in ['x','X','*']: val = a*b
                elif op == '+': val = a+b
                elif op == '-': val = a-b
                else: val = a/b if b!=0 else 'undefined'
            except Exception:
                val = 'undefined'
            print(f"Bot: {val}")
            memory.add_turn(user, str(val))
            recent_replies.append(str(val))
            continue
        # Simple capital lookup
        CAPITALS = {
            'france':'Paris','germany':'Berlin','italy':'Rome','spain':'Madrid','india':'New Delhi',
            'japan':'Tokyo','canada':'Ottawa','brazil':'Brasília','australia':'Canberra','china':'Beijing'
        }
        cap_match = re.match(r"^(?:what('?s)?|what is)?\s*the\s*capital\s*of\s*([a-zA-Z]+)\??$", lower)
        if cap_match:
            country = cap_match.group(2).lower()
            if country in CAPITALS:
                ans = CAPITALS[country]
                print(f"Bot: {ans}")
                memory.add_turn(user, ans)
                recent_replies.append(ans)
                continue
        chat_mode = is_chat_template_model(bundle)
        messages = []
        prompt = ""
        if chat_mode:
            messages = memory.build_messages(user)
        else:
            prompt = memory.build_context(user)

        try:
            if chat_mode:
                bot_reply = generate_reply_chat(bundle, messages, recent_replies=recent_replies)
            else:
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