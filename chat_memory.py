from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Iterable, Sequence


@dataclass
class Turn:
    # Represents a single completed conversational turn.
    user: str
    bot: str


DEFAULT_PERSONA = "You are a helpful, friendly assistant. Be concise and constructive."


class ChatMemory:
    """Sliding window memory with optional few-shot priming.

    Args:
        window_size: Max number of recent (user, bot) turns to retain.
        system_prompt: Optional plain-text directive (no role tags) placed at top.
        few_shot_examples: Optional list of (user, bot) example pairs that always
            remain at the top and are NOT trimmed (anchoring behavior).
    """

    def __init__(self, window_size: int = 4, system_prompt: Optional[str] = None,
                 few_shot_examples: Optional[Sequence[tuple[str, str]]] = None,
                 use_default_persona: bool = False):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._turns: Deque[Turn] = deque()
        # Hidden system prompt (not exported in transcript). If use_default_persona and none provided, set default.
        if system_prompt:
            self.system_prompt = system_prompt.strip()
        elif use_default_persona:
            self.system_prompt = DEFAULT_PERSONA
        else:
            self.system_prompt = ""
        self._few_shot: List[Turn] = []
        if few_shot_examples:
            for u, b in few_shot_examples:
                self._few_shot.append(Turn(user=u, bot=b))

    # Public API
    def add_turn(self, user: str, bot: str) -> None:
        self._turns.append(Turn(user=user, bot=bot))
        while len(self._turns) > self.window_size:
            self._turns.popleft()

    def reset(self) -> None:
        # Clear all stored turns.
        self._turns.clear()

    def build_context(self, next_user: str) -> str:
        """Assemble prompt string for next generation.

        Format (no bracketed role tags to better fit DialoGPT distribution):
            <system_prompt>\n
            User: <few-shot user 1>\n
            Bot: <few-shot bot 1>\n
            ... (few-shot) ...\n
            User: <recent user>\n
            Bot: <recent bot>\n
            ... up to window_size ...\n
            User: <next_user>\n
            Bot:
        """
        lines: List[str] = []
        if self.system_prompt.strip():
            lines.append(self.system_prompt.strip())
            lines.append("")
        for t in self._few_shot:
            lines.append(f"User: {t.user}")
            lines.append(f"Bot: {t.bot}")
        for t in self._turns:
            lines.append(f"User: {t.user}")
            lines.append(f"Bot: {t.bot}")
        lines.append(f"User: {next_user}")
        lines.append("Bot:")
        return "\n".join(lines)

    def export_transcript(self) -> str:
        """Return transcript excluding the hidden system prompt and few-shot examples."""
        out_lines: List[str] = []
        for t in self._turns:
            out_lines.append(f"User: {t.user}\nBot: {t.bot}\n")
        return "".join(out_lines).rstrip()

    def __len__(self) -> int: 
        return len(self._turns)

    def iter_turns(self) -> Iterable[Turn]: 
        return iter(self._turns)


if __name__ == "__main__":  # Simple manual test
    mem = ChatMemory(window_size=2)
    mem.add_turn("Hello", "Hi there!")
    mem.add_turn("How are you?", "I'm functioning as expected.")
    # This will push out the first when next added
    print(mem.build_context("What can you do?"))
    mem.add_turn("What can you do?", "I can generate text.")
    mem.add_turn("Another question", "Another answer")
    print("--- Transcript ---")
    print(mem.export_transcript())
