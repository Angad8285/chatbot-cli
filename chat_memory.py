from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional, Iterable


@dataclass
class Turn:
    # Represents a single completed conversational turn.
    user: str
    bot: str


class ChatMemory:
    """
        window_size: Maximum number of retained turns.
        system_prompt: Optional system role description to prepend.
    """

    def __init__(self, window_size: int = 4, system_prompt: Optional[str] = None):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._turns: Deque[Turn] = deque()
        self.system_prompt = (
            system_prompt or "You are a concise, factual assistant."
        )

    # Public API
    def add_turn(self, user: str, bot: str) -> None:
        self._turns.append(Turn(user=user, bot=bot))
        while len(self._turns) > self.window_size:
            self._turns.popleft()

    def reset(self) -> None:
        # Clear all stored turns.
        self._turns.clear()

    def build_context(self, next_user: str) -> str:
        # Construct a textual prompt including prior turns and new user message.

        lines: List[str] = [f"[System]: {self.system_prompt}"]
        for t in self._turns:
            lines.append(f"[User]: {t.user}")
            lines.append(f"[Bot]: {t.bot}")
        lines.append(f"[User]: {next_user}")
        lines.append("[Bot]:")
        return "\n".join(lines)

    def export_transcript(self) -> str:
        """Return a simple multi-line string transcript."""
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
