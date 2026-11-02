"""Terminal chat with DocBot (no browser needed).

Usage:  python chat_cli.py
"""

from __future__ import annotations

import sys

from docbot.config import settings
from docbot.llm import OllamaError
from docbot.rag import DocBot


def main() -> int:
    try:
        bot = DocBot(settings)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(f"DocBot ready ({len(bot.store)} chunks, model: {settings.chat_model}). Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        try:
            stream, hits = bot.ask(q, history)
            print("DocBot: ", end="", flush=True)
            answer = ""
            for token in stream:
                print(token, end="", flush=True)
                answer += token
            print()
            for i, h in enumerate(hits, 1):
                print(f"   [{i}] {h.chunk.citation}  (score {h.score:.2f})")
            print()
        except OllamaError as exc:
            print(f"Error: {exc}\n")
            continue
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
    return 0


if __name__ == "__main__":
    sys.exit(main())
