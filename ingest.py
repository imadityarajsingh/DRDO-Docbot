"""Build the search index from the documents folder.

Usage:
    python ingest.py                 # uses DOCS_DIR from .env (default: data/docs)
    python ingest.py --docs ./manuals
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from docbot.config import settings
from docbot.llm import OllamaError
from docbot.rag import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Index documents for DocBot.")
    parser.add_argument("--docs", type=Path, help="Folder containing documents to index")
    parser.add_argument("--index", type=Path, help="Where to store the index")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = settings
    if args.docs:
        cfg = replace(cfg, docs_dir=args.docs)
    if args.index:
        cfg = replace(cfg, index_dir=args.index)

    try:
        build_index(cfg, progress=print)
    except (OllamaError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
