"""Streamlit chat UI for DocBot.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from docbot.config import settings
from docbot.llm import OllamaClient, OllamaError
from docbot.loader import SUPPORTED_EXTENSIONS
from docbot.rag import DocBot, build_index, index_exists

st.set_page_config(page_title="DocBot · Offline Document Assistant", page_icon="📄", layout="wide")

client = OllamaClient(settings.ollama_url, settings.request_timeout)


# ---------------------------------------------------------------- helpers
def get_bot(cfg) -> DocBot | None:
    key = (cfg.chat_model, cfg.top_k, cfg.min_score)
    if st.session_state.get("bot_key") != key or st.session_state.get("bot") is None:
        if not index_exists(cfg):
            return None
        st.session_state.bot = DocBot(cfg, client)
        st.session_state.bot_key = key
    return st.session_state.bot


def list_docs() -> list[Path]:
    d = Path(settings.docs_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📄 DocBot")
    st.caption("Offline document Q&A · runs entirely on this machine")

    up = client.is_up()
    if up:
        st.success(f"Ollama connected · {settings.ollama_url}")
        try:
            models = client.list_models()
        except OllamaError:
            models = []
    else:
        st.error("Ollama is not running. Start it with `ollama serve`.")
        models = []

    chat_models = [m for m in models if "embed" not in m] or [settings.chat_model]
    default_idx = chat_models.index(settings.chat_model) if settings.chat_model in chat_models else 0
    chat_model = st.selectbox("Chat model", chat_models, index=default_idx)
    top_k = st.slider("Chunks to retrieve", 1, 10, settings.top_k)
    cfg = replace(settings, chat_model=chat_model, top_k=top_k)

    st.divider()
    st.subheader("Documents")
    uploads = st.file_uploader(
        "Add documents", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if uploads and st.button("Save uploaded files", use_container_width=True):
        Path(settings.docs_dir).mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (Path(settings.docs_dir) / Path(f.name).name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s). Now rebuild the index.")

    docs = list_docs()
    with st.expander(f"Indexed folder · {len(docs)} file(s)"):
        for p in docs:
            st.write(f"• {p.name}")

    if st.button("🔄 Rebuild index", type="primary", use_container_width=True, disabled=not up):
        with st.status("Building index...", expanded=True) as status:
            try:
                build_index(cfg, client, progress=st.write)
                st.session_state.bot = None
                status.update(label="Index ready", state="complete")
            except Exception as exc:
                status.update(label="Indexing failed", state="error")
                st.error(str(exc))

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- main
st.header("Ask your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

bot = get_bot(cfg) if up else None
if up and bot is None:
    st.info("No index yet. Add documents in the sidebar and click **Rebuild index**.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, (cite, score, text) in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}] {cite}** · relevance {score:.2f}")
                    st.caption(text[:600] + ("…" if len(text) > 600 else ""))

question = st.chat_input("Ask a question about your documents", disabled=bot is None)
if question and bot:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            stream, hits = bot.ask(question, history)
            answer = st.write_stream(stream)
        except (OllamaError, ValueError) as exc:
            answer, hits = f"⚠️ {exc}", []
            st.error(answer)
        sources = [(h.chunk.citation, h.score, h.chunk.text) for h in hits]
        if sources:
            with st.expander("Sources"):
                for i, (cite, score, text) in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {cite}** · relevance {score:.2f}")
                    st.caption(text[:600] + ("…" if len(text) > 600 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
