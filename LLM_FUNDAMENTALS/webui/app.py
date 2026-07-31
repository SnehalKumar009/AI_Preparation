"""Study Buddy — Streamlit capstone.

Run from the project root:
    streamlit run webui/app.py

Combines RAG grounding, tool calling, streaming, provider switching, and a
live USD cost meter — all on the shared study_buddy core.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run via `streamlit run webui/app.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from study_buddy import CostTracker, RagStore, get_provider
from study_buddy.providers import available_providers
from study_buddy.tools import TOOL_SCHEMAS, dispatch

st.set_page_config(page_title="Study Buddy", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Indexing notes...")
def build_store() -> RagStore:
    store = RagStore()
    store.add_directory()
    return store


def get_tracker() -> CostTracker:
    if "tracker" not in st.session_state:
        st.session_state.tracker = CostTracker()
    return st.session_state.tracker


# ---- Sidebar: provider switch + live cost meter ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider_name = st.selectbox("Provider", available_providers(), index=0)
    use_rag = st.checkbox("Ground answers in notes (RAG)", value=True)
    use_tools = st.checkbox("Enable tools (calc, time)", value=True)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)

    st.divider()
    st.subheader("💸 Session cost")
    tracker = get_tracker()
    st.metric("Total USD", f"${tracker.total_cost:.6f}")
    st.metric("Total tokens", tracker.total_tokens)
    if tracker.records:
        st.code(tracker.table(), language="text")

st.title("🎓 Study Buddy")
st.caption(f"Provider: **{provider_name}** · RAG: {use_rag} · Tools: {use_tools}")

# ---- Chat history ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


def build_prompt(question: str) -> list[dict]:
    system = "You are Study Buddy, a concise tutor. Use tools for math/time when needed."
    if use_rag:
        context = build_store().context_for(question, k=3)
        system += " Prefer the provided context and cite [source] when you use it."
        user = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user = question
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


if question := st.chat_input("Ask me anything about your notes..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    llm = get_provider(provider_name)
    prompt = build_prompt(question)

    # Optional tool step (non-streamed so we can capture tool calls).
    if use_tools:
        first = llm.chat(prompt, tools=TOOL_SCHEMAS, temperature=temperature)
        tracker.add(first)
        if first.tool_calls:
            prompt.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in first.tool_calls
                ],
            })
            for c in first.tool_calls:
                obs = dispatch(c["name"], c["arguments"])
                prompt.append({"role": "tool", "content": obs, "name": c["name"]})
                st.toast(f"🔧 {c['name']}({c['arguments']}) = {obs}")

    # Stream the final answer live.
    with st.chat_message("assistant"):
        answer = st.write_stream(llm.stream(prompt, temperature=temperature))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()  # refresh the sidebar cost meter
