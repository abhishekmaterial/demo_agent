"""Agent Deployment Demo — Streamlit + LangGraph + LangSmith.

Live-session app that demonstrates four production patterns in one file:

1. A LangGraph ReAct agent with tool calling (agent ⇄ tools loop).
2. Multi-turn memory via an in-memory checkpointer keyed by ``thread_id``.
3. Token-level streaming into Streamlit with ``graph.stream()`` + ``st.write_stream()``.
4. Optional LangSmith tracing, enabled automatically from secrets / env vars.

Designed for Streamlit Community Cloud. Secrets are read from ``st.secrets``
in production and fall back to ``os.getenv`` for local development.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

# MemorySaver is still the name used in most teaching material. LangGraph 1.x
# renamed the class to InMemorySaver and keeps MemorySaver as an alias.
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover - older/newer package layouts
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver


# ---------------------------------------------------------------------------
# Page config MUST be the first Streamlit command in the script.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🤖 Agent Deployment Demo",
    page_icon="⚡",
    layout="centered",
)

LANGSMITH_PROJECT = "deploy-demo-session"
MODEL_NAME = "gpt-4o-mini"


# ===========================================================================
# 1. Secrets & LangSmith observability
# ===========================================================================
def _read_secret(key: str) -> str | None:
    """Read a value from Streamlit secrets, then process env, then nothing.

    ``st.secrets`` raises if no secrets file / Cloud secret store exists, which
    is the normal local-dev case before ``.streamlit/secrets.toml`` is created.
    """
    try:
        value = st.secrets.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.getenv(key)
    return value.strip() if value else None


def configure_environment() -> dict[str, Any]:
    """Load API keys and enable LangSmith tracing when a key is present.

    Returns a small status dict used by the sidebar. LangChain / LangSmith
    clients read these variables at invoke-time, so we set them on every
    Streamlit rerun *before* the agent is called.
    """
    openai_key = _read_secret("OPENAI_API_KEY")
    langsmith_key = _read_secret("LANGCHAIN_API_KEY") or _read_secret("LANGSMITH_API_KEY")
    project = _read_secret("LANGCHAIN_PROJECT") or _read_secret("LANGSMITH_PROJECT") or LANGSMITH_PROJECT

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    tracing_on = bool(langsmith_key)
    if tracing_on:
        # Accept either historical LANGCHAIN_* names or current LANGSMITH_* names.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    else:
        # Do not leak a stale tracing flag from the host environment.
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    return {
        "openai_ok": bool(openai_key),
        "openai_key": openai_key or "",
        "langsmith_ok": tracing_on,
        "langsmith_project": project if tracing_on else None,
    }


ENV = configure_environment()


# ===========================================================================
# 2. Tools — kept deliberately small so the live session stays focused
# ===========================================================================
@tool
def get_system_time() -> str:
    """Return the current date and time in UTC and the host's local timezone.

    Call this whenever the user asks what time or day it is, or needs a
    timestamp. Do not guess the time from training data.
    """
    utc_now = datetime.now(timezone.utc)
    local_now = datetime.now().astimezone()
    return (
        f"UTC: {utc_now.isoformat(timespec='seconds')}\n"
        f"Local: {local_now.isoformat(timespec='seconds')} ({local_now.tzname()})"
    )


# Tiny in-memory "vector store" for the demo. A real app would replace this
# with a retriever over Chroma / OpenSearch / a vendor RAG API.
_PRODUCT_DOCS: list[dict[str, str]] = [
    {
        "id": "checkpointer",
        "title": "LangGraph checkpointers & thread_id",
        "body": (
            "Compile the graph with a checkpointer (MemorySaver in this demo, "
            "PostgresSaver in production). Pass config={'configurable': "
            "{'thread_id': '<id>'}} on every invoke/stream call. The "
            "checkpointer stores graph state per thread, so you only send the "
            "NEW user message each turn — not the full chat history."
        ),
    },
    {
        "id": "streaming",
        "title": "Token streaming to Streamlit",
        "body": (
            "Call graph.stream(..., stream_mode='messages'). Each chunk is an "
            "(AIMessageChunk, metadata) pair. Filter to the agent node, yield "
            "chunk.content strings, and pass that generator to st.write_stream() "
            "so tokens render in the chat bubble as they arrive."
        ),
    },
    {
        "id": "langsmith",
        "title": "LangSmith tracing on Streamlit Cloud",
        "body": (
            "Set LANGCHAIN_TRACING_V2=true plus LANGCHAIN_API_KEY (or "
            "LANGSMITH_API_KEY) in Streamlit secrets. Traces land in the "
            "LangSmith project named by LANGCHAIN_PROJECT. Each graph.stream() "
            "call becomes a run you can inspect: prompts, tool calls, tokens, latency."
        ),
    },
    {
        "id": "streamlit-cloud",
        "title": "Streamlit Community Cloud deployment",
        "body": (
            "Push app.py, requirements.txt, and (optionally) .streamlit/config.toml "
            "to GitHub. In the Cloud UI, paste secrets — never commit them. "
            "The process is a single replica: MemorySaver state survives Streamlit "
            "reruns via @st.cache_resource, but is lost on reboot. Use a durable "
            "checkpointer for real users."
        ),
    },
    {
        "id": "tools",
        "title": "Tool calling in a StateGraph",
        "body": (
            "Bind tools to the chat model, then route with a conditional edge: "
            "if the last AIMessage has tool_calls, go to the tools node; otherwise "
            "END. The tools node executes each call and appends ToolMessage(s). "
            "An edge from tools back to agent closes the ReAct loop."
        ),
    },
]


@tool
def search_product_docs(query: str) -> str:
    """Mock vector search over internal product documentation.

    Use this for questions about LangGraph, Streamlit deployment, LangSmith
    tracing, checkpointers, thread_id memory, or tool calling. Pass a short
    natural-language query.
    """
    q = (query or "").strip().lower()
    if not q:
        return "No query provided. Ask about memory, streaming, LangSmith, or Streamlit Cloud."

    tokens = {t for t in q.replace("-", " ").split() if len(t) > 2}

    scored: list[tuple[int, dict[str, str]]] = []
    for doc in _PRODUCT_DOCS:
        haystack = f"{doc['title']} {doc['body']}".lower()
        score = sum(1 for t in tokens if t in haystack)
        # Light phrase boosts so the demo "just works" during a live session.
        if "thread" in q and "thread_id" in haystack:
            score += 2
        if "trace" in q or "langsmith" in q:
            if "langsmith" in haystack:
                score += 2
        if score:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:2] or [(1, _PRODUCT_DOCS[0])]

    blocks = []
    for rank, (score, doc) in enumerate(top, start=1):
        blocks.append(
            f"[{rank}] {doc['title']} (score={score}, id={doc['id']})\n{doc['body']}"
        )
    return "\n\n".join(blocks)


TOOLS = [get_system_time, search_product_docs]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a concise principal engineer co-hosting a live demo of agent deployment.

You have two tools:
- get_system_time: authoritative clock. Always call it for time/date questions.
- search_product_docs: mock doc search. Use it for questions about this stack
  (LangGraph, checkpointers, streaming, Streamlit Cloud, LangSmith).

Rules:
- Prefer tools over guessing.
- After a tool returns, answer in 2–6 short sentences. Mention the tool you used.
- If the user is testing memory, recall facts they told you earlier in this thread.
"""


# ===========================================================================
# 3. LangGraph agent (cached for the lifetime of the Streamlit process)
# ===========================================================================
def _with_system(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Prepend the system prompt for the LLM call without writing it into state.

    The checkpointer already stores the conversation. If we persisted a new
    SystemMessage on every turn, the thread would accumulate duplicates.
    """
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT), *messages]


def _route_after_agent(state: MessagesState) -> str:
    """Conditional edge: run tools when the model requested them, else stop."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def _run_tools(state: MessagesState) -> dict[str, list[ToolMessage]]:
    """Execute every tool call on the last AI message and return ToolMessages."""
    last = state["messages"][-1]
    results: list[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        tool_fn = TOOL_MAP.get(name)
        if tool_fn is None:
            payload = f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}"
        else:
            try:
                payload = tool_fn.invoke(call["args"])
            except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
                payload = f"Tool '{name}' failed: {exc}"
        results.append(
            ToolMessage(content=str(payload), tool_call_id=call["id"], name=name)
        )
    return {"messages": results}


@st.cache_resource(show_spinner="Bootstrapping LangGraph agent…")
def build_agent(openai_api_key: str):
    """Compile the graph once and reuse it across Streamlit reruns.

    Caching is what makes MemorySaver actually persist in this app: Streamlit
    re-executes the script on every widget interaction. Without
    ``@st.cache_resource`` a new empty checkpointer would be created each time
    and ``thread_id`` memory would appear broken.
    """
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required to build the agent.")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=openai_api_key,
        temperature=0.2,
        streaming=True,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict[str, list[AIMessage]]:
        response = llm_with_tools.invoke(_with_system(state["messages"]))
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", _run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent)
    graph.add_edge("tools", "agent")

    # In-memory checkpointer: perfect for a live demo, not durable across
    # process restarts. Production: PostgresSaver / a LangGraph Platform thread.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ===========================================================================
# 4. Streaming helpers — adapt LangGraph chunks into strings for Streamlit
# ===========================================================================
def _unpack_message_chunk(chunk: Any) -> tuple[Any, dict[str, Any]]:
    """Normalize LangGraph v1 tuples and v2 StreamPart dicts to (message, meta)."""
    if isinstance(chunk, dict) and chunk.get("type") == "messages":
        data = chunk.get("data")
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1] or {}
        return data, {}
    if isinstance(chunk, tuple) and len(chunk) == 2:
        message, metadata = chunk
        return message, metadata or {}
    return chunk, {}


def _message_text(message: Any) -> str:
    """Extract plain text from an AIMessage / AIMessageChunk."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        return "".join(parts)
    return ""


def _tool_names(message: Any) -> list[str]:
    """Collect tool names from complete tool_calls or streamed tool_call_chunks."""
    names: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name:
            names.append(name)
    for call in getattr(message, "tool_call_chunks", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name:
            names.append(name)
    return names


def iter_assistant_tokens(
    graph,
    user_text: str,
    thread_id: str,
    tool_log: list[str],
) -> Iterator[str]:
    """Yield assistant tokens for ``st.write_stream``.

    Only the *new* HumanMessage is sent. Prior turns live in MemorySaver under
    this ``thread_id`` — that is the persistence story to highlight on stage.
    """
    seen: set[str] = set()
    stream = graph.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 12,
        },
        stream_mode="messages",
    )
    for chunk in stream:
        message, metadata = _unpack_message_chunk(chunk)
        node = str(metadata.get("langgraph_node", "") or "")

        for name in _tool_names(message):
            if name not in seen:
                seen.add(name)
                tool_log.append(name)

        # Skip ToolMessage payloads (raw doc dumps) — only stream model prose.
        if node == "tools" or isinstance(message, ToolMessage):
            continue

        text = _message_text(message)
        if text:
            yield text


# ===========================================================================
# 5. Streamlit UI
# ===========================================================================
def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _reset_conversation() -> None:
    """Mint a new thread_id so the checkpointer starts a blank transcript."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def _render_sidebar() -> None:
    st.sidebar.header("Deployment status")

    openai_label = "connected" if ENV["openai_ok"] else "missing OPENAI_API_KEY"
    langsmith_label = (
        f"tracing → `{ENV['langsmith_project']}`"
        if ENV["langsmith_ok"]
        else "idle (no API key)"
    )
    st.sidebar.markdown(f"{'🟢' if ENV['openai_ok'] else '🔴'} **OpenAI** — {openai_label}")
    st.sidebar.markdown(
        f"{'🟢' if ENV['langsmith_ok'] else '⚪'} **LangSmith** — {langsmith_label}"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Active thread_id")
    st.sidebar.caption(
        "LangGraph stores checkpointed state under this id. Reset mints a new "
        "thread so the agent forgets the conversation."
    )
    st.sidebar.code(st.session_state.thread_id, language=None)

    if st.sidebar.button("Reset conversation", use_container_width=True):
        _reset_conversation()
        st.rerun()

    with st.sidebar.expander("Architecture behind the scene", expanded=False):
        st.markdown(
            """
**Request path**

```
User message
  → Streamlit  (session_state.messages = UI history)
  → LangGraph StateGraph
        agent  --tool_calls?--> tools ↩
           \\-- no tools --> END
  → MemorySaver[thread_id]   (true multi-turn memory)
  → LangSmith                (optional trace)
```

**Why `@st.cache_resource`?** Streamlit reruns this file on every interaction.
The compiled graph *and* its `MemorySaver` must survive those reruns, otherwise
`thread_id` would point at a brand-new empty checkpointer.

**What is *not* persisted?** In-memory checkpoints die when the Cloud app
sleeps or reboots. Production swaps `MemorySaver` for `PostgresSaver`.

**Two histories, on purpose**
- `st.session_state.messages` — what you *see*
- checkpointer[thread_id] — what the *agent* remembers

We send only the latest user turn into `graph.stream()`. The graph already
has the rest.
            """
        )


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            tools_used = message.get("tools") or []
            if tools_used:
                st.caption("Tools used: " + ", ".join(f"`{name}`" for name in tools_used))


def main() -> None:
    _init_session()
    _render_sidebar()

    st.title("🤖 Agent Deployment Demo")
    st.caption(
        f"LangGraph tool-calling agent · `{MODEL_NAME}` · streamed tokens · "
        "checkpointer memory · optional LangSmith traces"
    )

    if not ENV["openai_ok"]:
        st.error(
            "No OpenAI key found. For Streamlit Community Cloud, add "
            "`OPENAI_API_KEY` under **App settings → Secrets**. For local runs, "
            "copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
            "or export the variable in your shell."
        )
        st.stop()

    graph = build_agent(ENV["openai_key"])
    _render_history()

    prompt = st.chat_input("Ask about time, memory, tracing, or how this app is deployed…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tool_log: list[str] = []
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                iter_assistant_tokens(
                    graph,
                    user_text=prompt,
                    thread_id=st.session_state.thread_id,
                    tool_log=tool_log,
                )
            )
        except Exception as exc:  # noqa: BLE001 - show a clean demo failure, not a traceback
            answer = ""
            st.error(f"Agent call failed: {exc}")

        if tool_log:
            st.caption("Tools used: " + ", ".join(f"`{name}`" for name in tool_log))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer or "_(no text returned)_",
            "tools": tool_log,
        }
    )


main()
