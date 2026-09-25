# Agent Deployment Demo

Streamlit app for a live session on **deploying a LangGraph agent**: tool calling, multi-turn memory, token streaming, and LangSmith tracing.

The LLM is Groq (`openai/gpt-oss-20b`) via Groq’s OpenAI-compatible API. No OpenAI key is required.

## What it demonstrates

| Pattern | How |
| --- | --- |
| Tool-calling agent | `StateGraph(MessagesState)` ReAct loop (`agent ⇄ tools`) |
| Multi-turn memory | `MemorySaver` checkpointer keyed by `thread_id` |
| Token streaming | `graph.stream(stream_mode="messages")` → `st.write_stream()` |
| Observability | LangSmith tracing when `LANGSMITH_KEY` is set |

Tools:

- `get_system_time()` — current UTC / local time
- `search_product_docs(query)` — mock doc search for this stack

## Repo files

| File | Upload to GitHub? |
| --- | --- |
| `app.py` | Yes |
| `requirements.txt` | Yes |
| `README.md` | Yes |
| `.env` | **Never** |
| `.streamlit/secrets.toml` | **Never** |

## Local run

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the same folder as `app.py`:

```bash
GROQ_KEY=gsk_...
LANGSMITH_KEY=lsv2_...
```

Get keys from [Groq Console](https://console.groq.com/keys) and [LangSmith](https://smith.langchain.com). LangSmith is optional; omit `LANGSMITH_KEY` to run without tracing.

```bash
streamlit run app.py
```

The sidebar should show **Groq connected**. If LangSmith is configured it shows **tracing → `deploy-demo-session`**.

Try:

- “What time is it?”
- “How does thread_id memory work?”
- “My name is Alex. Remember that.” then “What’s my name?”
- **Reset conversation** — new `thread_id`, memory cleared

Traces: [smith.langchain.com](https://smith.langchain.com) → project **deploy-demo-session**.

## Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and this README to GitHub.
2. Create the app from `app.py`.
3. **App settings → Secrets** — paste:

```toml
GROQ_KEY = "gsk_..."
LANGSMITH_KEY = "lsv2_..."
```

Cloud cannot read your local `.env`. After saving secrets, reboot the app.

Optional:

```toml
GROQ_MODEL = "openai/gpt-oss-20b"
LANGSMITH_PROJECT = "deploy-demo-session"
```

## Architecture (short)

```
User message
  → Streamlit          (session_state.messages = UI history)
  → LangGraph          agent --tool_calls?--> tools ↩
  → MemorySaver[thread_id]
  → Groq               https://api.groq.com/openai/v1
  → LangSmith          optional trace
```

`@st.cache_resource` keeps the compiled graph and checkpointer alive across Streamlit reruns. Only the latest user turn is sent into `graph.stream()`; prior turns live in the checkpointer.

`MemorySaver` is in-process RAM. It survives reruns, not Cloud sleep/reboot. Production would swap in `PostgresSaver`.
