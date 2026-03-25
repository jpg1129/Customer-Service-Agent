"""
Bookly Support Agent — Streamlit web interface.

Provides a chat UI with:
  - Streaming token-by-token responses
  - Tool call transparency sidebar showing agent actions in real time
  - Session state persistence for multi-turn conversations
  - New conversation button to reset context

Run with: uv run streamlit run bookly/app.py
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import os
import re

import streamlit as st
from dotenv import load_dotenv

from bookly.agent import build_agent, get_config, stream_response

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bookly Support",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Clean up default Streamlit padding */
    .block-container { padding-top: 2rem; max-width: 900px; }

    /* Chat message styling */
    .stChatMessage { border-radius: 12px; }

    /* Sidebar tool call styling */
    .tool-call-box {
        background-color: rgba(74, 144, 217, 0.15);
        border-left: 3px solid #4A90D9;
        padding: 10px 12px;
        margin-bottom: 4px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9em;
        color: #e0e0e0;
    }
    .tool-call-box strong {
        color: #93c5fd;
    }
    .tool-call-box code {
        color: #a5d6a7 !important;
        background-color: rgba(0, 0, 0, 0.3) !important;
        font-size: 0.85em;
        padding: 2px 5px;
        border-radius: 3px;
    }

    /* Sidebar result block — word-wraps and scrolls vertically */
    .tool-result-block {
        color: #d4d4d4;
        background-color: rgba(0, 0, 0, 0.35);
        font-family: 'Source Code Pro', 'Menlo', 'Consolas', monospace;
        font-size: 0.8em;
        line-height: 1.5;
        padding: 10px 12px;
        border-radius: 6px;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    /* JSON-style syntax highlighting inside result blocks */
    .tool-result-block .json-key { color: #9cdcfe; }
    .tool-result-block .json-str { color: #ce9178; }
    .tool-result-block .json-num { color: #b5cea8; }
    .tool-result-block .json-bool { color: #569cd6; }
    .tool-result-block .json-null { color: #569cd6; }

    /* Header styling */
    .bookly-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "**OPENAI_API_KEY not set.** Copy `.env.example` to `.env` and add your key.\n\n"
            "```bash\ncp .env.example .env\n# Edit .env and add your OpenAI API key\n```"
        )
        st.stop()

    graph, _ = build_agent()
    st.session_state.agent = graph
    st.session_state.config = get_config(thread_id="streamlit-session")
    st.session_state.messages = []
    st.session_state.tool_log = []  # sidebar tool call history


def _escape_dollars(text: str) -> str:
    """Escape bare $ signs so Streamlit doesn't render them as LaTeX."""
    return text.replace("$", "\\$")


def _format_tool_result(raw: str) -> str:
    """Extract the content from a raw tool result and pretty-format it.

    LangChain ToolMessages stringify as:
      content='...' name='tool' tool_call_id='call_...'
    We extract just the content value. If the content is JSON-like (dict),
    we pretty-print it. Otherwise we unescape \\n to real newlines.
    """
    # Try to extract the content='...' or content="..." portion
    match = re.search(r"content=['\"](.+?)['\"]\s+name=", raw, re.DOTALL)
    text = match.group(1) if match else raw

    # Try to parse as JSON for pretty-printing
    try:
        parsed = json.loads(text.replace("'", '"'))
        return json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try parsing as a Python dict literal
    try:
        import ast
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, indent=2)
    except (ValueError, SyntaxError):
        pass

    # Plain text — unescape literal \n to real newlines
    return text.replace("\\n", "\n")


def _highlight_result(text: str) -> str:
    """Apply VS Code-style syntax highlighting to a formatted tool result.

    Handles JSON output (keys, strings, numbers, booleans, null) and
    falls back to plain escaped HTML for non-JSON text.
    """
    escaped = _html.escape(text)

    # JSON keys: "key":
    escaped = re.sub(
        r'(&quot;)([\w_]+)(&quot;)(\s*:)',
        r'<span class="json-key">\1\2\3</span>\4',
        escaped,
    )
    # JSON string values: "value" (after a colon or in arrays)
    escaped = re.sub(
        r':\s*(&quot;)(.*?)(&quot;)',
        r': <span class="json-str">\1\2\3</span>',
        escaped,
    )
    # Numbers
    escaped = re.sub(
        r'(?<=: )(-?\d+\.?\d*)',
        r'<span class="json-num">\1</span>',
        escaped,
    )
    # Booleans
    escaped = re.sub(
        r'\b(true|false)\b',
        r'<span class="json-bool">\1</span>',
        escaped,
    )
    # Null
    escaped = re.sub(
        r'\bnull\b',
        r'<span class="json-null">null</span>',
        escaped,
    )
    return escaped


def reset_conversation() -> None:
    """Reset conversation state for a fresh session."""
    graph, _ = build_agent()
    st.session_state.agent = graph
    st.session_state.config = get_config(thread_id="streamlit-session-new")
    st.session_state.messages = []
    st.session_state.tool_log = []


# ---------------------------------------------------------------------------
# Sidebar — tool call transparency + controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Bookly Support Agent")
    st.caption(
        "AI-powered customer service agent for Bookly, an online bookstore. "
        "Built with LangGraph and GPT-4o, featuring a ReAct reasoning loop, "
        "tool-calling transparency, and policy-grounded responses."
    )
    st.divider()

    if st.button("New Conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()
    st.markdown("## Agent Activity")
    st.caption("Real-time visibility into tool calls and reasoning.")

    # Display tool call history
    if st.session_state.tool_log:
        for entry in st.session_state.tool_log:
            if entry["type"] == "tool_call":
                st.markdown(
                    f'<div class="tool-call-box">'
                    f'<strong>🔧 {entry["name"]}</strong><br>'
                    f'<code>{entry["args"]}</code>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif entry["type"] == "tool_result":
                with st.expander("View result", expanded=False):
                    formatted = _format_tool_result(str(entry["result"]))
                    highlighted = _highlight_result(formatted)
                    st.markdown(
                        f'<div class="tool-result-block">{highlighted}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.caption("Tool calls will appear here as the agent works.")

    st.divider()
    st.markdown("**Try these queries:**")
    _sample_queries = [
        "What's the status of order #1002?",
        "I'd like to return order #1003",
        "My book from order #1004 arrived damaged",
        "What are your shipping options?",
        "How do I reset my password?",
        "Can I get a refund on an ebook?",
        "What happened to order #1005?",
    ]
    for _q in _sample_queries:
        if st.button(_q, key=f"sample_{_q}", use_container_width=True):
            st.session_state.pending_query = _q

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="bookly-header">'
    "<h1>📚 Bookly Support</h1>"
    "<p>Hi! I'm your Bookly support agent. Ask me about orders, returns, shipping, or account help.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="📚" if msg["role"] == "assistant" else "👤"):
        st.markdown(_escape_dollars(msg["content"]))

# ---------------------------------------------------------------------------
# Handle new user input
# ---------------------------------------------------------------------------

# Check for a pending query from sidebar sample buttons
prompt = st.chat_input("Ask about your order, returns, shipping...")
if prompt is None and "pending_query" in st.session_state:
    prompt = st.session_state.pop("pending_query")
elif "pending_query" in st.session_state:
    del st.session_state["pending_query"]

if prompt:
    # Display and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Stream the agent response
    with st.chat_message("assistant", avatar="📚"):
        message_placeholder = st.empty()
        # Use mutable containers so the async function can update them
        state = {"response": ""}
        turn_tool_calls: list[dict] = []

        async def run_stream() -> None:
            """Run the async stream and collect events."""
            graph = st.session_state.agent
            config = st.session_state.config

            async for event in stream_response(graph, config, prompt):
                if "token" in event:
                    state["response"] += event["token"]
                    message_placeholder.markdown(_escape_dollars(state["response"]) + "▌")

                elif "tool_call" in event:
                    tc = event["tool_call"]
                    turn_tool_calls.append({
                        "type": "tool_call",
                        "name": tc["name"],
                        "args": str(tc["args"]),
                    })

                elif "tool_result" in event:
                    turn_tool_calls.append({
                        "type": "tool_result",
                        "result": event["tool_result"],
                    })

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_stream())
        finally:
            loop.close()

        # Remove cursor and show final response
        message_placeholder.markdown(_escape_dollars(state["response"]))

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": state["response"]})

    # Update tool log in sidebar
    st.session_state.tool_log.extend(turn_tool_calls)

    # Rerun to update sidebar with new tool calls
    if turn_tool_calls:
        st.rerun()
