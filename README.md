# Bookly Customer Support Agent

An AI-powered customer support agent for **Bookly**, a fictional online bookstore. Built with LangGraph and GPT-4o, featuring a multi-step return workflows with confirmation gates, and a Streamlit web interface with real-time tool call transparency.

> See [DESIGN_DOC.md](./DESIGN_DOC.md) for architecture decisions, conversation design rationale, and production readiness analysis.

## Quick Start

**Prerequisites:** Python 3.12+, an [OpenAI API key](https://platform.openai.com/api-keys)

```bash
# 1. Clone the repo
git clone <repo-url> && cd AI-Agent

# 2. Set up environment and install (pick one)

# Option A: pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Option B: uv (creates .venv automatically)
uv sync

# 3. Configure your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Launch the web UI
streamlit run bookly/app.py        # if using pip
uv run streamlit run bookly/app.py # if using uv
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

A CLI interface is also available for quick testing:

```bash
python -m bookly.agent
```

## What It Does

The agent handles three core customer support workflows:

| Workflow             | What Happens                                                         | Key Capability                                                        |
| -------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Order status**     | Customer asks about an order, agent looks it up and presents details | Multi-turn (asks for order number if missing), tool use               |
| **Return/refund**    | Agent checks eligibility, presents options, confirms, then initiates | Multi-step workflow with confirmation gate before irreversible action |
| **Policy questions** | Shipping, returns, account help, password reset                      | Policy-grounded responses from embedded text, not LLM training data   |

The agent also handles edge cases: expired return windows, non-refundable digital items, cancelled orders, damaged item exceptions, out-of-scope requests, and prompt injection attempts.

## Example Queries to Try

| Query                                      | What It Demonstrates                                              |
| ------------------------------------------ | ----------------------------------------------------------------- |
| "Where's my order?"                        | Clarifying question -- asks for order number before acting        |
| "What's the status of order #1002?"        | Tool use -- looks up order and formats tracking info              |
| "I'd like to return order #1003"           | Full return flow -- eligibility check, options, confirmation gate |
| "My book from order #1004 arrived damaged" | Damaged item bypass -- eligible despite expired return window     |
| "Can I get a refund on an ebook?"          | Policy grounding -- explains digital items are non-refundable     |
| "What are your shipping options?"          | Policy search -- retrieves shipping tiers and prices              |
| "How do I reset my password?"              | Account help -- directs to the correct URL                        |
| "What happened to order #1005?"            | Edge case -- explains cancelled order was already refunded        |

Each of the five mock orders (#1001--#1005) is designed to exercise different agent behavior. See the sidebar sample queries in the web UI for a guided tour.

## Project Structure

```
bookly/
  agent.py       LangGraph ReAct agent (StateGraph, streaming, CLI)
  tools.py       4 LangChain @tool definitions with rich docstrings
  mock_data.py   Mock order database (5 orders) and helper functions
  prompts.py     System prompt, policy constants, searchable policy sections
  app.py         Streamlit web interface with streaming + tool call sidebar
tests/
  test_tools.py  46 deterministic unit tests (no API key needed)
  test_agent.py  43 integration tests -- behavioral, adversarial, policy grounding
DESIGN_DOC.md   One-page architecture & design document
```

## Architecture

```
                          User Message
                               |
                               v
+--------------------------------------------------------------+
|                    LangGraph StateGraph                       |
|                                                               |
|                    +------------------+                        |
|   START ────────>  |  "agent" node   |  <──────────────+      |
|                    |  (LLM + system  |                 |      |
|                    |   prompt)       |                 |      |
|                    +--------+--------+                 |      |
|                             |                          |      |
|                      tools_condition                   |      |
|                       /          \                     |      |
|                      /            \                    |      |
|              no tool call      tool call               |      |
|                    |               |                   |      |
|                    v               v                   |      |
|                   END      +------------------+        |      |
|                (respond)   |  "tools" node   |  ──────+      |
|                            |  (Tool Executor)|   result       |
|                            +--------+--------+                |
|                                     |                         |
|                                     v                         |
|                              Mock Database                    |
|                           (orders, policies)                  |
|                                                               |
|   Memory: InMemorySaver (conversation persistence)            |
+--------------------------------------------------------------+
```

| Tool                       | Type             | Purpose                                           |
| -------------------------- | ---------------- | ------------------------------------------------- |
| `lookup_order`             | Read-only        | Retrieve order status, tracking, and item details |
| `check_return_eligibility` | Read-only        | Per-item eligibility check with denial reasons    |
| `initiate_return`          | **Irreversible** | Execute a return and issue a refund               |
| `search_policy`            | Read-only        | Keyword search across policy documentation        |

**Key guardrail:** `initiate_return` is the only tool that modifies state. The agent must (1) check eligibility, (2) present the return summary and refund amount, and (3) receive explicit confirmation before executing. A programmatic gate (`ELIGIBILITY_VERIFIED`) enforces the prerequisite chain at the code level regardless of LLM behavior.

## Running Tests

```bash
# Unit tests -- deterministic, no API key needed (<1s)
python -m pytest tests/test_tools.py -v

# Integration tests -- requires OPENAI_API_KEY
python -m pytest tests/test_agent.py -n 8 -v

# Fast CI mode with gpt-4o-mini (skips 15 tests needing stronger reasoning)
AGENT_MODEL=gpt-4o-mini python -m pytest tests/ -n 8 -v
```

**89 total tests:** 46 unit tests covering all tool functions and edge cases, 43 integration tests covering behavioral flows, adversarial attacks (prompt injection, social engineering, logic manipulation), policy grounding, and error recovery.

## Tech Stack

| Component       | Technology                                           |
| --------------- | ---------------------------------------------------- |
| LLM             | OpenAI GPT-4o                                        |
| Agent Framework | LangGraph (StateGraph with ReAct loop)               |
| Tool Binding    | LangChain `@tool` decorator                          |
| Memory          | LangGraph `InMemorySaver` checkpointer               |
| Web Interface   | Streamlit with streaming + tool transparency sidebar |
| Package Manager | uv                                                   |
| Language        | Python 3.12+                                         |
