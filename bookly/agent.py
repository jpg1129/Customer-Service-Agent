"""
Bookly Support Agent — LangGraph ReAct agent.

Implements the core agent loop:
  START -> LLM (reason + decide) -> tool call? -> Tool Executor -> LLM (reflect) -> ...
                                  -> no tool   -> END (respond to customer)

Built with StateGraph using:
  - An "agent" node: LLM with bound tools decides to respond or call a tool
  - A "tools" node: executes tool calls and returns results to the agent
  - Conditional edges: tools_condition routes based on whether the LLM
    requested a tool call
  - InMemorySaver: conversation persistence across turns
"""

from __future__ import annotations

import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from bookly.prompts import SYSTEM_PROMPT
from bookly.tools import ALL_TOOLS

load_dotenv()

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(
    model_name: str = "gpt-4o",
    temperature: float = 0.3,
) -> tuple[CompiledStateGraph, InMemorySaver]:
    """Build the LangGraph agent and return (graph, checkpointer).

    Args:
        model_name: OpenAI model to use.
        temperature: Lower = more deterministic. 0.3 balances reliability
                     with natural-sounding responses for customer support.

    Returns:
        (compiled_graph, checkpointer) tuple.
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # -- Nodes ---------------------------------------------------------------

    async def agent_node(state: MessagesState) -> dict:
        """LLM node: prepend system prompt, invoke LLM with bound tools."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools=ALL_TOOLS)

    # -- Graph ---------------------------------------------------------------

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    checkpointer = InMemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph, checkpointer


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def get_config(thread_id: str | None = None) -> dict:
    """Build a LangGraph config dict with a thread ID for conversation memory.

    Each unique thread_id maintains its own conversation history via the
    InMemorySaver checkpointer. Pass the same thread_id across turns to
    preserve multi-turn context.
    """
    if thread_id is None:
        thread_id = uuid.uuid4().hex
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15,
    }


async def chat(graph, config: dict, user_message: str) -> str:
    """Send a message and return the agent's text response.

    Args:
        graph: Compiled LangGraph agent from build_agent().
        config: Config dict from get_config() with thread_id.
        user_message: The customer's message.

    Returns:
        The agent's final text response as a string.
    """
    inputs = {"messages": [HumanMessage(content=user_message)]}
    response = await graph.ainvoke(inputs, config)
    ai_messages = [
        m for m in response["messages"] if isinstance(m, AIMessage) and m.content
    ]
    if ai_messages:
        return ai_messages[-1].content
    return "I'm sorry, I wasn't able to process that. Please try again."


async def stream_response(graph, config: dict, user_message: str):
    """Stream the agent's response token-by-token.

    Yields dicts with keys:
      - {"token": str}       — a chunk of the agent's text response
      - {"tool_call": dict}  — a tool invocation (name + args)
      - {"tool_result": str} — the tool's return value

    This powers both the Streamlit UI streaming and the tool transparency panel.
    """
    inputs = {"messages": [HumanMessage(content=user_message)]}

    async for event in graph.astream_events(inputs, config, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield {"token": chunk.content}

        elif kind == "on_tool_start":
            yield {
                "tool_call": {
                    "name": event["name"],
                    "args": event["data"].get("input", {}),
                }
            }

        elif kind == "on_tool_end":
            yield {"tool_result": str(event["data"].get("output", ""))}


# ---------------------------------------------------------------------------
# CLI for quick testing (python -m bookly.agent)
# ---------------------------------------------------------------------------

async def _cli_loop() -> None:
    """Interactive CLI for testing the agent without Streamlit."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        return

    graph, _ = build_agent()
    config = get_config(thread_id="cli-session")

    print("Bookly Support Agent (type 'quit' to exit)")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        print("Agent: ", end="", flush=True)
        async for event in stream_response(graph, config, user_input):
            if "tool_call" in event:
                tc = event["tool_call"]
                print(f"\n  [Tool: {tc['name']}({tc['args']})]", flush=True)
            elif "token" in event:
                print(event["token"], end="", flush=True)
        print("\n")


if __name__ == "__main__":
    asyncio.run(_cli_loop())
