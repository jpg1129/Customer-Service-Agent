"""
Bookly Support Agent — LangChain tool definitions.

Each tool wraps a helper function from mock_data.py and provides a rich
docstring that serves as prompt engineering for the LLM (Principle E:
"Tool descriptions are prompt engineering" — Anthropic).

Docstrings are 3-4+ sentences and include: what, when, parameters,
edge cases, and caveats per Anthropic's tool authoring guide.
"""

from __future__ import annotations

from langchain_core.tools import tool

from bookly.mock_data import check_eligibility, create_return, get_order, search_policies


@tool
def lookup_order(order_id: str) -> dict:
    """Retrieves complete order details from the Bookly database.

    Use this tool when a customer asks about their order status, tracking
    information, delivery date, or order contents. The order_id should be
    a 4-digit number (e.g., "1002"). Returns order status, items with
    prices, shipping details, and tracking information when available.

    If the order is not found, returns an error message — relay this to
    the customer and ask them to verify their order number.

    This is a read-only lookup with no side effects.
    """
    return get_order(order_id)


@tool
def check_return_eligibility(order_id: str, reason: str) -> dict:
    """Checks whether items in a delivered order are eligible for return.

    Use this tool AFTER you have already looked up the customer's order
    with lookup_order. The reason should describe why the customer wants
    to return (e.g., "customer_request", "damaged", "wrong item").

    Returns per-item eligibility: each item is marked eligible or
    ineligible with a specific explanation. Digital items are always
    non-refundable. Physical items outside the 30-day window are
    ineligible unless the reason is "damaged" (no time limit).

    If the order has not been delivered yet or was already cancelled,
    returns an explanation of why a return cannot be processed.

    This check does not initiate a return or issue a refund. It records
    internally that eligibility was verified, which is required before
    initiate_return can be called.
    """
    return check_eligibility(order_id, reason)


@tool
def initiate_return(order_id: str, items: list[str], refund_method: str) -> dict:
    """Initiates an irreversible return and refund for specific items in an order.

    WARNING: This action cannot be undone. Only call this tool AFTER all
    three conditions are met:
      1. You called check_return_eligibility and confirmed the items are eligible.
      2. You presented the return summary (items, refund amount, refund method)
         to the customer.
      3. The customer explicitly confirmed they want to proceed (e.g., "yes",
         "go ahead", "please proceed").

    Args:
        order_id: The 4-digit order number (e.g., "1003").
        items: List of exact book titles to return (e.g., ["Atomic Habits"]).
               Titles must match the order exactly.
        refund_method: Either "original_payment" or "store_credit".

    Returns a confirmation with return ID, shipping label status, and
    refund processing timeline. If inputs are invalid (wrong order ID,
    items not in order, bad refund method), returns an actionable error.
    """
    return create_return(order_id, items, refund_method)


@tool
def search_policy(query: str) -> str:
    """Searches Bookly's policy documentation for relevant information.

    Use this tool when a customer asks general questions about shipping,
    returns, refunds, or account management that are not tied to a
    specific order. Pass the customer's question or key terms as the
    query (e.g., "shipping options", "return window", "password reset").

    Always use this tool to retrieve specific policy details (prices,
    timeframes, rules) rather than stating them from memory. The system
    prompt contains only a high-level summary — this tool has the
    complete, authoritative policy text.

    Returns the most relevant policy section(s) as formatted text, or a
    fallback message directing the customer to human support if no match
    is found. This is a read-only lookup with no side effects.
    """
    return search_policies(query)


# Collect all tools for binding to the LLM
ALL_TOOLS = [lookup_order, check_return_eligibility, initiate_return, search_policy]
