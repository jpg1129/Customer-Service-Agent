"""
Bookly Support Agent — Mock database and helper functions.

Five orders, each designed to force different agent behavior:
  #1001  PROCESSING  — no tracking yet (anti-hallucination test)
  #1002  SHIPPED     — happy path with tracking + multi-item
  #1003  DELIVERED   — recent delivery, eligible for return
  #1004  DELIVERED   — expired window + digital item (two denial reasons)
  #1005  CANCELLED   — already refunded, nothing to return
"""

from __future__ import annotations

from datetime import date, timedelta

from bookly.prompts import BOOKLY_POLICIES, POLICY_SECTIONS

# ---------------------------------------------------------------------------
# Reference date — all "days ago" math is relative to this.
# Set to today's date so return-window checks stay accurate for the demo.
# ---------------------------------------------------------------------------

REFERENCE_DATE: date = date.today()

# ---------------------------------------------------------------------------
# Orders database
# ---------------------------------------------------------------------------

ORDERS: dict[str, dict] = {
    "1001": {
        "order_id": "1001",
        "customer_email": "sarah@email.com",
        "status": "processing",
        "items": [
            {"title": "Dune", "format": "hardcover", "price": 18.99},
        ],
        "order_date": REFERENCE_DATE - timedelta(days=1),
        "shipping_method": "standard",
        "shipping_cost": 4.99,
        "tracking_number": None,
        "carrier": None,
        "estimated_delivery": None,
        "shipped_date": None,
        "delivered_date": None,
        "cancelled_date": None,
        "refund": None,
    },
    "1002": {
        "order_id": "1002",
        "customer_email": "mike@email.com",
        "status": "shipped",
        "items": [
            {"title": "Project Hail Mary", "format": "paperback", "price": 12.99},
            {"title": "The Martian", "format": "paperback", "price": 11.99},
        ],
        "order_date": REFERENCE_DATE - timedelta(days=5),
        "shipping_method": "standard",
        "shipping_cost": 0.00,
        "tracking_number": "1Z999AA10123456784",
        "carrier": "UPS",
        "estimated_delivery": REFERENCE_DATE + timedelta(days=3),
        "shipped_date": REFERENCE_DATE - timedelta(days=3),
        "delivered_date": None,
        "cancelled_date": None,
        "refund": None,
    },
    "1003": {
        "order_id": "1003",
        "customer_email": "emma@email.com",
        "status": "delivered",
        "items": [
            {"title": "Atomic Habits", "format": "hardcover", "price": 16.99},
        ],
        "order_date": REFERENCE_DATE - timedelta(days=10),
        "shipping_method": "express",
        "shipping_cost": 9.99,
        "tracking_number": "1Z999AA10123456785",
        "carrier": "UPS",
        "estimated_delivery": None,
        "shipped_date": REFERENCE_DATE - timedelta(days=7),
        "delivered_date": REFERENCE_DATE - timedelta(days=2),
        "cancelled_date": None,
        "refund": None,
    },
    "1004": {
        "order_id": "1004",
        "customer_email": "james@email.com",
        "status": "delivered",
        "items": [
            {"title": "Sapiens", "format": "paperback", "price": 14.99},
            {"title": "Python Crash Course", "format": "digital", "price": 29.99},
        ],
        "order_date": REFERENCE_DATE - timedelta(days=60),
        "shipping_method": "standard",
        "shipping_cost": 0.00,
        "tracking_number": "1Z999AA10123456786",
        "carrier": "UPS",
        "estimated_delivery": None,
        "shipped_date": REFERENCE_DATE - timedelta(days=55),
        "delivered_date": REFERENCE_DATE - timedelta(days=46),
        "cancelled_date": None,
        "refund": None,
    },
    "1005": {
        "order_id": "1005",
        "customer_email": "alex@email.com",
        "status": "cancelled",
        "items": [
            {"title": "Becoming", "format": "hardcover", "price": 19.99},
        ],
        "order_date": REFERENCE_DATE - timedelta(days=15),
        "shipping_method": "standard",
        "shipping_cost": 4.99,
        "tracking_number": None,
        "carrier": None,
        "estimated_delivery": None,
        "shipped_date": None,
        "delivered_date": None,
        "cancelled_date": REFERENCE_DATE - timedelta(days=8),
        "refund": {
            "amount": 19.99,
            "method": "original_payment",
            "status": "processed",
        },
    },
}

# ---------------------------------------------------------------------------
# Returns ledger — populated by initiate_return tool at runtime
# ---------------------------------------------------------------------------

RETURNS: dict[str, dict] = {}

# Tracks which orders have had eligibility *verified* (i.e., check_eligibility
# was called on a delivered order). Does NOT mean items are eligible — only
# that the check ran. initiate_return rejects any order not in this dict,
# enforcing the prerequisite chain at the code level regardless of LLM
# behavior. Storing the reason lets create_return apply the same damage
# exception that check_eligibility uses.
ELIGIBILITY_VERIFIED: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helper functions (used by tools.py)
# ---------------------------------------------------------------------------

def get_order(order_id: str) -> dict:
    """Retrieve an order by ID. Returns an actionable error dict if not found."""
    order = ORDERS.get(order_id)
    if order is None:
        return {
            "error": "order_not_found",
            "message": (
                f"No order found with ID '{order_id}'. "
                "Bookly order numbers are 4-digit numbers (e.g., 1001 through 1005). "
                "Ask the customer to check their order confirmation email for the "
                "correct number."
            ),
        }
    return _format_order(order)


def _format_order(order: dict) -> dict:
    """Return a clean, agent-friendly representation of an order."""
    items_summary = [
        {
            "title": item["title"],
            "format": item["format"],
            "price": f"${item['price']:.2f}",
        }
        for item in order["items"]
    ]
    total = sum(item["price"] for item in order["items"])

    result: dict = {
        "order_id": order["order_id"],
        "status": order["status"],
        "customer_email": order["customer_email"],
        "items": items_summary,
        "item_count": len(order["items"]),
        "subtotal": f"${total:.2f}",
        "order_date": _fmt_date(order["order_date"]),
        "shipping_method": order["shipping_method"],
        "shipping_cost": f"${order['shipping_cost']:.2f}",
    }

    if order["status"] == "processing":
        result["note"] = "Order is being prepared. No tracking available yet."

    if order["shipped_date"]:
        result["shipped_date"] = _fmt_date(order["shipped_date"])
    if order["carrier"]:
        result["carrier"] = order["carrier"]
    if order["tracking_number"]:
        result["tracking_number"] = order["tracking_number"]
    if order["estimated_delivery"]:
        result["estimated_delivery"] = _fmt_date(order["estimated_delivery"])
    if order["delivered_date"]:
        result["delivered_date"] = _fmt_date(order["delivered_date"])
        days_since = (REFERENCE_DATE - order["delivered_date"]).days
        result["days_since_delivery"] = days_since
    if order["cancelled_date"]:
        result["cancelled_date"] = _fmt_date(order["cancelled_date"])
    if order["refund"]:
        result["refund"] = {
            "amount": f"${order['refund']['amount']:.2f}",
            "method": order["refund"]["method"],
            "status": order["refund"]["status"],
        }

    return result


def check_eligibility(order_id: str, reason: str) -> dict:
    """
    Check return eligibility for every item in an order.

    Returns per-item eligibility with clear reasons for denials so the agent
    can explain each one to the customer.
    """
    raw_order = ORDERS.get(order_id)
    if raw_order is None:
        return {
            "error": "order_not_found",
            "message": (
                f"No order found with ID '{order_id}'. "
                "Verify the order number before checking eligibility."
            ),
        }

    if raw_order["status"] == "cancelled":
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "order_cancelled",
            "message": (
                "This order was cancelled and has already been refunded. "
                "No return is needed."
            ),
            "eligible_items": [],
            "ineligible_items": [
                {"title": item["title"], "reason": "Order was cancelled and refunded"}
                for item in raw_order["items"]
            ],
        }

    if raw_order["status"] in ("processing", "shipped"):
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": "not_delivered",
            "message": (
                f"This order has not been delivered yet (status: {raw_order['status']}). "
                "Returns can only be initiated after delivery."
            ),
            "eligible_items": [],
            "ineligible_items": [
                {"title": item["title"], "reason": "Order not yet delivered"}
                for item in raw_order["items"]
            ],
        }

    is_damaged = "damage" in reason.lower() or "defective" in reason.lower()
    window = BOOKLY_POLICIES["returns"]["physical_window_days"]
    delivered = raw_order["delivered_date"]
    days_since = (REFERENCE_DATE - delivered).days if delivered else None

    eligible_items = []
    ineligible_items = []

    for item in raw_order["items"]:
        if item["format"] == "digital":
            ineligible_items.append({
                "title": item["title"],
                "format": item["format"],
                "price": f"${item['price']:.2f}",
                "reason": "Digital purchases (ebooks, audiobooks) are non-refundable",
            })
        elif is_damaged:
            eligible_items.append({
                "title": item["title"],
                "format": item["format"],
                "price": f"${item['price']:.2f}",
                "reason": "Damaged items can be returned at any time",
            })
        elif days_since is not None and days_since > window:
            ineligible_items.append({
                "title": item["title"],
                "format": item["format"],
                "price": f"${item['price']:.2f}",
                "reason": (
                    f"Delivered {days_since} days ago, which is outside the "
                    f"{window}-day return window"
                ),
            })
        else:
            eligible_items.append({
                "title": item["title"],
                "format": item["format"],
                "price": f"${item['price']:.2f}",
                "reason": "Within return window",
            })

    any_eligible = len(eligible_items) > 0
    refund_total = sum(
        float(i["price"].replace("$", "")) for i in eligible_items
    )

    # Record that eligibility was checked, along with the reason
    ELIGIBILITY_VERIFIED[order_id] = reason

    return {
        "order_id": order_id,
        "eligible": any_eligible,
        "eligible_items": eligible_items,
        "ineligible_items": ineligible_items,
        "refund_total": f"${refund_total:.2f}" if any_eligible else "$0.00",
        "refund_methods": BOOKLY_POLICIES["returns"]["refund_methods"],
    }


def create_return(order_id: str, items: list[str], refund_method: str) -> dict:
    """
    Initiate a return. Writes to the RETURNS ledger and returns a confirmation.

    Validates inputs — including eligibility checks as a safety net — and
    returns actionable errors on failure.
    """
    raw_order = ORDERS.get(order_id)
    if raw_order is None:
        return {
            "error": "order_not_found",
            "message": f"No order found with ID '{order_id}'.",
        }

    # --- Prerequisite gate: eligibility must be checked first ---
    if order_id not in ELIGIBILITY_VERIFIED:
        return {
            "error": "eligibility_not_checked",
            "message": (
                f"Return eligibility has not been verified for order #{order_id}. "
                "You must call check_return_eligibility before initiating a return. "
                "This ensures the items are eligible and the customer has been "
                "informed of the return details."
            ),
        }

    # --- Duplicate return check ---
    existing_return = RETURNS.get(f"RET-{order_id}")
    if existing_return is not None:
        already_returned = existing_return["items"]
        overlap = [t for t in items if t in already_returned]
        if overlap:
            return {
                "error": "already_returned",
                "message": (
                    f"A return has already been initiated for order #{order_id} "
                    f"(Return ID: {existing_return['return_id']}). "
                    f"Items already in return: {already_returned}. "
                    "The customer can check their email for the return label, "
                    "or contact bookly.com/contact for help with an existing return."
                ),
            }

    valid_methods = BOOKLY_POLICIES["returns"]["refund_methods"]
    if refund_method not in valid_methods:
        return {
            "error": "invalid_refund_method",
            "message": (
                f"'{refund_method}' is not a valid refund method. "
                f"Choose one of: {', '.join(valid_methods)}"
            ),
        }

    if not items:
        return {
            "error": "no_items_specified",
            "message": (
                "No items were specified for return. "
                "Please provide the exact book title(s) to return."
            ),
        }

    order_titles = {item["title"] for item in raw_order["items"]}
    unknown = [t for t in items if t not in order_titles]
    if unknown:
        return {
            "error": "items_not_in_order",
            "message": (
                f"These items are not in order #{order_id}: {unknown}. "
                f"Items in this order: {sorted(order_titles)}"
            ),
        }

    # --- Eligibility safety net (defense-in-depth) ---
    if raw_order["status"] in ("processing", "shipped"):
        return {
            "error": "order_not_delivered",
            "message": (
                f"Order #{order_id} has not been delivered yet "
                f"(status: {raw_order['status']}). "
                "Returns can only be initiated after delivery."
            ),
        }

    if raw_order["status"] == "cancelled":
        return {
            "error": "order_cancelled",
            "message": (
                f"Order #{order_id} was already cancelled and refunded. "
                "No return is needed."
            ),
        }

    window = BOOKLY_POLICIES["returns"]["physical_window_days"]
    delivered = raw_order.get("delivered_date")
    days_since = (REFERENCE_DATE - delivered).days if delivered else None

    # Retrieve the reason stored during the eligibility check so we can
    # apply the same damage exception here.
    stored_reason = ELIGIBILITY_VERIFIED.get(order_id, "")
    is_damaged = "damage" in stored_reason.lower() or "defective" in stored_reason.lower()

    rejected_items: list[dict] = []
    eligible_items: list[dict] = []

    for item in raw_order["items"]:
        if item["title"] not in items:
            continue
        if item["format"] == "digital":
            rejected_items.append({
                "title": item["title"],
                "reason": "Digital purchases (ebooks, audiobooks) are non-refundable",
            })
        elif is_damaged:
            eligible_items.append(item)
        elif days_since is not None and days_since > window:
            rejected_items.append({
                "title": item["title"],
                "reason": (
                    f"Delivered {days_since} days ago, outside the "
                    f"{window}-day return window"
                ),
            })
        else:
            eligible_items.append(item)

    if rejected_items:
        return {
            "error": "items_not_eligible",
            "message": (
                "The following items are not eligible for return: "
                + "; ".join(
                    f"'{r['title']}' — {r['reason']}" for r in rejected_items
                )
                + ". Use check_return_eligibility first to verify which items "
                "can be returned."
            ),
            "rejected_items": rejected_items,
        }

    # --- All validations passed — create the return ---
    return_id = f"RET-{order_id}"

    refund_amount = sum(item["price"] for item in eligible_items)

    return_record = {
        "return_id": return_id,
        "order_id": order_id,
        "items": items,
        "refund_amount": f"${refund_amount:.2f}",
        "refund_method": refund_method,
        "status": "initiated",
        "label_sent": True,
        "refund_eta": BOOKLY_POLICIES["returns"]["refund_processing_days"],
    }
    RETURNS[return_id] = return_record

    return {
        "return_id": return_id,
        "status": "initiated",
        "items_returned": items,
        "refund_amount": f"${refund_amount:.2f}",
        "refund_method": refund_method,
        "label_sent": True,
        "message": (
            f"Return {return_id} has been initiated. "
            f"A prepaid shipping label will be sent to the customer's email "
            f"within 24 hours. Refund of ${refund_amount:.2f} will be processed "
            f"{BOOKLY_POLICIES['returns']['refund_processing_days']} after we "
            f"receive the returned item."
        ),
    }


def search_policies(query: str) -> str:
    """
    Simple keyword search across policy sections.

    Returns the most relevant policy section(s) based on keyword overlap.
    If nothing matches, returns a helpful fallback.
    """
    query_lower = query.lower()
    scored: list[tuple[int, str, str]] = []

    for section_name, section_text in POLICY_SECTIONS.items():
        score = sum(
            1
            for word in query_lower.split()
            if word in section_name or word in section_text.lower()
        )
        if score > 0:
            scored.append((score, section_name, section_text))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return (
            "No specific policy found for that query. "
            "The customer can contact our team at bookly.com/contact or "
            "call 1-800-BOOKLY for further assistance."
        )

    results = [text for _, _, text in scored[:2]]
    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fmt_date(d: date | None) -> str | None:
    """Format a date as 'March 25, 2026' for agent-friendly output, or None."""
    if d is None:
        return None
    return d.strftime("%B %d, %Y")
