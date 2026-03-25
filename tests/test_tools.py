"""
Layer 1: Tool unit tests — deterministic, no API key needed.

Tests every tool function against the mock data to ensure correct behavior
across all order states and edge cases from the implementation plan.
"""

from __future__ import annotations

import pytest

from bookly.mock_data import ELIGIBILITY_VERIFIED, RETURNS
from bookly.tools import (
    check_return_eligibility,
    initiate_return,
    lookup_order,
    search_policy,
)


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------


class TestLookupOrder:
    def test_found_shipped(self):
        result = lookup_order.invoke({"order_id": "1002"})
        assert result["status"] == "shipped"
        assert result["tracking_number"] == "1Z999AA10123456784"
        assert result["carrier"] == "UPS"
        assert len(result["items"]) == 2

    def test_found_processing_no_tracking(self):
        """Order #1001: processing — must NOT have tracking info."""
        result = lookup_order.invoke({"order_id": "1001"})
        assert result["status"] == "processing"
        assert "tracking_number" not in result
        assert "note" in result

    def test_found_delivered(self):
        result = lookup_order.invoke({"order_id": "1003"})
        assert result["status"] == "delivered"
        assert "delivered_date" in result
        assert "days_since_delivery" in result

    def test_found_cancelled(self):
        result = lookup_order.invoke({"order_id": "1005"})
        assert result["status"] == "cancelled"
        assert result["refund"]["status"] == "processed"

    def test_not_found(self):
        result = lookup_order.invoke({"order_id": "9999"})
        assert result["error"] == "order_not_found"
        assert "9999" in result["message"]

    def test_not_found_empty_string(self):
        result = lookup_order.invoke({"order_id": ""})
        assert "error" in result


# ---------------------------------------------------------------------------
# check_return_eligibility
# ---------------------------------------------------------------------------


class TestCheckReturnEligibility:
    def test_eligible_within_window(self):
        """Order #1003: delivered 2 days ago — eligible."""
        result = check_return_eligibility.invoke(
            {"order_id": "1003", "reason": "customer_request"}
        )
        assert result["eligible"] is True
        assert len(result["eligible_items"]) == 1
        assert result["eligible_items"][0]["title"] == "Atomic Habits"
        assert float(result["refund_total"].replace("$", "")) > 0

    def test_ineligible_expired_window(self):
        """Order #1004 Sapiens: delivered 46 days ago — outside 30-day window."""
        result = check_return_eligibility.invoke(
            {"order_id": "1004", "reason": "customer_request"}
        )
        assert result["eligible"] is False
        sapiens = [i for i in result["ineligible_items"] if i["title"] == "Sapiens"]
        assert len(sapiens) == 1
        assert "30-day" in sapiens[0]["reason"]

    def test_ineligible_digital_item(self):
        """Order #1004 Python Crash Course: digital — non-refundable."""
        result = check_return_eligibility.invoke(
            {"order_id": "1004", "reason": "customer_request"}
        )
        digital = [
            i for i in result["ineligible_items"] if i["title"] == "Python Crash Course"
        ]
        assert len(digital) == 1
        assert "non-refundable" in digital[0]["reason"].lower()

    def test_ineligible_cancelled_order(self):
        """Order #1005: cancelled — already refunded."""
        result = check_return_eligibility.invoke(
            {"order_id": "1005", "reason": "customer_request"}
        )
        assert result["eligible"] is False
        assert "cancelled" in result.get("reason", "")

    def test_ineligible_not_delivered(self):
        """Order #1001: still processing — can't return yet."""
        result = check_return_eligibility.invoke(
            {"order_id": "1001", "reason": "customer_request"}
        )
        assert result["eligible"] is False
        assert "not_delivered" in result.get("reason", "")

    def test_damaged_always_eligible(self):
        """Order #1004 Sapiens: expired window but damaged — should be eligible."""
        result = check_return_eligibility.invoke(
            {"order_id": "1004", "reason": "damaged"}
        )
        sapiens = [i for i in result["eligible_items"] if i["title"] == "Sapiens"]
        assert len(sapiens) == 1
        assert "damaged" in sapiens[0]["reason"].lower()

    def test_damaged_digital_still_ineligible(self):
        """Digital items are non-refundable even if reported damaged."""
        result = check_return_eligibility.invoke(
            {"order_id": "1004", "reason": "damaged"}
        )
        digital = [
            i for i in result["ineligible_items"] if i["title"] == "Python Crash Course"
        ]
        assert len(digital) == 1

    def test_not_found(self):
        result = check_return_eligibility.invoke(
            {"order_id": "9999", "reason": "customer_request"}
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# initiate_return
# ---------------------------------------------------------------------------


class TestInitiateReturn:
    def setup_method(self):
        """Clear state before each test."""
        RETURNS.clear()
        ELIGIBILITY_VERIFIED.clear()

    def _check_eligibility(self, order_id: str, reason: str = "customer_request"):
        """Helper: run the eligibility check to satisfy the prerequisite gate."""
        check_return_eligibility.invoke({"order_id": order_id, "reason": reason})

    def test_successful_return(self):
        self._check_eligibility("1003")
        result = initiate_return.invoke(
            {
                "order_id": "1003",
                "items": ["Atomic Habits"],
                "refund_method": "original_payment",
            }
        )
        assert result["status"] == "initiated"
        assert result["return_id"] == "RET-1003"
        assert result["label_sent"] is True
        assert "$16.99" in result["refund_amount"]

    def test_store_credit(self):
        self._check_eligibility("1003")
        result = initiate_return.invoke(
            {
                "order_id": "1003",
                "items": ["Atomic Habits"],
                "refund_method": "store_credit",
            }
        )
        assert result["refund_method"] == "store_credit"
        assert result["status"] == "initiated"

    def test_invalid_refund_method(self):
        self._check_eligibility("1003")
        result = initiate_return.invoke(
            {
                "order_id": "1003",
                "items": ["Atomic Habits"],
                "refund_method": "bitcoin",
            }
        )
        assert result["error"] == "invalid_refund_method"

    def test_item_not_in_order(self):
        self._check_eligibility("1003")
        result = initiate_return.invoke(
            {
                "order_id": "1003",
                "items": ["Nonexistent Book"],
                "refund_method": "original_payment",
            }
        )
        assert result["error"] == "items_not_in_order"

    def test_order_not_found(self):
        result = initiate_return.invoke(
            {
                "order_id": "9999",
                "items": ["Anything"],
                "refund_method": "original_payment",
            }
        )
        assert "error" in result

    def test_duplicate_return_blocked(self):
        """Second return for the same order should be rejected."""
        self._check_eligibility("1003")
        initiate_return.invoke(
            {
                "order_id": "1003",
                "items": ["Atomic Habits"],
                "refund_method": "original_payment",
            }
        )
        result = initiate_return.invoke(
            {
                "order_id": "1003",
                "items": ["Atomic Habits"],
                "refund_method": "store_credit",
            }
        )
        assert result["error"] == "already_returned"

    def test_digital_item_blocked(self):
        """Non-refundable digital item should be rejected at the tool level."""
        self._check_eligibility("1004")
        result = initiate_return.invoke(
            {
                "order_id": "1004",
                "items": ["Python Crash Course"],
                "refund_method": "original_payment",
            }
        )
        assert result["error"] == "items_not_eligible"

    def test_expired_window_blocked(self):
        """Item outside 30-day return window should be rejected."""
        self._check_eligibility("1004")
        result = initiate_return.invoke(
            {
                "order_id": "1004",
                "items": ["Sapiens"],
                "refund_method": "original_payment",
            }
        )
        assert result["error"] == "items_not_eligible"

    def test_undelivered_order_blocked(self):
        """Cannot return an order that hasn't been delivered.
        check_eligibility for undelivered orders doesn't open the gate,
        so initiate_return is blocked by the prerequisite check."""
        self._check_eligibility("1001")
        result = initiate_return.invoke(
            {
                "order_id": "1001",
                "items": ["Dune"],
                "refund_method": "original_payment",
            }
        )
        assert result["error"] == "eligibility_not_checked"

    def test_cancelled_order_blocked(self):
        """Cannot return a cancelled order.
        check_eligibility for cancelled orders doesn't open the gate."""
        self._check_eligibility("1005")
        result = initiate_return.invoke(
            {
                "order_id": "1005",
                "items": ["Becoming"],
                "refund_method": "original_payment",
            }
        )
        assert result["error"] == "eligibility_not_checked"

    def test_empty_items_blocked(self):
        """Empty items list should be rejected."""
        self._check_eligibility("1003")
        result = initiate_return.invoke(
            {
                "order_id": "1003",
                "items": [],
                "refund_method": "original_payment",
            }
        )
        assert result["error"] == "no_items_specified"

    def test_eligibility_gate_blocks_without_check(self):
        """initiate_return should reject if check_return_eligibility was never called."""
        result = initiate_return.invoke(
            {
                "order_id": "1003",
                "items": ["Atomic Habits"],
                "refund_method": "original_payment",
            }
        )
        assert result["error"] == "eligibility_not_checked"


# ---------------------------------------------------------------------------
# search_policy
# ---------------------------------------------------------------------------


class TestSearchPolicy:
    """Tests for the search_policy tool — keyword matching, ranking, content accuracy."""

    # -- Section retrieval: each policy section is reachable --

    def test_shipping_section(self):
        result = search_policy.invoke({"query": "shipping options"})
        assert "Standard" in result
        assert "Express" in result
        assert "Overnight" in result

    def test_shipping_prices(self):
        """Shipping section includes all three price points."""
        result = search_policy.invoke({"query": "shipping cost"})
        assert "$4.99" in result
        assert "$9.99" in result
        assert "$19.99" in result

    def test_shipping_free_threshold(self):
        result = search_policy.invoke({"query": "free shipping"})
        assert "$35" in result

    def test_returns_section(self):
        result = search_policy.invoke({"query": "return policy"})
        assert "30 days" in result
        assert "non-refundable" in result

    def test_returns_refund_methods(self):
        result = search_policy.invoke({"query": "return refund"})
        assert "original payment" in result.lower()
        assert "store credit" in result.lower()

    def test_account_section(self):
        result = search_policy.invoke({"query": "password reset"})
        assert "bookly.com/reset-password" in result

    def test_account_email_change(self):
        result = search_policy.invoke({"query": "change email account"})
        assert "bookly.com/account/settings" in result

    def test_refund_section(self):
        result = search_policy.invoke({"query": "refund processing time"})
        assert "5-7 business days" in result

    def test_refund_cancelled_orders(self):
        result = search_policy.invoke({"query": "cancelled order refund"})
        assert "original payment method" in result.lower()

    def test_damaged_section(self):
        result = search_policy.invoke({"query": "damaged book"})
        assert "30-day return window does not apply" in result

    def test_damaged_prepaid_label(self):
        result = search_policy.invoke({"query": "damaged return label"})
        assert "prepaid return label" in result.lower()

    def test_tracking_section(self):
        result = search_policy.invoke({"query": "tracking number"})
        assert "ups.com/track" in result.lower()

    def test_tracking_processing_status(self):
        result = search_policy.invoke({"query": "order tracking processing"})
        assert "processing" in result.lower()

    # -- Ranking: multi-keyword queries return the most relevant section first --

    def test_ranking_shipping_first(self):
        """Query with 'shipping' + 'express' should rank shipping above others."""
        result = search_policy.invoke({"query": "express shipping delivery"})
        # Shipping section should appear before any other section
        assert result.index("Express shipping") < len(result) // 2

    def test_ranking_returns_two_sections(self):
        """A broad query can return up to 2 sections."""
        result = search_policy.invoke({"query": "return refund damaged"})
        # Should match multiple sections — result should be longer than any single section
        assert len(result) > 200

    # -- Edge cases --

    def test_no_match_fallback(self):
        """Gibberish query returns the fallback with contact info."""
        result = search_policy.invoke({"query": "xyzzy gibberish"})
        assert "bookly.com/contact" in result or "1-800-BOOKLY" in result

    def test_empty_query(self):
        """Empty string should return fallback, not crash."""
        result = search_policy.invoke({"query": ""})
        assert "bookly.com/contact" in result or "1-800-BOOKLY" in result

    def test_single_word_query(self):
        """Single relevant word still matches."""
        result = search_policy.invoke({"query": "shipping"})
        assert "Standard" in result

    def test_case_insensitive(self):
        """Query matching is case-insensitive."""
        result = search_policy.invoke({"query": "SHIPPING OPTIONS"})
        assert "Standard" in result

    def test_result_is_string(self):
        """search_policy always returns a string, not a dict."""
        result = search_policy.invoke({"query": "shipping"})
        assert isinstance(result, str)
