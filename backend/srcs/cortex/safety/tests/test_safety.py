"""Test suite for the safety module."""
import pytest
from cortex.safety.policy_enforcer import check_action, escalate_for_admin_approval


@pytest.mark.asyncio
async def test_pii_detection():
    """Test that PII is detected in content."""
    pii_content = "My API key is sk-12345 and my SSN is 000-00-0000."
    decision = await check_action("draft_email", {"content": pii_content})
    assert decision.decision == "require_approval"
    assert "Sensitive content detected" in decision.reason


@pytest.mark.asyncio
async def test_high_risk_action_requires_approval():
    """Test that a high-risk action requires approval."""
    decision = await check_action("send_email", {"content": "This is a test."})
    assert decision.decision == "require_approval"


@pytest.mark.asyncio
async def test_admin_approval_flow():
    """Test the 'four-eyes' admin approval workflow."""
    admin_user = {"email": "admin@cortex.dev", "roles": ["admin"]}
    normal_user = {"email": "user@cortex.dev", "roles": ["user"]}

    decision = await check_action("send_email", {"content": "This is a test."})
    assert decision.decision == "require_approval"

    # A normal user cannot approve
    escalated_by_user = escalate_for_admin_approval(decision, normal_user)
    assert escalated_by_user.decision == "require_approval"

    # An admin *can* approve
    escalated_by_admin = escalate_for_admin_approval(decision, admin_user)
    assert escalated_by_admin.decision == "allow"
