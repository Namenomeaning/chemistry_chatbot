"""Conditional edge functions for routing."""

from ..state import AgentState


def check_relevance_route(state: AgentState) -> str:
    """Route after context node based on chemistry relevance.

    Args:
        state: Current agent state

    Returns:
        "extract" if chemistry-related, "end" if not
    """
    if state.get("is_chemistry_related", False):
        return "extract"
    return "end"


def check_validity_route(state: AgentState) -> str:
    """Route after extraction node based on validity.

    Args:
        state: Current agent state

    Returns:
        "retrieve" if valid, "end" if invalid
    """
    if state.get("is_valid", False):
        return "retrieve"
    return "end"
