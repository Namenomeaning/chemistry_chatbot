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
    """Route after extraction node based on needs_rag.

    Args:
        state: Current agent state

    Returns:
        "retrieve" if needs RAG lookup (specific compound query)
        "generate" if general knowledge query (skip RAG)
    """
    # Check if RAG retrieval is needed
    needs_rag = state.get("needs_rag", True)

    if needs_rag:
        # Specific compound query - need RAG for image/audio/detailed info
        return "retrieve"
    else:
        # General knowledge query - LLM answers directly (skip RAG)
        return "generate"
