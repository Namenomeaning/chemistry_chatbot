"""Retrieval node: Retrieve from RAG."""

from typing import Dict, Any
from ..state import AgentState
from ...services import qdrant_service


def retrieve_from_rag(state: AgentState) -> Dict[str, Any]:
    """Retrieve relevant documents from RAG.

    Performs hybrid search (dense + sparse with RRF fusion).

    Args:
        state: Current agent state

    Returns:
        Updated state with rag_context
    """
    # Use search_query (expanded query from extraction node)
    query = state.get("search_query", "")

    # Hybrid search with RRF
    results = qdrant_service.hybrid_search(query)

    return {
        "rag_context": results
    }
