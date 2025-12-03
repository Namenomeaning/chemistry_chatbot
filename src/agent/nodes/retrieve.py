"""Retrieval node: Retrieve from RAG."""

from typing import Dict, Any
from ..state import AgentState
from ...services import qdrant_service
from ...core.logging import setup_logging

logger = setup_logging(__name__)


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

    logger.info(f"Retrieve - searching for: '{query[:50]}...'")

    # Hybrid search with RRF
    results = qdrant_service.hybrid_search(query)

    logger.info(f"Retrieve - found {len(results)} documents")

    return {
        "rag_context": results
    }
