"""Agent state definition using MessagesState."""

from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State for the chemistry chatbot agent.

    Uses LangGraph's MessagesState pattern with add_messages reducer
    for automatic conversation history management.
    """

    # Conversation history (managed by add_messages reducer)
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # Input fields
    input_text: Optional[str]
    input_image: Optional[bytes]

    # Context node output (query rephrasing + relevance check)
    rephrased_query: str
    is_chemistry_related: bool
    error_message: Optional[str]

    # Extraction node output (keyword expansion + validation)
    search_query: str
    is_valid: bool
    needs_rag: bool  # True = need RAG lookup, False = general knowledge (LLM answers directly)

    # Retrieval node output
    rag_context: List[Dict[str, Any]]

    # Generation node output
    final_response: Dict[str, Any]
