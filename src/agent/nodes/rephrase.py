"""Rephrase node: Convert follow-up questions to standalone queries."""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from ..state import AgentState
from ..schemas import RephraseResponse
from ...services import gemini_service
from ...core.logging import setup_logging

logger = setup_logging(__name__)


def rephrase_query(state: AgentState) -> Dict[str, Any]:
    """Rephrase query using conversation context.

    Converts follow-up questions with pronouns into standalone queries
    by leveraging the last Q&A pair from conversation history.

    Args:
        state: Current agent state

    Returns:
        Updated state with rephrased_query and updated messages
    """
    current_query = state.get("input_text") or ""
    messages = state.get("messages", [])

    # First turn - no context needed
    if not messages:
        # For image-only first turn, set placeholder text
        query_for_message = current_query if current_query else "(hình ảnh)"
        logger.info(f"Rephrase (first turn) - query: '{query_for_message}'")
        return {
            "rephrased_query": query_for_message,
            "messages": [HumanMessage(content=query_for_message)]
        }

    # Get last Q&A pair (last 2 messages: HumanMessage, AIMessage)
    recent_messages = messages[-2:] if len(messages) >= 2 else messages
    prev_question = ""
    prev_answer = ""

    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            prev_question = msg.content
        elif isinstance(msg, AIMessage):
            prev_answer = msg.content

    prompt = f"""Bạn là chuyên gia xử lý ngữ cảnh hội thoại.

Hãy chuyển câu hỏi thành dạng độc lập bằng cách thay đại từ với tên cụ thể từ lịch sử.

Input:
- Hiện tại: {current_query if current_query else "(hình ảnh)"}
- Trước: Q: {prev_question} | A: {prev_answer}

Output:
- rephrased_query: Câu hỏi độc lập (thay đại từ nếu có, giữ nguyên nếu đã rõ)
"""

    # Call Gemini 2.5 Flash Lite
    response: RephraseResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=RephraseResponse,
        image=state.get("input_image"),
        temperature=0.1
    )

    logger.info(f"Rephrase (follow-up) - original: '{current_query}', rephrased: '{response.rephrased_query}'")

    return {
        "rephrased_query": response.rephrased_query,
        "messages": [HumanMessage(content=current_query)]
    }
