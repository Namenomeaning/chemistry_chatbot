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
    import json
    import re

    current_query_raw = state.get("input_text") or ""
    messages = state.get("messages", [])

    # Clean up input_text if it's in dict/list string format
    current_query = current_query_raw
    if current_query_raw.startswith("[{"):
        # Try to parse as JSON list
        try:
            parsed = json.loads(current_query_raw)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                current_query = parsed[0].get("text", "")
                # Remove markdown image syntax
                current_query = re.sub(r'!\[.*?\]\(.*?\)', '', current_query).strip()
        except:
            pass
    else:
        # Remove base64 image data from plain text
        current_query = re.sub(r'!\[.*?\]\(data:[^)]+\)', '[IMAGE]', current_query_raw).strip()

    # First turn - no context needed
    if not messages:
        # For image-only first turn, set placeholder text
        query_for_message = current_query if current_query else "(hình ảnh)"
        # Log without base64 data
        log_text = re.sub(r'data:[^)]+', '[BASE64_DATA]', current_query[:100])
        logger.info(f"Rephrase (first turn) - query: '{log_text}'")
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

    # Truncate prev_answer to extract key info (first 300 chars)
    prev_answer_short = prev_answer[:300] if prev_answer else ""

    prompt = f"""Bạn là chuyên gia xử lý ngữ cảnh hội thoại Hóa học.

NHIỆM VỤ: Thay đại từ "nó", "chất đó", "cái này" bằng TÊN HỢP CHẤT/NGUYÊN TỐ từ câu hỏi trước.

Input:
- Câu hỏi hiện tại: {current_query if current_query else "(hình ảnh)"}
- Câu hỏi trước: {prev_question}
- Câu trả lời trước (trích): {prev_answer_short}

QUY TẮC:
1. "nó", "chất đó", "nguyên tố đó" → thay bằng TÊN HỢP CHẤT/NGUYÊN TỐ (KHÔNG phải CHEMI - đó là tên chatbot)
2. Nếu câu hỏi đã rõ ràng → giữ nguyên
3. Nếu hỏi về hình ảnh → giữ nguyên

VÍ DỤ:
- Q trước: "Methane là gì?" → "nó có ứng dụng gì?" → "Methane có ứng dụng gì?"
- Q trước: "Ethanol" → "công thức của nó?" → "công thức của Ethanol?"
- Q trước: "Natri là gì?" → "nó có tính chất gì?" → "Sodium có tính chất gì?"

Output:
- rephrased_query: Câu hỏi độc lập (đã thay thế đại từ)
"""

    # Call Gemini 2.0 Flash (cheapest for simple task)
    response: RephraseResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=RephraseResponse,
        image=state.get("input_image"),
        temperature=0.1,
        model="gemini-2.0-flash"
    )

    # Log without base64 data
    log_query = re.sub(r'data:[^)]+', '[BASE64]', current_query[:80])
    logger.info(f"Rephrase (follow-up) - query: '{log_query}', rephrased: '{response.rephrased_query}'")

    return {
        "rephrased_query": response.rephrased_query,
        "messages": [HumanMessage(content=current_query)]
    }
