"""Relevance node: Check if query is chemistry-related."""

from typing import Dict, Any
from ..state import AgentState
from ..schemas import RelevanceResponse
from ...services import gemini_service


def check_relevance(state: AgentState) -> Dict[str, Any]:
    """Check if rephrased query is chemistry-related.

    Args:
        state: Current agent state

    Returns:
        Updated state with is_chemistry_related and error_message
    """
    # Use rephrased query (already standalone)
    query = state.get("rephrased_query") or state.get("input_text") or ""
    has_image = state.get("input_image") is not None

    # Build prompt based on input type (text, image, or both)
    if has_image and not query:
        # Case 1: Image only
        prompt = """Bạn là chuyên gia phân loại nội dung Hóa học.

Hãy kiểm tra hình có liên quan Hóa học lớp 11 không.

Input: Hình ảnh

Output:
- is_chemistry_related: true nếu là cấu trúc phân tử/công thức/phản ứng/thiết bị, false nếu không
- error_message: Thông báo nếu false, null nếu true
"""
    elif has_image and query:
        # Case 2: Both text and image
        prompt = f"""Bạn là chuyên gia phân loại nội dung Hóa học.

Hãy kiểm tra câu hỏi + hình có liên quan Hóa học lớp 11 không.

Input: {query} (kèm hình)

Output:
- is_chemistry_related: true nếu về hợp chất/phản ứng/công thức/tính chất, false nếu không
- error_message: Thông báo nếu false, null nếu true
"""
    else:
        # Case 3: Text only
        prompt = f"""Bạn là chuyên gia phân loại nội dung Hóa học.

Hãy kiểm tra câu hỏi có liên quan Hóa học lớp 11 không.

Input: {query}

Output:
- is_chemistry_related: true nếu về hợp chất/phản ứng/công thức/tính chất, false nếu không
- error_message: Thông báo nếu false, null nếu true
"""

    # Call Gemini 2.5 Flash Lite (simple binary classification)
    response: RelevanceResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=RelevanceResponse,
        image=state.get("input_image"),
        temperature=0.1
    )

    return {
        "is_chemistry_related": response.is_chemistry_related,
        "error_message": response.error_message if not response.is_chemistry_related else None
    }
