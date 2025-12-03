"""Extraction node: Extract, expand, and validate chemical query."""

from typing import Dict, Any
from ..state import AgentState
from ..schemas import ExtractionResponse
from ...services import gemini_service
from ...core.logging import setup_logging

logger = setup_logging(__name__)


def extract_and_validate(state: AgentState) -> Dict[str, Any]:
    """Expand query with keywords and validate chemical name/formula.

    Uses rephrased_query from context node (already standalone).

    Args:
        state: Current agent state

    Returns:
        Updated state with search_query, is_valid, and error_message
    """
    # Use rephrased query from context node (already includes conversation context)
    query = state.get("rephrased_query") or state.get("input_text") or "(hình ảnh)"

    # Check if this is an image-only query
    has_image = state.get("input_image") is not None
    is_image_only = has_image and (not state.get("input_text") or query == "(hình ảnh)")

    if is_image_only:
        prompt = """Bạn là chuyên gia nhận dạng cấu trúc phân tử.

Hãy phân tích hình cấu trúc phân tử, nhận dạng hợp chất, mở rộng search keywords.

Input: Hình cấu trúc phân tử

Output:
- search_query: Tên IUPAC + tên EN/VI + công thức (mở rộng từ hình)
- is_valid: true nếu nhận dạng được, false nếu không
- error_message: null hoặc lỗi nếu không nhận dạng được
"""
    else:
        prompt = f"""Bạn là chuyên gia danh pháp hóa học.

Hãy mở rộng query với keywords và kiểm tra tên/công thức IUPAC.

Input: {query}

Output:
- search_query: Tên EN + VI + công thức (mở rộng)
- is_valid: true nếu IUPAC đúng, false nếu sai
- error_message: Gợi ý sửa nếu sai, null nếu đúng
"""

    # Call Gemini 2.5 Flash with structured output
    response: ExtractionResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=ExtractionResponse,
        image=state.get("input_image"),
        temperature=0.1
    )

    logger.info(f"Extract & Validate - input: '{query[:50]}...', valid: {response.is_valid}, search_query: '{response.search_query[:50]}...'")

    return {
        "search_query": response.search_query,
        "is_valid": response.is_valid,
        "error_message": response.error_message if not response.is_valid else None
    }
