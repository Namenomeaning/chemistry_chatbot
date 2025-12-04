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

Input: Hình cấu trúc phân tử

Nhiệm vụ: Nhận dạng và trả về TÊN IUPAC duy nhất để search.

Output:
- search_query: CHỈ tên IUPAC (VD: "Ethanol", "Methane")
- is_valid: true nếu nhận dạng được
- error_message: null hoặc lỗi
- needs_rag: true (luôn true cho image query vì cần tra cứu thông tin chi tiết)
"""
    else:
        prompt = f"""Bạn là chuyên gia chuẩn hóa truy vấn Hóa học.

Input: {query}

NHIỆM VỤ 1 - PHÂN LOẠI needs_rag:

needs_rag = TRUE nếu hỏi về MỘT CHẤT CỤ THỂ và cần:
  + Hình ảnh cấu trúc phân tử
  + Audio phát âm tên
  + Thông tin chi tiết từ CSDL
  VD: "Ethanol là gì?", "CH4", "Cấu trúc của methane", "Natri"

needs_rag = FALSE nếu là KIẾN THỨC TỔNG QUÁT:
  + Danh sách/nhóm chất: "các halogen", "nhóm 7A", "danh sách ankan C1-C6"
  + Tính chất chung: "tính chất của kim loại kiềm", "đặc điểm của ancol"
  + So sánh: "so sánh alkane và alkene"
  + Quy tắc gọi tên: "cách đặt tên IUPAC", "quy tắc danh pháp"
  + Câu hỏi lý thuyết: "liên kết hóa học là gì", "phản ứng cộng là gì"

NHIỆM VỤ 2 - CHUẨN HÓA (chỉ khi needs_rag = TRUE):

Quy tắc chuẩn hóa:
1. Tên tiếng Việt → Tên IUPAC:
   - "Natri" → "Sodium", "Hydro" → "Hydrogen"
   - "Metan" → "Methane", "Rượu/Cồn" → "Ethanol"

2. Công thức → Giữ nguyên: "CH4" → "CH4", "Na" → "Na"

3. Tên IUPAC → Giữ nguyên: "Ethanol" → "Ethanol"

Output:
- needs_rag: true/false (QUAN TRỌNG - xác định trước)
- search_query: Tên IUPAC hoặc công thức (nếu needs_rag=true), hoặc "" (nếu needs_rag=false)
- is_valid: true
- error_message: null

VÍ DỤ:
"Natri là gì?" → needs_rag: true, search_query: "Sodium"
"CH4" → needs_rag: true, search_query: "CH4"
"Danh sách nhóm 7A" → needs_rag: false, search_query: ""
"Tính chất của halogen" → needs_rag: false, search_query: ""
"So sánh alkane và alkene" → needs_rag: false, search_query: ""
"""

    # Call Gemini 2.5 Flash Lite (balanced for name normalization)
    response: ExtractionResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=ExtractionResponse,
        image=state.get("input_image"),
        temperature=0.1,
        model="gemini-2.5-flash-lite"
    )

    logger.info(f"Extract & Validate - input: '{query[:50]}...', needs_rag: {response.needs_rag}, search_query: '{response.search_query[:50] if response.search_query else ''}'")

    return {
        "search_query": response.search_query,
        "is_valid": response.is_valid,
        "error_message": response.error_message if not response.is_valid else None,
        "needs_rag": response.needs_rag
    }
