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
    from ...core.logging import setup_logging
    logger = setup_logging(__name__)

    import re

    # Use rephrased query (already standalone)
    query = state.get("rephrased_query") or state.get("input_text") or ""
    has_image = state.get("input_image") is not None

    # Log without base64 data
    log_query = re.sub(r'data:[^)]+', '[BASE64]', query[:100])
    logger.info(f"Check relevance - query: '{log_query}', has_image: {has_image}")

    # Build prompt based on input type (text, image, or both)
    if has_image and not query:
        # Case 1: Image only
        prompt = """Bạn là CHEMI - trợ lý Hóa học thân thiện.

Hãy kiểm tra hình có liên quan Hóa học lớp 11 không.

Input: Hình ảnh

Output:
- is_chemistry_related: true nếu là cấu trúc phân tử/công thức/phản ứng/thiết bị, false nếu không
- error_message: NẾU false, trả lời thân thiện:
  "Hmm, hình này không phải cấu trúc Hóa học rồi 😅 CHEMI chỉ nhận dạng được công thức phân tử, cấu trúc hợp chất thôi nha! Bạn thử upload hình công thức hóa học đi! 🧪"
"""
    elif has_image and query:
        # Case 2: Both text and image
        prompt = f"""Bạn là CHEMI - trợ lý Hóa học thân thiện.

Hãy kiểm tra câu hỏi + hình có liên quan Hóa học lớp 11 không.

Input: {query} (kèm hình)

CÂU HỎI + HÌNH LIÊN QUAN HÓA HỌC (trả về is_chemistry_related = true):
- Tên tiếng Việt: Natri, Kali, Sắt, Kẽm, Đồng, Hidro, Oxi, Canxi, Metan, Etan, Cồn...
- Tên IUPAC: Sodium, Potassium, Iron, Zinc, Methane, Ethanol...
- Công thức: Na, K, Fe, CH4, C2H5OH...
- Hỏi về cấu trúc, công thức, xác nhận hình ảnh

VÍ DỤ:
- "Kali có công thức như này đúng không?" + hình → TRUE
- "Đây có phải Natri không?" + hình → TRUE
- "Cấu trúc này là chất gì?" + hình → TRUE

Output:
- is_chemistry_related: true nếu về hợp chất/phản ứng/công thức/tính chất (mặc định TRUE cho tên nguyên tố/hợp chất)
- error_message: NẾU false, trả lời thân thiện:
  "Ôi, câu hỏi và hình này không liên quan đến Hóa học rồi 😅 CHEMI chỉ biết về nguyên tố, hợp chất thôi nha! Bạn thử hỏi về chất Hóa học nào đi! 🧪"
"""
    else:
        # Case 3: Text only
        prompt = f"""Bạn là CHEMI - trợ lý Hóa học thân thiện.

Hãy kiểm tra câu hỏi có liên quan Hóa học lớp 11 không.

Input: {query}

CÂU HỎI LIÊN QUAN HÓA HỌC (trả về is_chemistry_related = true):
- Tên TIẾNG VIỆT của nguyên tố: Natri, Sắt, Kẽm, Đồng, Hidro, Oxi, Canxi, Kali...
- Tên IUPAC/quốc tế: Sodium, Iron, Zinc, Copper, Hydrogen, Oxygen...
- Tên hợp chất tiếng Việt: Metan, Etan, Cồn, Rượu, Axit, Muối ăn...
- Tên hợp chất quốc tế: Methane, Ethanol, Acetic acid...
- Công thức hóa học: Na, Fe, CH4, C2H5OH, NaCl, H2O...
- Tính chất, phản ứng, ứng dụng của chất hóa học
- Cách phát âm tên hóa học
- Danh pháp IUPAC, quy tắc gọi tên
- Bảng tuần hoàn, nhóm nguyên tố

VÍ DỤ CÂU HỎI HÓA HỌC:
- "Natri là gì?" → TRUE (tên Việt của Sodium)
- "Sắt" → TRUE (tên Việt của Iron)
- "CH4 là gì?" → TRUE (công thức hóa học)
- "Ethanol" → TRUE (tên IUPAC)
- "Danh sách nhóm 7A" → TRUE (bảng tuần hoàn)

Output:
- is_chemistry_related: true nếu liên quan Hóa học (mặc định TRUE cho các tên nguyên tố/hợp chất)
- error_message: NẾU false, trả lời thân thiện:
  "Ôi, câu hỏi này không liên quan đến Hóa học rồi 😅 CHEMI chỉ biết về hóa học thôi nha! Bạn thử hỏi về Sodium, Ethanol hay bất kỳ chất nào đi, CHEMI sẽ giúp ngay! 🧪"
"""

    # Call Gemini 2.0 Flash (cheapest for simple binary classification)
    response: RelevanceResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=RelevanceResponse,
        image=state.get("input_image"),
        temperature=0.1,
        model="gemini-2.0-flash"
    )

    return {
        "is_chemistry_related": response.is_chemistry_related,
        "error_message": response.error_message if not response.is_chemistry_related else None
    }
