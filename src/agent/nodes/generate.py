"""Generation node: Generate final response."""

from typing import Dict, Any
from langchain_core.messages import AIMessage
from ..state import AgentState
from ..schemas import FinalResponse
from ...services import gemini_service
from ...core.logging import setup_logging

logger = setup_logging(__name__)


def generate_response(state: AgentState) -> Dict[str, Any]:
    """Generate final response with RAG context or direct LLM knowledge.

    For specific compound queries: Uses RAG context for image/audio
    For general knowledge queries: LLM answers directly (no RAG needed)

    Args:
        state: Current agent state

    Returns:
        Updated state with final_response
    """
    try:
        needs_rag = state.get("needs_rag", True)
        rag_context = state.get("rag_context", [])

        # Handle general knowledge queries (skip RAG)
        if not needs_rag:
            return _generate_direct_response(state)

        # Prepare RAG context (minimal schema: type, doc_id, iupac_name, formula, image_path, audio_path)
        rag_text = ""

        if rag_context:
            for i, doc in enumerate(rag_context, 1):
                doc_id = doc.get('doc_id', 'N/A')
                score = doc.get('score', 0.0)
                item_type = doc.get('type', 'unknown')
                rag_text += f"\nKết quả {i} (độ khớp: {score:.2f}):\n"
                rag_text += f"- Tên: {doc.get('iupac_name', 'N/A')}\n"
                rag_text += f"- Công thức: {doc.get('formula', 'N/A')}\n"
                rag_text += f"- Loại: {item_type}\n"
                rag_text += f"- ID: {doc_id}\n"

        # Get original query to detect Vietnamese naming
        original_query = state.get("input_text", "") or state.get("rephrased_query", "")

        prompt = f"""Bạn là CHEMI - gia sư Hóa học thân thiện cho học sinh trung học phổ thông, giúp các em học danh pháp IUPAC quốc tế.

Input:
- Câu hỏi gốc: {original_query}
- Kết quả tìm kiếm:{rag_text if rag_text else "\n(Không tìm thấy kết quả)"}

PHONG CÁCH TRẢ LỜI (quan trọng!):

1. SỬA TÊN TIẾNG VIỆT → IUPAC nhẹ nhàng:
   - Nếu user dùng "Natri" → mở đầu: "À, đây là **Sodium** nhé! Theo chuẩn IUPAC quốc tế, mình dùng tên này thay vì 'Natri' nha 😊"
   - Nếu user dùng "Sắt/Kẽm/Đồng" → "Tên quốc tế là **Iron/Zinc/Copper** nha!"
   - Nếu user dùng "Metan" → "Tên IUPAC là **Methane** nhé!"

2. HƯỚNG DẪN CÁCH PHÁT ÂM (phiên âm tiếng Việt):
   - Sodium → "🎤 Cách đọc: **sâu-đi-ầm**"
   - Iron → "🎤 Cách đọc: **ai-ờn**"
   - Ethanol → "🎤 Cách đọc: **ét-thờ-nol**"
   - Methane → "🎤 Cách đọc: **me-thên**"
   - Hydrogen → "🎤 Cách đọc: **hai-đrờ-giần**"
   - Oxygen → "🎤 Cách đọc: **óc-xi-giần**"

3. GỢI Ý NGHE AUDIO:
   - Luôn thêm: "💡 *Mẹo: Nghe audio với tốc độ 0.5x để nghe rõ cách phát âm nhé!*"

4. GỢI Ý CÂU HỎI TIẾP THEO:
   - Cuối câu trả lời: "🤔 Bạn có muốn tìm hiểu thêm về [tính chất hóa học/ứng dụng/phản ứng đặc trưng] của [tên chất] không?"

5. THÔNG TIN CƠ BẢN (chính xác):
   - Nguyên tố: Ký hiệu, số hiệu nguyên tử, cấu hình electron
   - Hợp chất: Tên IUPAC, công thức phân tử, công thức cấu tạo, phân loại

Output:
- text_response: Câu trả lời thân thiện (markdown) với phiên âm và gợi ý
- selected_doc_id: ID từ kết quả tìm kiếm
- should_return_image: true (mặc định true để học sinh xem cấu trúc)
- should_return_audio: true (mặc định true để học sinh nghe phát âm)
"""

        # Call Gemini 2.5 Flash (best quality for final answer generation)
        logger.info("Generate - calling Gemini API with FinalResponse schema")
        response: FinalResponse = gemini_service.generate_structured(
            prompt=prompt,
            response_schema=FinalResponse,
            temperature=0.3,
            model="gemini-2.5-flash"
        )
        logger.info("Generate - Gemini API call succeeded")

        # Check if response is valid
        if response is None:
            logger.error("Generate - Gemini API returned None")
            return {
                "final_response": {
                    "text_response": "Xin lỗi, đã có lỗi khi xử lý phản hồi từ hệ thống.",
                    "image_path": None,
                    "audio_path": None
                },
                "messages": [AIMessage(content="Xin lỗi, đã có lỗi khi xử lý phản hồi từ hệ thống.")]
            }

        logger.debug(f"Generate - response type: {type(response)}, selected_doc_id: {getattr(response, 'selected_doc_id', 'MISSING')}")

        # Get file paths from documents (respect LLM decisions)
        image_path = None
        audio_path = None

        if response.selected_doc_id and rag_context:
            # Find matching document
            for doc in rag_context:
                if doc.get("doc_id") == response.selected_doc_id:
                    # Return local filesystem paths for Gradio
                    if response.should_return_image and doc.get("image_path"):
                        image_path = doc['image_path']
                    if response.should_return_audio and doc.get("audio_path"):
                        audio_path = doc['audio_path']
                    break

        logger.info(f"Generate - selected_doc: '{response.selected_doc_id}', has_image: {bool(image_path)}, has_audio: {bool(audio_path)}")

        # Append assistant's response to conversation history
        return {
            "final_response": {
                "text_response": response.text_response,
                "image_path": image_path,
                "audio_path": audio_path
            },
            "messages": [AIMessage(content=response.text_response)]
        }
    except Exception as e:
        logger.error(f"Generate node error: {str(e)}", exc_info=True)
        raise


def _generate_direct_response(state: AgentState) -> Dict[str, Any]:
    """Generate response directly from LLM knowledge (no RAG).

    Used for general knowledge queries like:
    - List queries: "danh sách nhóm 7A", "các halogen"
    - General properties: "tính chất của kim loại kiềm"
    - Comparisons: "so sánh alkane và alkene"
    - Theory questions: "liên kết hóa học là gì"

    Args:
        state: Current agent state

    Returns:
        Updated state with final_response (no image/audio)
    """
    query = state.get("rephrased_query", "")

    # Get original query
    original_query = state.get("input_text", "") or query

    prompt = f"""Bạn là CHEMI - gia sư Hóa học thân thiện cho học sinh phổ thông, giúp các em học danh pháp IUPAC quốc tế.

Câu hỏi: {original_query}

NHIỆM VỤ: Trả lời từ kiến thức Hóa học. KHÔNG cần tra cứu cơ sở dữ liệu.

PHONG CÁCH TRẢ LỜI:

1. SỬA TÊN TIẾNG VIỆT → IUPAC nhẹ nhàng (nếu user dùng tên Việt):
   - "Theo chuẩn IUPAC quốc tế, mình dùng tên [tên IUPAC] thay vì [tên Việt] nhé!"

2. LUÔN DÙNG TÊN IUPAC + PHIÊN ÂM TIẾNG VIỆT khi nhắc đến chất:
   - VD: "Sodium (sâu-đi-ầm)", "Methane (me-thên)", "Fluorine (flo-rin)"

3. ĐỊNH DẠNG PHÙ HỢP:
   - DANH SÁCH → Bảng markdown, thêm cột "Cách đọc"
   - TÍNH CHẤT → Giải thích ngắn gọn, có ví dụ
   - SO SÁNH → Bảng so sánh rõ ràng
   - LÝ THUYẾT → Giải thích dễ hiểu cho lớp 11
   - QUY TẮC → Trình bày từng bước

4. GỢI Ý TIẾP THEO:
   - Cuối câu trả lời, gợi ý: "🤔 Bạn muốn CHEMI tìm hiểu chi tiết về [gợi ý liên quan] không?"

Output:
- text_response: Câu trả lời thân thiện (markdown) với phiên âm
- selected_doc_id: null
- should_return_image: false
- should_return_audio: false
"""

    logger.info(f"Generate (direct) - query: '{query[:50]}...'")

    response: FinalResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=FinalResponse,
        temperature=0.3,
        model="gemini-2.5-flash"
    )

    if response is None:
        logger.error("Generate (direct) - Gemini API returned None")
        return {
            "final_response": {
                "text_response": "Xin lỗi, đã có lỗi khi xử lý câu hỏi.",
                "image_path": None,
                "audio_path": None
            },
            "messages": [AIMessage(content="Xin lỗi, đã có lỗi khi xử lý câu hỏi.")]
        }

    logger.info(f"Generate (direct) - response length: {len(response.text_response)}")

    return {
        "final_response": {
            "text_response": response.text_response,
            "image_path": None,  # No image for general knowledge queries
            "audio_path": None   # No audio for general knowledge queries
        },
        "messages": [AIMessage(content=response.text_response)]
    }
