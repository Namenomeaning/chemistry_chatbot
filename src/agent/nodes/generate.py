"""Generation node: Generate final response."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from ..state import AgentState
from ..schemas import FinalResponse
from ...services import gemini_service
from ...core.logging import setup_logging

logger = setup_logging(__name__)


def _get_conversation_context(messages: List[BaseMessage], max_turns: int = 2) -> str:
    """Extract recent conversation history for context.

    Args:
        messages: List of conversation messages
        max_turns: Maximum number of Q&A turns to include

    Returns:
        Formatted string with recent conversation history
    """
    if not messages or len(messages) < 2:
        return ""

    # Get last few messages (exclude current - we want previous context)
    recent = messages[-(max_turns * 2 + 1):-1] if len(messages) > max_turns * 2 else messages[:-1]

    if not recent:
        return ""

    context_parts = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            content = str(msg.content)[:200]
            context_parts.append(f"User: {content}")
        elif isinstance(msg, AIMessage):
            # Truncate but keep key info from previous answers
            content = str(msg.content)[:600]
            context_parts.append(f"CHEMI: {content}")

    return "\n".join(context_parts) if context_parts else ""


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

        # Get original query and conversation history
        original_query = state.get("input_text", "") or state.get("rephrased_query", "")
        messages = state.get("messages", [])
        conversation_history = _get_conversation_context(messages)

        # Build conversation context section
        history_section = ""
        if conversation_history:
            history_section = f"""
LỊCH SỬ HỘI THOẠI (quan trọng - KHÔNG lặp lại thông tin đã nói):
{conversation_history}
---"""

        prompt = f"""Bạn là CHEMI - gia sư Hóa học thân thiện cho học sinh trung học phổ thông, giúp các em học danh pháp IUPAC quốc tế.
{history_section}
Input:
- Câu hỏi hiện tại: {original_query}
- Kết quả tìm kiếm:{rag_text if rag_text else "\n(Không tìm thấy kết quả)"}

QUY TẮC QUAN TRỌNG:
1. Nếu user yêu cầu "thêm thông tin", "chi tiết hơn", "còn gì nữa" → BỔ SUNG thông tin MỚI, KHÔNG lặp lại những gì đã nói
2. Thông tin bổ sung có thể bao gồm:
   - Tính chất vật lý (nhiệt độ sôi, nhiệt độ nóng chảy, màu sắc, mùi)
   - Tính chất hóa học (phản ứng đặc trưng, khả năng phản ứng)
   - Ứng dụng thực tế trong đời sống
   - Phương pháp điều chế
   - Lịch sử phát hiện
   - Vai trò trong cơ thể/môi trường

PHONG CÁCH TRẢ LỜI:

1. SỬA TÊN TIẾNG VIỆT → IUPAC nhẹ nhàng (chỉ lần đầu):
   - Nếu user dùng "Natri" → "À, đây là **Sodium** nhé!"
   - Nếu đã giới thiệu tên IUPAC trước đó → không cần nhắc lại

2. HƯỚNG DẪN CÁCH PHÁT ÂM (phiên âm tiếng Việt) - chỉ khi chưa nói:
   - Sodium → "🎤 Cách đọc: **sâu-đi-ầm**"
   - Ethanol → "🎤 Cách đọc: **ét-thờ-nol**"

3. THÔNG TIN CHI TIẾT (sử dụng kiến thức Hóa học):
   - Nguyên tố: Số hiệu, cấu hình electron, vị trí bảng tuần hoàn, tính chất đặc trưng
   - Hợp chất: Công thức, cấu trúc, tính chất, ứng dụng, điều chế

4. GỢI Ý CÂU HỎI TIẾP THEO:
   - Cuối câu trả lời: "🤔 Bạn muốn tìm hiểu thêm về [gợi ý cụ thể] không?"

Output:
- text_response: Câu trả lời thân thiện (markdown), BỔ SUNG thông tin mới nếu là follow-up
- selected_doc_id: ID từ kết quả tìm kiếm
- should_return_image: true (mặc định)
- should_return_audio: true (mặc định)
"""

        # Log conversation context for debugging
        if conversation_history:
            logger.info(f"Generate - has conversation history ({len(messages)} messages)")
        else:
            logger.info("Generate - no conversation history (first query)")

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
    original_query = state.get("input_text", "") or query
    messages = state.get("messages", [])
    conversation_history = _get_conversation_context(messages)

    # Build conversation context section
    history_section = ""
    if conversation_history:
        history_section = f"""
LỊCH SỬ HỘI THOẠI (KHÔNG lặp lại thông tin đã nói):
{conversation_history}
---"""

    prompt = f"""Bạn là CHEMI - gia sư Hóa học thân thiện cho học sinh phổ thông, giúp các em học danh pháp IUPAC quốc tế.
{history_section}
Câu hỏi hiện tại: {original_query}

NHIỆM VỤ: Trả lời từ kiến thức Hóa học. KHÔNG cần tra cứu cơ sở dữ liệu.

QUY TẮC QUAN TRỌNG:
- Nếu user yêu cầu "thêm thông tin" → BỔ SUNG thông tin MỚI, không lặp lại
- Sử dụng kiến thức Hóa học để cung cấp thông tin chi tiết, chính xác

PHONG CÁCH TRẢ LỜI:

1. SỬA TÊN TIẾNG VIỆT → IUPAC nhẹ nhàng (chỉ lần đầu):
   - "Theo chuẩn IUPAC quốc tế, mình dùng tên [tên IUPAC] thay vì [tên Việt] nhé!"

2. DÙNG TÊN IUPAC + PHIÊN ÂM TIẾNG VIỆT:
   - VD: "Sodium (sâu-đi-ầm)", "Methane (me-thên)", "Fluorine (flo-rin)"

3. ĐỊNH DẠNG PHÙ HỢP:
   - DANH SÁCH → Bảng markdown
   - TÍNH CHẤT → Giải thích ngắn gọn, có ví dụ
   - SO SÁNH → Bảng so sánh rõ ràng
   - LÝ THUYẾT → Giải thích dễ hiểu cho lớp 11

4. GỢI Ý TIẾP THEO:
   - Cuối câu trả lời: "🤔 Bạn muốn tìm hiểu thêm về [gợi ý cụ thể] không?"

Output:
- text_response: Câu trả lời thân thiện (markdown), BỔ SUNG thông tin mới nếu là follow-up
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
