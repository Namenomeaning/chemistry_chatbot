"""Generation node: Generate final response."""

from typing import Dict, Any
from langchain_core.messages import AIMessage
from ..state import AgentState
from ..schemas import FinalResponse
from ...services import gemini_service


def generate_response(state: AgentState) -> Dict[str, Any]:
    """Generate final response with RAG context.

    Synthesizes answer from retrieved documents.

    Args:
        state: Current agent state

    Returns:
        Updated state with final_response
    """
    # Prepare RAG context for prompt (already filtered by threshold >= 0.4)
    rag_text = ""
    rag_context = state.get("rag_context", [])

    if rag_context:
        for i, doc in enumerate(rag_context, 1):
            doc_id = doc.get('doc_id', 'N/A')
            score = doc.get('score', 0.0)
            rag_text += f"\nTài liệu {i} (doc_id: '{doc_id}', điểm: {score:.3f}):\n"
            rag_text += f"- Tên: {doc.get('iupac_name', 'N/A')}\n"
            rag_text += f"- Tên thông thường: {', '.join(doc.get('common_names', []))}\n"
            rag_text += f"- Công thức: {doc.get('formula', 'N/A')}\n"
            rag_text += f"- Phân loại: {doc.get('class', 'N/A')}\n"
            rag_text += f"- Thông tin: {doc.get('info', 'N/A')}\n"
            rag_text += f"- Quy tắc đặt tên: {doc.get('naming_rule', 'N/A')}\n"

    prompt = f"""Bạn là trợ lý Hóa học lớp 11.

Hãy trả lời câu hỏi dựa trên tài liệu từ CSDL.

Input:
- Câu hỏi: {state.get("rephrased_query", "")}
{rag_text if rag_text else "- KHÔNG có dữ liệu"}

Output:
- text_response: Trả lời chi tiết (markdown) nếu có tài liệu khớp, "Xin lỗi, không tìm thấy thông tin trong CSDL." nếu không
- selected_doc_id: doc_id của tài liệu khớp (null nếu không khớp/không có)
- should_return_image: true nếu hỏi công thức/cấu trúc/tổng quan/thông tin chung về hợp chất, false chỉ khi hỏi câu hỏi cụ thể không cần hình (VD: "tên là gì?", "ứng dụng gì?")
- should_return_audio: true nếu hỏi phát âm/tên/tổng quan/thông tin chung về hợp chất, false chỉ khi hỏi câu hỏi cụ thể không liên quan phát âm
"""

    # Call Gemini 2.5 Flash with structured output
    response: FinalResponse = gemini_service.generate_structured(
        prompt=prompt,
        response_schema=FinalResponse,
        temperature=0.3  # Slightly higher for natural response
    )

    # Extract URLs from selected document (respect LLM decisions)
    image_url = ""
    audio_url = ""

    if response.selected_doc_id and rag_context:
        # Find matching document
        for doc in rag_context:
            if doc.get("doc_id") == response.selected_doc_id:
                # Only extract if LLM decided to return them
                if response.should_return_image:
                    image_url = doc.get("image_url", "")
                if response.should_return_audio:
                    audio_url = doc.get("audio_url", "")
                break

    # Append assistant's response to conversation history
    return {
        "final_response": {
            "text_response": response.text_response,
            "image_url": image_url,
            "audio_url": audio_url
        },
        "messages": [AIMessage(content=response.text_response)]
    }
