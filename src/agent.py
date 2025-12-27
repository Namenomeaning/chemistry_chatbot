"""CHEMI Agent - ReAct agent for chemistry chatbot."""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.tools import search_image, generate_speech

load_dotenv()


# ============== Response Schema ==============

class ChemistryResponse(BaseModel):
    """Agent structured output."""
    text_response: str = Field(description="Câu trả lời bằng tiếng Việt")
    image_url: Optional[str] = Field(default=None, description="URL hình ảnh cấu trúc")
    audio_url: Optional[str] = Field(default=None, description="Đường dẫn file audio")


# ============== System Prompt ==============

SYSTEM_PROMPT = """Bạn là CHEMI - chatbot trợ lý Hóa học THPT thân thiện.

## Tools:
- search_image(keyword): Tìm hình ảnh hóa học. Keyword linh hoạt:
  + "ethanol structure" → công thức cấu tạo
  + "ethanol bottle" → hình thực tế
  + "water 3d molecule" → mô hình 3D
  + "chemistry lab" → phòng thí nghiệm
- generate_speech(text): Tạo audio phát âm (text = tên IUPAC tiếng Anh)

## Quy tắc:

### 1. Hỏi về chất ("X là gì?", "thông tin về X"):
   - Gọi search_image("<tên EN> structure") + generate_speech("<tên IUPAC>")
   - Trả lời: tên IUPAC, công thức, tính chất, ứng dụng
   - Giải thích cách phát âm tên chất (ví dụ: "Ethanol" đọc là "Et-tha-nol")
   - Giải thích nếu tên Việt khác quốc tế (Natri = Sodium)

### 2. Hỏi về hình ảnh:
   - Dùng keyword phù hợp yêu cầu (structure/bottle/3d/lab...)

### 3. Hỏi về phát âm:
   - Chỉ gọi generate_speech
   - Giải thích cách phát âm tên chất (ví dụ: "Ethanol" đọc là "Et-tha-nol")

### 4. Ngoài phạm vi của chatbot hóa học:
   - Từ chối lịch sự, không gọi tool

### 5. Format output:
   - KHÔNG viết ![image](...) hay link trong text_response
   - Hệ thống tự hiển thị từ image_url và audio_url
   - Trả lời tiếng Việt, thân thiện, dùng emoji
"""


# ============== Config ==============

TIMEOUT = 60
RECURSION_LIMIT = 10


# ============== Agent ==============

_agent = None
_executor = ThreadPoolExecutor(max_workers=4)


def get_agent():
    """Get or create agent."""
    global _agent
    if _agent:
        return _agent

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://gpt3.shupremium.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.3,
        max_completion_tokens=1000,
    )

    _agent = create_agent(
        model=llm,
        tools=[search_image, generate_speech],
        system_prompt=SYSTEM_PROMPT,
        response_format=ChemistryResponse,
        checkpointer=InMemorySaver(),
    )
    return _agent


async def invoke_agent(messages: list, thread_id: str) -> dict:
    """Invoke agent.

    Args:
        messages: List of message dicts with 'role' and 'content'
        thread_id: Conversation thread ID for memory

    Returns:
        Agent result dict with 'structured_response' key
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_executor, lambda: get_agent().invoke({"messages": messages}, config)),
        timeout=TIMEOUT,
    )
