"""Chemistry chatbot agent with ReAct pattern."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

from ..core.logging import setup_logging

load_dotenv()
logger = setup_logging(__name__)


# Structured output schema
class ChemistryResponse(BaseModel):
    """Structured response from chemistry chatbot."""
    text_response: str = Field(description="Response text in Vietnamese")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    audio_url: Optional[str] = Field(default=None, description="Audio URL")


SYSTEM_PROMPT = """Bạn là CHEMI - gia sư Hóa học THPT thân thiện, vui vẻ.

<tools>
1. search_compound(query) - Tra cứu thông tin hợp chất/nguyên tố
2. search_image(keyword) - Tìm hình ảnh cấu trúc từ Internet
3. generate_speech(text, voice) - Tạo âm thanh phát âm

<rules>
- Gọi search_compound TRƯỚC để lấy thông tin chính xác
- Nếu không tìm thấy → DỪNG, không gọi tool khác
- Hỏi về "hình ảnh" hoặc "cấu trúc" → gọi search_image
- Hỏi về "phát âm" → gọi generate_speech
- Trả lời bằng Tiếng Việt, thân thiện

<style>
- Khích lệ: "Câu hỏi hay!", "Cùng tìm hiểu nhé!"
- Giải thích dễ hiểu, có ví dụ
- Kết thúc với gợi ý tiếp theo
</style>
"""


# Global agent instance
_agent_executor = None


def get_agent() -> AgentExecutor:
    """Get or create the agent executor."""
    global _agent_executor
    
    if _agent_executor is not None:
        return _agent_executor
    
    # Setup LLM
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://gpt3.shupremium.com/v1")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.3,
    )
    
    # Import tools here to avoid circular imports
    from .search import search_compound
    from .image_search import search_image
    from .speech import generate_speech
    
    tools = [search_compound, search_image, generate_speech]
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    _agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    logger.info("Chemistry agent initialized")
    return _agent_executor


def build_agent() -> AgentExecutor:
    """Build and return the agent (alias for backward compatibility)."""
    return get_agent()
