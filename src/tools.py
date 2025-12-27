"""Chemistry chatbot tools."""

import os
import time
import logging
from pathlib import Path

from langchain_core.tools import tool
from ddgs import DDGS
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio output directory
AUDIO_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "tts_output"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@tool
def search_image(keyword: str) -> str:
    """Tìm kiếm hình ảnh hóa học từ Internet (cấu trúc, thực tế, sơ đồ, v.v.).

    Args:
        keyword: Từ khóa tìm kiếm. Ví dụ:
            - "ethanol structure" → công thức cấu tạo
            - "ethanol bottle" → hình ảnh thực tế
            - "ethanol 3d model" → mô hình 3D
            - "chemistry lab" → phòng thí nghiệm

    Returns:
        URL hình ảnh hoặc thông báo lỗi.
    """
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                if results := list(ddgs.images(query=keyword, max_results=1)):
                    url = results[0]["image"]
                    logger.info(f"Image: {keyword} → {url}")
                    return url
        except Exception as e:
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                return f"Không tìm thấy hình ảnh cho '{keyword}'"
    return f"Không tìm thấy hình ảnh cho '{keyword}'"


@tool
def generate_speech(text: str, voice: str = "autumn") -> str:
    """Tạo âm thanh phát âm cho văn bản.

    Args:
        text: Văn bản cần phát âm
        voice: Giọng nói Orpheus (autumn, breeze, cove, juniper, etc.)

    Returns:
        Đường dẫn tệp audio hoặc thông báo lỗi.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Lỗi: GROQ_API_KEY chưa được cấu hình"

    try:
        client = Groq(api_key=api_key)
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice,
            response_format="wav",
            input=text,
        )

        safe_name = "".join(c if c.isalnum() else "_" for c in text.lower()[:30])
        output_file = AUDIO_OUTPUT_DIR / f"{safe_name}_{voice}.wav"
        response.write_to_file(output_file)

        logger.info(f"Speech: {output_file}")
        return str(output_file)

    except Exception as e:
        return f"Lỗi: Không thể tạo âm thanh - {e}"


