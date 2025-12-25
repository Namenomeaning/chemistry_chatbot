"""Text-to-speech generation tool using Groq API."""

import os
from pathlib import Path
from langchain_core.tools import tool
from groq import Groq

from ...core.logging import setup_logging

logger = setup_logging(__name__)

# Output directory for audio files
_AUDIO_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "tts_output"
_AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@tool
def generate_speech(text: str, voice: str = "autumn") -> str:
    """Tạo âm thanh phát âm cho văn bản bằng Groq API.

    Args:
        text: Văn bản cần phát âm (vd: "ethanol", "water molecule")
        voice: Giọng nói để sử dụng. Mặc định: "autumn"
               Các lựa chọn: "autumn", "breeze", "cove", "juniper", "onyx", "shimmer", "slate"

    Returns:
        Đường dẫn tệp audio hoặc thông báo lỗi.

    Example:
        generate_speech("ethanol", voice="autumn") → "/data/tts_output/ethanol_autumn.wav"
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment")
        return "Lỗi: GROQ_API_KEY chưa được cấu hình"

    try:
        client = Groq(api_key=api_key)

        # Generate audio using Groq API
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice,
            response_format="wav",
            input=text,
        )

        # Save to file
        safe_filename = "".join(c if c.isalnum() else "_" for c in text.lower()[:30])
        output_file = _AUDIO_OUTPUT_DIR / f"{safe_filename}_{voice}.wav"
        response.write_to_file(output_file)

        logger.info(f"Generated speech: {output_file}")
        return str(output_file)

    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        return f"Lỗi: Không thể tạo âm thanh - {str(e)}"
