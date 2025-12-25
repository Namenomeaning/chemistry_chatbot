"""Image search tool using DuckDuckGo."""

import time
from langchain_core.tools import tool
from ddgs import DDGS

from ...core.logging import setup_logging

logger = setup_logging(__name__)


@tool
def search_image(keyword: str) -> str:
    """Tìm kiếm hình ảnh cấu trúc hóa học từ Internet.

    Args:
        keyword: Từ khóa tìm kiếm (vd: "ethanol structure", "water molecule")

    Returns:
        URL hình ảnh hoặc thông báo "Không tìm thấy" nếu không có kết quả.

    Example:
        search_image("ethanol structure") → "https://example.com/ethanol.jpg"
    """
    retries = 3
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query=keyword, max_results=1))
                if results:
                    image_url = results[0]['image']
                    logger.info(f"Image search '{keyword}' → {image_url}")
                    return image_url
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Image search failed after {retries} attempts: {e}")
                return f"Không thể tìm thấy hình ảnh cho '{keyword}'"

    return f"Không tìm thấy hình ảnh cho '{keyword}'"
