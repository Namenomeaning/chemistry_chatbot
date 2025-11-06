"""Streamlit chat interface for chemistry chatbot."""

import os
import uuid
import base64
from typing import Optional, List, Tuple

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv(override=True)


# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
    if "image_url" not in st.session_state:
        st.session_state.image_url = None
    if "audio_url" not in st.session_state:
        st.session_state.audio_url = None


def process_message(message: str, image_file: Optional[bytes]) -> dict:
    """Process user message via API.

    Args:
        message: User text input
        image_file: Uploaded image bytes (if any)

    Returns:
        API response dict
    """
    # Prepare payload
    payload = {
        "text": message,
        "thread_id": st.session_state.thread_id
    }

    # Encode image if provided
    if image_file:
        payload["image_base64"] = base64.b64encode(image_file).decode("utf-8")

    # Call API
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Không thể kết nối API tại {API_BASE_URL}"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "API timeout (quá 30 giây)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def clear_chat():
    """Clear chat history and create new thread."""
    st.session_state.messages = []
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
    st.session_state.image_url = None
    st.session_state.audio_url = None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Chatbot Hóa Học Lớp 11",
        page_icon="🧪",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("🧪 Chatbot Hóa Học Lớp 11")
    st.markdown("Hỏi về hợp chất hóa học, công thức, tính chất và danh pháp IUPAC.")

    # Sidebar for controls
    with st.sidebar:
        st.header("Điều khiển")

        # Clear chat button
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            clear_chat()
            st.rerun()

        # Thread ID display
        st.info(f"**Thread ID:** `{st.session_state.thread_id}`")

        # Image uploader
        st.header("Upload ảnh")
        uploaded_image = st.file_uploader(
            "Ảnh cấu trúc hóa học",
            type=["png", "jpg", "jpeg"],
            help="Upload ảnh công thức hoặc cấu trúc phân tử"
        )

        # Examples
        st.header("Ví dụ")
        examples = [
            "Cho tôi biết về ethanol",
            "Công thức của methane là gì?",
            "Axit axetic có công thức nào?",
            "Phân loại hợp chất C2H5OH"
        ]
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.example_query = example

    # Main chat area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Trò chuyện")

        # Display chat history
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Nhập câu hỏi hoặc tên/công thức hóa học...")

        # Handle example click
        if "example_query" in st.session_state:
            user_input = st.session_state.example_query
            del st.session_state.example_query

        # Process user input (only when user explicitly submits)
        if user_input:
            # Display user message
            user_msg = user_input
            if uploaded_image:
                user_msg += " [+Ảnh]"
            st.session_state.messages.append({"role": "user", "content": user_msg})

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_msg)

            # Call API
            with st.spinner("Đang xử lý..."):
                image_bytes = uploaded_image.read() if uploaded_image else None
                result = process_message(user_input, image_bytes)

            # Handle response
            if result.get("success"):
                response_text = result.get("text_response", "Không có phản hồi")
                st.session_state.image_url = result.get("image_url")
                st.session_state.audio_url = result.get("audio_url")
            else:
                response_text = f"❌ {result.get('error', 'Lỗi không xác định')}"
                st.session_state.image_url = None
                st.session_state.audio_url = None

            # Display assistant message
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            # Rerun to update sidebar outputs
            st.rerun()

    with col2:
        st.header("Kết quả")

        # Structure image output
        st.subheader("Cấu trúc phân tử")
        if st.session_state.image_url:
            try:
                st.image(st.session_state.image_url, use_container_width=True)
            except Exception as e:
                st.error(f"Không thể hiển thị ảnh: {e}")
        else:
            st.info("Chưa có ảnh")

        # Audio output
        st.subheader("Phát âm")
        if st.session_state.audio_url:
            try:
                st.audio(st.session_state.audio_url)
            except Exception as e:
                st.error(f"Không thể phát âm thanh: {e}")
        else:
            st.info("Chưa có âm thanh")


if __name__ == "__main__":
    main()
