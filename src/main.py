"""CHEMI - FastAPI Backend for Chemistry Chatbot."""

import os
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.agent import invoke_agent, ChemistryResponse

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Request/Response Schemas ==============

class QueryRequest(BaseModel):
    """API request for text query."""
    text: Optional[str] = None
    image_base64: Optional[str] = None
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    """API response."""
    success: bool
    thread_id: str
    text_response: Optional[str] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None
    error: Optional[str] = None


# ============== Helpers ==============

def to_base64(file_path: Optional[str]) -> Optional[str]:
    """Convert local file to base64 or return URL directly."""
    if not file_path:
        return None
    if file_path.startswith(("http://", "https://")):
        return file_path
    path = Path(file_path)
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else None


async def process_query(
    text: Optional[str],
    image_base64: Optional[str],
    thread_id: Optional[str]
) -> QueryResponse:
    """Process a query and return response."""
    if not text and not image_base64:
        raise HTTPException(400, "text or image required")

    thread_id = thread_id or f"thread-{os.urandom(8).hex()}"

    # Build message content
    if image_base64:
        content = [
            {"type": "text", "text": text or "Đây là hợp chất gì?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ]
    else:
        content = text

    try:
        result = await invoke_agent([{"role": "user", "content": content}], thread_id)
    except asyncio.TimeoutError:
        return QueryResponse(success=False, thread_id=thread_id, error="Timeout - vui lòng thử lại")
    except Exception as e:
        if "recursion" in str(e).lower():
            return QueryResponse(success=False, thread_id=thread_id, error="Không tìm được thông tin")
        raise HTTPException(500, str(e)) from e

    if not (sr := result.get("structured_response")):
        return QueryResponse(success=False, thread_id=thread_id, error="No response from agent")
    response = QueryResponse(
        success=True,
        thread_id=thread_id,
        text_response=sr.text_response,
        image_base64=to_base64(sr.image_url),
        audio_base64=to_base64(sr.audio_url),
    )
    # Log response with truncated base64
    img_len = len(response.image_base64) if response.image_base64 else 0
    audio_len = len(response.audio_base64) if response.audio_base64 else 0
    logger.info(f"Response: text={response.text_response[:80]}... | image={img_len} chars | audio={audio_len} chars")
    return response


# ============== FastAPI App ==============

app = FastAPI(title="CHEMI - Chemistry Chatbot API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "CHEMI Chemistry Chatbot"}


@app.get("/")
async def root():
    """Serve the main chat UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "ok", "service": "CHEMI Chemistry Chatbot", "ui": "Not found"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint for text/base64 image input."""
    return await process_query(request.text, request.image_base64, request.thread_id)


@app.post("/query/upload", response_model=QueryResponse)
async def query_upload(
    text: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Query endpoint for file upload input (used by Gradio)."""
    image_base64 = None
    if image:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode()

    return await process_query(text, image_base64, thread_id)


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
