"""Gemini API client helper."""

import os
import json
import time
from typing import Optional, Dict, Any, Type, TypeVar
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ..core.logging import setup_logging

T = TypeVar('T', bound=BaseModel)

load_dotenv(override=True)
logger = setup_logging(__name__)


class GeminiService:
    """Service class for Gemini API interactions."""

    def __init__(self):
        """Initialize Gemini client."""
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")

        # Cost optimization defaults
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
        self.top_p = float(os.getenv("GEMINI_TOP_P", "0.95"))

    def generate_json(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        temperature: float = 0.1,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate JSON response from Gemini.

        Args:
            prompt: Text prompt for the model
            image: Optional image bytes for multimodal input
            temperature: Sampling temperature (0.0-1.0)
            model: Optional model override (default: from env GEMINI_LLM_MODEL)

        Returns:
            Parsed JSON response as dictionary
        """
        model_name = model or self.model_name
        # Prepare content
        if image:
            # Multimodal: text + image
            content = [
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",  # Use PNG for molecular structure images
                        data=image
                    )
                )
            ]
        else:
            # Text only
            content = prompt

        # Generate response
        response = self.client.models.generate_content(
            model=model_name,
            contents=content,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
        )

        # Parse JSON
        try:
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError as e:
            # Fallback: return raw text in error field
            return {"error": f"JSON parse error: {e}", "raw_text": response.text}

    def generate_text(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate text response from Gemini.

        Args:
            prompt: Text prompt for the model
            image: Optional image bytes for multimodal input
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        # Prepare content
        if image:
            content = [
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",  # Use PNG for molecular structure images
                        data=image
                    )
                )
            ]
        else:
            content = prompt

        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=content,
            config=types.GenerateContentConfig(temperature=temperature)
        )

        return response.text

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        image: Optional[bytes] = None,
        temperature: float = 0.1,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: int = 10
    ) -> T:
        """Generate structured response using Pydantic schema with retry on overload.

        Args:
            prompt: Text prompt for the model
            response_schema: Pydantic BaseModel class for response structure
            image: Optional image bytes for multimodal input
            temperature: Sampling temperature (0.0-1.0)
            model: Optional model override
            max_output_tokens: Maximum tokens (default from env/1024)
            top_p: Nucleus sampling (default from env/0.95)
            max_retries: Maximum retry attempts (default 3)
            retry_delay: Delay in seconds between retries (default 10)

        Returns:
            Instantiated Pydantic model with parsed response
        """
        model_name = model or self.model_name

        # Prepare content
        if image:
            content = [
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",  # Use PNG for molecular structure images
                        data=image
                    )
                )
            ]
        else:
            content = prompt

        # Build config with defaults
        config = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "max_output_tokens": max_output_tokens or self.max_output_tokens,
            "top_p": top_p or self.top_p
        }

        # Retry loop for handling overload
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.debug(f"Gemini API call (attempt {attempt + 1}/{max_retries}) - model: {model_name}, schema: {response_schema.__name__}")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=content,
                    config=config
                )
                return response.parsed
            except Exception as e:
                last_error = e
                error_message = str(e).lower()

                # Check if it's an overload/rate limit error
                is_overload = any(keyword in error_message for keyword in [
                    'resource exhausted',
                    'overload',
                    'rate limit',
                    '429',
                    '503',
                    'too many requests',
                    'quota exceeded'
                ])

                if is_overload and attempt < max_retries - 1:
                    # Retry after delay
                    logger.warning(f"Gemini API overload (attempt {attempt + 1}/{max_retries}) - retrying after {retry_delay}s - error: {str(e)}")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Not overload error or final attempt - raise immediately
                    logger.error(f"Gemini API error (attempt {attempt + 1}/{max_retries}) - model: {model_name}, schema: {response_schema.__name__}, error: {str(e)}", exc_info=True)
                    raise

        # Should not reach here, but just in case
        raise last_error


# Global instance
gemini_service = GeminiService()
