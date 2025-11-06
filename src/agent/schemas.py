"""Pydantic schemas for structured LLM responses."""

from typing import Optional
from pydantic import BaseModel, Field


class RephraseResponse(BaseModel):
    """Response schema for query rephrasing."""
    rephrased_query: str = Field(description="Standalone query rephrased with conversation context")


class RelevanceResponse(BaseModel):
    """Response schema for chemistry relevance check."""
    is_chemistry_related: bool = Field(description="Whether query is chemistry-related")
    error_message: Optional[str] = Field(default=None, description="Error message if not chemistry-related")


class ExtractionResponse(BaseModel):
    """Response schema for query extraction and validation."""
    search_query: str = Field(description="Optimized search query (expanded with keywords)")
    is_valid: bool = Field(description="Whether the formula/name is valid")
    error_message: Optional[str] = Field(default=None, description="Error message with suggestion if invalid")


class FinalResponse(BaseModel):
    """Response schema for final answer generation."""
    text_response: str = Field(description="Full answer in markdown format")
    selected_doc_id: Optional[str] = Field(default=None, description="doc_id of the compound being answered about")
    should_return_image: bool = Field(default=False, description="Whether to return structure image (True if asking about structure/formula)")
    should_return_audio: bool = Field(default=False, description="Whether to return audio pronunciation (True if asking about pronunciation/name)")
