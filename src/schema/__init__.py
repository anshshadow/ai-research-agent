"""
Schema definitions for API requests and responses.
Includes input validation for security.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


# Maximum message length to prevent abuse
MAX_MESSAGE_LENGTH = 10000


class ChatRequest(BaseModel):
    """Request model for chat endpoint with validation."""
    message: str = Field(
        ..., 
        description="User message to send to the agent",
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH
    )
    thread_id: Optional[str] = Field(
        None, 
        description="Thread ID for conversation persistence",
        max_length=100
    )
    provider: Optional[str] = Field(
        None, 
        description="LLM provider (groq, openai, gemini, anthropic)"
    )
    model: Optional[str] = Field(
        None, 
        description="Model ID to use",
        max_length=100
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and sanitize message."""
        # Strip whitespace
        v = v.strip()
        
        # Check not empty after strip
        if not v:
            raise ValueError("Message cannot be empty")
        
        return v
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: Optional[str]) -> Optional[str]:
        """Validate provider is one of the allowed values."""
        if v is None:
            return v
        
        allowed = ['groq', 'openai', 'gemini', 'anthropic']
        if v.lower() not in allowed:
            raise ValueError(f"Provider must be one of: {', '.join(allowed)}")
        
        return v.lower()


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str = Field(..., description="Agent's response message")
    thread_id: str = Field(..., description="Thread ID for this conversation")
    provider: str = Field(..., description="Provider used for this response")
    model: str = Field(..., description="Model used for this response")


class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    name: str


class ProviderInfo(BaseModel):
    """Information about a provider and its models."""
    id: str
    name: str
    models: list[ModelInfo]


class ServiceInfo(BaseModel):
    """Service metadata."""
    name: str
    version: str
    default_provider: str
    default_model: str
    providers: list[ProviderInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


__all__ = [
    "ChatRequest", 
    "ChatResponse", 
    "ServiceInfo", 
    "ModelInfo", 
    "ProviderInfo",
    "HealthResponse",
    "ErrorResponse",
    "MAX_MESSAGE_LENGTH"
]
