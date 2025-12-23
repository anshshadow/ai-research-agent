"""
LLM factory module supporting multiple providers.
Supports: Groq, OpenAI, Google Gemini, Anthropic Claude
"""
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from core.settings import (
    Provider,
    GroqModel,
    OpenAIModel,
    GeminiModel,
    AnthropicModel,
    settings,
)


def get_model(provider: str | None = None, model_name: str | None = None) -> BaseChatModel:
    """
    Get an LLM model instance based on provider and model name.
    
    Args:
        provider: The provider to use (groq, openai, gemini, anthropic)
        model_name: The specific model to use
        
    Returns:
        A configured chat model instance.
    """
    # Default to settings if not provided
    if provider is None:
        provider = settings.DEFAULT_PROVIDER
    if model_name is None:
        model_name = settings.DEFAULT_MODEL
    
    # Convert string to Provider enum if needed
    if isinstance(provider, str):
        provider = Provider(provider.lower())
    
    # Create model based on provider
    if provider == Provider.GROQ:
        return _create_groq_model(model_name)
    elif provider == Provider.OPENAI:
        return _create_openai_model(model_name)
    elif provider == Provider.GEMINI:
        return _create_gemini_model(model_name)
    elif provider == Provider.ANTHROPIC:
        return _create_anthropic_model(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _create_groq_model(model_name: str) -> BaseChatModel:
    """Create a Groq model instance."""
    from langchain_groq import ChatGroq
    
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")
    
    return ChatGroq(
        model=model_name,
        temperature=0.3,  # Lower temperature for less hallucination
        groq_api_key=settings.GROQ_API_KEY.get_secret_value(),
        streaming=True,
    )


def _create_openai_model(model_name: str) -> BaseChatModel:
    """Create an OpenAI model instance."""
    from langchain_openai import ChatOpenAI
    
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    
    return ChatOpenAI(
        model=model_name,
        temperature=0.3,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        streaming=True,
    )


def _create_gemini_model(model_name: str) -> BaseChatModel:
    """Create a Google Gemini model instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
        streaming=True,
    )


def _create_anthropic_model(model_name: str) -> BaseChatModel:
    """Create an Anthropic Claude model instance."""
    from langchain_anthropic import ChatAnthropic
    
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    
    return ChatAnthropic(
        model=model_name,
        temperature=0.3,
        api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
        streaming=True,
    )


def get_available_models() -> dict[str, list[dict[str, str]]]:
    """
    Get all available models grouped by provider.
    Only returns models for providers with valid API keys.
    """
    available = {}
    
    if settings.GROQ_API_KEY:
        available["groq"] = [
            {"id": m.value, "name": _format_model_name(m.value, "Groq")}
            for m in GroqModel
        ]
    
    if settings.OPENAI_API_KEY:
        available["openai"] = [
            {"id": m.value, "name": _format_model_name(m.value, "OpenAI")}
            for m in OpenAIModel
        ]
    
    if settings.GOOGLE_API_KEY:
        available["gemini"] = [
            {"id": m.value, "name": _format_model_name(m.value, "Gemini")}
            for m in GeminiModel
        ]
    
    if settings.ANTHROPIC_API_KEY:
        available["anthropic"] = [
            {"id": m.value, "name": _format_model_name(m.value, "Claude")}
            for m in AnthropicModel
        ]
    
    return available


def _format_model_name(model_id: str, provider: str) -> str:
    """Format model ID into a display name."""
    name = model_id.replace("-", " ").replace("_", " ").title()
    return f"{provider}: {name}"
