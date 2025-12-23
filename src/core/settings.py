"""
Application settings with support for multiple LLM providers.
"""
from enum import StrEnum
from typing import Any

from dotenv import find_dotenv
from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(StrEnum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class GroqModel(StrEnum):
    """Available Groq models."""
    LLAMA_33_70B = "llama-3.3-70b-versatile"
    LLAMA_31_8B = "llama-3.1-8b-instant"
    LLAMA_32_90B_VISION = "llama-3.2-90b-vision-preview"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA2_9B = "gemma2-9b-it"


class OpenAIModel(StrEnum):
    """Available OpenAI models."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_35_TURBO = "gpt-3.5-turbo"


class GeminiModel(StrEnum):
    """Available Google Gemini models."""
    GEMINI_2_FLASH = "gemini-2.0-flash-exp"
    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_10_PRO = "gemini-1.0-pro"


class AnthropicModel(StrEnum):
    """Available Anthropic Claude models."""
    CLAUDE_35_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"


# All models combined
ALL_MODELS = {
    Provider.GROQ: list(GroqModel),
    Provider.OPENAI: list(OpenAIModel),
    Provider.GEMINI: list(GeminiModel),
    Provider.ANTHROPIC: list(AnthropicModel),
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # API Keys for different providers
    GROQ_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    
    # Default model
    DEFAULT_PROVIDER: Provider = Provider.GROQ
    DEFAULT_MODEL: str = GroqModel.LLAMA_33_70B.value
    
    # App settings
    APP_NAME: str = "AI Research Agent"
    DEBUG: bool = False
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that at least one API key is present."""
        if not any([
            self.GROQ_API_KEY,
            self.OPENAI_API_KEY,
            self.GOOGLE_API_KEY,
            self.ANTHROPIC_API_KEY,
        ]):
            raise ValueError(
                "At least one LLM API key must be set in environment variables or .env file. "
                "Supported: GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY"
            )
    
    def get_available_providers(self) -> list[Provider]:
        """Get list of providers with valid API keys."""
        providers = []
        if self.GROQ_API_KEY:
            providers.append(Provider.GROQ)
        if self.OPENAI_API_KEY:
            providers.append(Provider.OPENAI)
        if self.GOOGLE_API_KEY:
            providers.append(Provider.GEMINI)
        if self.ANTHROPIC_API_KEY:
            providers.append(Provider.ANTHROPIC)
        return providers
    
    @computed_field
    @property
    def BASE_URL(self) -> str:
        """Compute the base URL for the service."""
        return f"http://{self.HOST}:{self.PORT}"


# Global settings instance
settings = Settings()
