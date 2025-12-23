"""
Tests for the FastAPI service.

These tests verify:
1. API endpoints work correctly
2. Request/response schemas are valid
3. Error handling works
"""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_chat_request_schema(self):
        """Test ChatRequest schema validation."""
        from schema import ChatRequest
        
        # Valid request
        request = ChatRequest(
            message="What is AI?",
            thread_id="test-123",
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
        
        assert request.message == "What is AI?"
        assert request.thread_id == "test-123"
    
    def test_chat_request_minimal(self):
        """Test ChatRequest with only required fields."""
        from schema import ChatRequest
        
        request = ChatRequest(message="Hello")
        
        assert request.message == "Hello"
        assert request.thread_id is None
        assert request.provider is None
    
    def test_chat_response_schema(self):
        """Test ChatResponse schema."""
        from schema import ChatResponse
        
        response = ChatResponse(
            message="This is the answer",
            thread_id="abc-123",
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
        
        assert response.message == "This is the answer"
        assert response.thread_id == "abc-123"
    
    def test_service_info_schema(self):
        """Test ServiceInfo schema."""
        from schema import ServiceInfo, ProviderInfo, ModelInfo
        
        info = ServiceInfo(
            name="Test Agent",
            version="1.0.0",
            default_provider="groq",
            default_model="llama-3.3-70b-versatile",
            providers=[
                ProviderInfo(
                    id="groq",
                    name="Groq",
                    models=[ModelInfo(id="llama", name="Llama")]
                )
            ]
        )
        
        assert info.name == "Test Agent"
        assert len(info.providers) == 1


class TestCoreModules:
    """Tests for core modules."""
    
    def test_settings_import(self):
        """Test that settings can be imported."""
        # This will fail if no API key is set, which is expected
        try:
            from core import settings
            assert settings is not None
        except ValueError as e:
            # Expected if no API key is configured
            assert "API key" in str(e)
    
    def test_get_model_function_exists(self):
        """Test that get_model function is importable."""
        from core.llm import get_model
        assert callable(get_model)
    
    def test_provider_enum(self):
        """Test Provider enum values."""
        from core.settings import Provider
        
        assert Provider.GROQ == "groq"
        assert Provider.OPENAI == "openai"
        assert Provider.GEMINI == "gemini"
        assert Provider.ANTHROPIC == "anthropic"


class TestAPIEndpoints:
    """Tests for API endpoint definitions."""
    
    def test_app_exists(self):
        """Test that FastAPI app is created."""
        from service.service import app
        assert app is not None
    
    def test_app_has_routes(self):
        """Test that app has expected routes."""
        from service.service import app
        
        routes = [route.path for route in app.routes]
        
        assert "/" in routes
        assert "/health" in routes
        assert "/chat" in routes
        assert "/models" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
