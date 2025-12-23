"""
Unit tests for the LangGraph agent.

These tests verify:
1. Agent builds correctly
2. Nodes exist and are connected
3. State transitions work
4. Parsing functions handle edge cases
"""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAgentStructure:
    """Tests for agent graph structure."""
    
    def test_agent_imports(self):
        """Test that agent module can be imported."""
        from agents.chatbot import chatbot, build_research_agent
        assert chatbot is not None
        assert build_research_agent is not None
    
    def test_agent_builds_successfully(self):
        """Test that the agent graph compiles without errors."""
        from agents.chatbot import build_research_agent
        agent = build_research_agent()
        assert agent is not None
    
    def test_agent_has_correct_nodes(self):
        """Test that all 4 nodes are present in the graph."""
        from agents.chatbot import build_research_agent
        agent = build_research_agent()
        
        # Get the graph representation
        graph_str = str(agent.get_graph())
        
        # Check all nodes exist
        assert "input" in graph_str
        assert "search" in graph_str
        assert "verify" in graph_str
        assert "final" in graph_str


class TestParsingFunctions:
    """Tests for the parsing helper functions."""
    
    def test_parse_input_response_valid(self):
        """Test parsing a valid input response."""
        from agents.chatbot import parse_input_response
        
        response = """SANITIZED_QUERY: What is AI?
QUERY_TYPE: factual
IS_VALID: true
VALIDATION_NOTE: OK
SENSITIVE_DOMAIN: none"""
        
        result = parse_input_response(response, "What is AI?")
        
        assert result["sanitized_query"] == "What is AI?"
        assert result["query_type"] == "factual"
        assert result["is_valid"] is True
        assert result["validation_error"] == ""
        assert result["sensitive_domain"] == ""
    
    def test_parse_input_response_invalid(self):
        """Test parsing an invalid input response."""
        from agents.chatbot import parse_input_response
        
        response = """SANITIZED_QUERY: harmful content
QUERY_TYPE: factual
IS_VALID: false
VALIDATION_NOTE: Contains inappropriate content
SENSITIVE_DOMAIN: none"""
        
        result = parse_input_response(response, "harmful content")
        
        assert result["is_valid"] is False
        assert "inappropriate" in result["validation_error"].lower()
    
    def test_parse_input_response_malformed(self):
        """Test parsing handles malformed response gracefully."""
        from agents.chatbot import parse_input_response
        
        response = "This is not a valid format at all"
        
        result = parse_input_response(response, "some query")
        
        # Should return defaults without crashing
        assert result["query_type"] == "factual"
        assert result["is_valid"] is True
    
    def test_parse_verify_response_pass(self):
        """Test parsing a passing verification."""
        from agents.chatbot import parse_verify_response
        
        response = """VERIFICATION_STATUS: PASS
CONFIDENCE: HIGH
REASON: Found reliable sources

VERIFIED_FACTS:
• AI is advancing rapidly (Source: TechNews)"""
        
        result = parse_verify_response(response)
        
        assert result["verification_passed"] is True
        assert "AI is advancing" in result["verified_facts"]
    
    def test_parse_verify_response_fail(self):
        """Test parsing a failing verification."""
        from agents.chatbot import parse_verify_response
        
        response = """VERIFICATION_STATUS: FAIL
CONFIDENCE: LOW
REASON: No relevant results found"""
        
        result = parse_verify_response(response)
        
        assert result["verification_passed"] is False


class TestConstants:
    """Tests for constants and configuration."""
    
    def test_max_retries_set(self):
        """Test that MAX_RETRIES is defined."""
        from agents.chatbot import MAX_RETRIES
        
        assert MAX_RETRIES >= 1
        assert MAX_RETRIES <= 5  # Reasonable limit
    
    def test_prompts_defined(self):
        """Test that all prompts are defined and non-empty."""
        from agents.chatbot import (
            INPUT_PROMPT,
            SEARCH_PROMPT,
            VERIFY_PROMPT,
            FINAL_PROMPT,
            CREATIVE_PROMPT
        )
        
        assert len(INPUT_PROMPT) > 50
        assert len(SEARCH_PROMPT) > 20
        assert len(VERIFY_PROMPT) > 50
        assert len(FINAL_PROMPT) > 50
        assert len(CREATIVE_PROMPT) > 20


class TestAgentState:
    """Tests for the AgentState TypedDict."""
    
    def test_agent_state_fields(self):
        """Test that AgentState has required fields."""
        from agents.chatbot import AgentState
        
        # Check annotations exist
        annotations = AgentState.__annotations__
        
        required_fields = [
            "messages",
            "sanitized_query",
            "query_type",
            "is_valid",
            "search_query",
            "search_results",
            "verified_facts",
            "verification_passed",
            "retry_count",
            "should_retry",
            "provider",
            "model"
        ]
        
        for field in required_fields:
            assert field in annotations, f"Missing field: {field}"


class TestRouting:
    """Tests for routing functions."""
    
    def test_route_after_input_factual(self):
        """Test routing factual queries to search."""
        from agents.chatbot import route_after_input
        
        state = {
            "is_valid": True,
            "query_type": "factual"
        }
        
        result = route_after_input(state)
        assert result == "search"
    
    def test_route_after_input_creative(self):
        """Test routing creative queries to final."""
        from agents.chatbot import route_after_input
        
        state = {
            "is_valid": True,
            "query_type": "creative"
        }
        
        result = route_after_input(state)
        assert result == "final"
    
    def test_route_after_input_invalid(self):
        """Test routing invalid queries to final."""
        from agents.chatbot import route_after_input
        
        state = {
            "is_valid": False,
            "query_type": "factual"
        }
        
        result = route_after_input(state)
        assert result == "final"
    
    def test_route_after_verify_should_retry(self):
        """Test routing when retry is needed."""
        from agents.chatbot import route_after_verify
        
        state = {
            "should_retry": True,
            "retry_count": 1
        }
        
        result = route_after_verify(state)
        assert result == "search"
    
    def test_route_after_verify_done(self):
        """Test routing when verification is complete."""
        from agents.chatbot import route_after_verify
        
        state = {
            "should_retry": False,
            "retry_count": 0
        }
        
        result = route_after_verify(state)
        assert result == "final"


class TestSensitiveDomainDetection:
    """Tests for sensitive domain detection."""
    
    def test_detect_medical_query(self):
        """Test detection of medical queries."""
        from agents.chatbot import detect_sensitive_domain
        
        medical_queries = [
            "What should I take for a fever?",
            "How to treat headache pain?",
            "Is this medication safe during pregnancy?",
            "What are the symptoms of diabetes?",
            "Should I see a doctor for this rash?",
        ]
        
        for query in medical_queries:
            result = detect_sensitive_domain(query)
            assert result == "medical", f"Failed for: {query}"
    
    def test_detect_legal_query(self):
        """Test detection of legal queries."""
        from agents.chatbot import detect_sensitive_domain
        
        legal_queries = [
            "Can I sue my employer?",
            "How to file for divorce?",
            "What are my rights if arrested?",
            "Do I need a lawyer for this contract?",
            "How does child custody work?",
        ]
        
        for query in legal_queries:
            result = detect_sensitive_domain(query)
            assert result == "legal", f"Failed for: {query}"
    
    def test_detect_financial_query(self):
        """Test detection of financial queries."""
        from agents.chatbot import detect_sensitive_domain
        
        financial_queries = [
            "Should I invest in stocks?",
            "How to file my taxes?",
            "Is this a good mortgage rate?",
            "Should I sell my 401k?",
            "What insurance should I buy?",
        ]
        
        for query in financial_queries:
            result = detect_sensitive_domain(query)
            assert result == "financial", f"Failed for: {query}"
    
    def test_detect_non_sensitive_query(self):
        """Test that non-sensitive queries return empty string."""
        from agents.chatbot import detect_sensitive_domain
        
        non_sensitive_queries = [
            "What is the weather today?",
            "Who won the World Cup?",
            "How does photosynthesis work?",
            "What is the capital of France?",
            "Tell me about AI",
        ]
        
        for query in non_sensitive_queries:
            result = detect_sensitive_domain(query)
            assert result == "", f"Should be non-sensitive: {query}"
    
    def test_disclaimers_defined(self):
        """Test that refusal templates are defined."""
        from agents.chatbot import (
            MEDICAL_REFUSAL,
            LEGAL_REFUSAL,
            FINANCIAL_REFUSAL
        )
        
        assert len(MEDICAL_REFUSAL) > 100
        assert "medical advice" in MEDICAL_REFUSAL.lower()
        
        assert len(LEGAL_REFUSAL) > 100
        assert "legal advice" in LEGAL_REFUSAL.lower()
        
        assert len(FINANCIAL_REFUSAL) > 100
        assert "financial" in FINANCIAL_REFUSAL.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
