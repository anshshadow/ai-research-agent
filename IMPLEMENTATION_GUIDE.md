# 🔍 AI Research Agent - Complete Implementation Guide

## Project Assessment Document

**Project Name:** AI Research Agent  
**Technology Stack:** LangGraph, FastAPI, Streamlit, Python 3.11+  
**Last Updated:** December 23, 2024

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Project Structure](#3-project-structure)
4. [Core Components](#4-core-components)
5. [LangGraph Agent (4-Node Architecture)](#5-langgraph-agent-4-node-architecture)
6. [Sensitive Domain Handling](#6-sensitive-domain-handling)
7. [API Service](#7-api-service)
8. [Frontend UI](#8-frontend-ui)
9. [Data Flow Examples](#9-data-flow-examples)
10. [Testing](#10-testing)
11. [Evaluation Criteria Mapping](#11-evaluation-criteria-mapping)

---

## 1. Project Overview

This is an **AI Research Agent** built with LangGraph that:

- Accepts user questions via a web interface
- Searches the web for real-time information using DuckDuckGo
- Verifies facts before using them (with retry mechanism)
- Returns clean, cited answers
- Safely handles sensitive domains (medical, legal, financial) with safe refusals
- Supports multiple LLM providers (Groq, OpenAI, Gemini, Anthropic)

### Key Technologies:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Agent Framework** | LangGraph | 4-node state graph with conditional routing |
| **LLM Integration** | LangChain | Multi-provider LLM support |
| **Web Search** | DuckDuckGo | Real-time information retrieval |
| **Backend API** | FastAPI | REST API endpoints |
| **Frontend** | Streamlit | Interactive web UI |
| **Configuration** | Pydantic Settings | Environment-based config |
| **Testing** | Pytest | Unit tests (30 tests) |

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                           (Streamlit Frontend)                              │
│                          http://localhost:8501                              │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTP Request
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI SERVICE                                │
│                          http://localhost:8000                              │
│                                                                             │
│   Endpoints:                                                                │
│   ├── GET  /        → Service info                                         │
│   ├── GET  /health  → Health check                                         │
│   ├── GET  /models  → Available models                                     │
│   ├── POST /chat    → Send message (sync)                                  │
│   └── POST /chat/stream → Send message (streaming)                         │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ Invokes Agent
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH RESEARCH AGENT                            │
│                                                                             │
│  ┌─────────┐                                                                │
│  │  INPUT  │──── Sensitive? ────────────────────────────┐                   │
│  │ Node 1  │                                             │                   │
│  └────┬────┘                                             │                   │
│       │ Factual                                          │                   │
│       ▼                                                  │                   │
│  ┌─────────┐      ┌─────────┐                           │                   │
│  │ SEARCH  │─────▶│ VERIFY  │                           │                   │
│  │ Node 2  │      │ Node 3  │                           │                   │
│  └─────────┘      └────┬────┘                           │                   │
│       ▲                │                                 │                   │
│       └── Retry ───────┤ (if fail & retries left)       │                   │
│                        │                                 │                   │
│                        ▼                                 ▼                   │
│                   ┌─────────┐                                               │
│                   │  FINAL  │◀──────────────────────────┘                   │
│                   │ Node 4  │                                               │
│                   └─────────┘                                               │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌───────────┐     ┌───────────┐     ┌───────────┐
            │   GROQ    │     │  OPENAI   │     │  GEMINI   │
            │  (Llama)  │     │  (GPT-4)  │     │           │
            └───────────┘     └───────────┘     └───────────┘
```

---

## 3. Project Structure

```
ai-agent-toolkit/
│
├── src/                              # Source code
│   ├── agents/                       # LangGraph agent
│   │   ├── __init__.py              # Exports chatbot
│   │   └── chatbot.py               # 4-node graph (687 lines)
│   │
│   ├── core/                         # Core utilities
│   │   ├── __init__.py              # Exports get_model, settings
│   │   ├── settings.py              # Pydantic Settings config
│   │   └── llm.py                   # Multi-provider LLM factory
│   │
│   ├── schema/                       # Pydantic models
│   │   └── __init__.py              # Request/Response schemas with validation
│   │
│   ├── service/                      # FastAPI service
│   │   ├── __init__.py
│   │   └── service.py               # API endpoints
│   │
│   ├── run_service.py               # Backend entry point
│   └── streamlit_app.py             # Frontend UI
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_agent.py                # Agent tests (21 tests)
│   └── test_service.py              # Service tests (9 tests)
│
├── .env                              # API keys (gitignored)
├── .env.example                      # Environment template
├── .gitignore
├── pytest.ini                        # Test configuration
├── requirements.txt                  # Dependencies
├── LICENSE                           # MIT License
└── README.md                         # Documentation
```

---

## 4. Core Components

### 4.1 Settings (src/core/settings.py)

Manages configuration from environment variables using Pydantic Settings:

```python
from pydantic_settings import BaseSettings
from enum import Enum

class Provider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

class Settings(BaseSettings):
    # API Keys (at least one required)
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Defaults
    DEFAULT_PROVIDER: str = "groq"
    DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Design Decision:** Using Pydantic Settings ensures type safety and automatic validation of environment variables.

### 4.2 LLM Factory (src/core/llm.py)

Factory pattern for creating LLM instances:

```python
def get_model(provider: str, model_name: str) -> BaseChatModel:
    """Factory function to get LLM by provider and model."""
    
    if provider == "groq":
        return ChatGroq(
            model=model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3  # Low for accuracy, reduces hallucination
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.GOOGLE_API_KEY,
            temperature=0.3
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.3
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
```

**Design Decision:** Low temperature (0.3) reduces creative/random outputs and improves factual accuracy.

### 4.3 Schema (src/schema/__init__.py)

Pydantic models with input validation:

```python
from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_LENGTH = 10000

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    thread_id: Optional[str] = Field(None, max_length=100)
    provider: Optional[str] = None
    model: Optional[str] = None
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: Optional[str]) -> Optional[str]:
        allowed = ['groq', 'openai', 'gemini', 'anthropic']
        if v and v.lower() not in allowed:
            raise ValueError(f"Provider must be one of: {allowed}")
        return v.lower() if v else v

class ChatResponse(BaseModel):
    message: str
    thread_id: str
    provider: str
    model: str
```

**Security Feature:** Input validation prevents abuse (max length, required fields, allowed values).

---

## 5. LangGraph Agent (4-Node Architecture)

The core of the system is in `src/agents/chatbot.py`.

### 5.1 Agent State Definition

```python
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State that flows through all nodes."""
    
    # Core
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Input Node outputs
    sanitized_query: str          # Cleaned query
    query_type: str               # "factual", "creative", "conversational"
    is_valid: bool                # Passed validation?
    validation_error: str         # Error message if invalid
    sensitive_domain: str         # "medical", "legal", "financial", or ""
    
    # Search Node outputs
    search_query: str             # Optimized search query
    search_results: str           # Raw results from DuckDuckGo
    search_error: str             # Error message if search failed
    
    # Verify Node outputs
    verified_facts: str           # Verified information with sources
    verification_passed: bool     # Passed verification?
    retry_count: int              # Number of retries attempted
    should_retry: bool            # Flag to trigger retry
    
    # Configuration
    provider: str                 # Selected LLM provider
    model: str                    # Selected model name
```

### 5.2 Node 1: INPUT

**Purpose:** Sanitize, validate, classify query type, and detect sensitive domains.

```python
async def input_node(state: AgentState, config: RunnableConfig) -> dict:
    """INPUT NODE: Validate, classify, detect sensitive domain."""
    logger.info("INPUT NODE: Processing")
    
    # Get configuration
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", settings.DEFAULT_PROVIDER)
    model_name = configurable.get("model", settings.DEFAULT_MODEL)
    
    # Get user message
    user_input = state["messages"][-1].content
    
    try:
        model = get_model(provider, model_name)
        
        # Use LLM to classify input
        response = await model.ainvoke([
            SystemMessage(content=INPUT_PROMPT),
            HumanMessage(content=f"Input: {user_input}")
        ])
        
        # Parse LLM response
        parsed = parse_input_response(response.content, user_input)
        
        # Fallback: Local regex-based sensitive domain detection
        if not parsed["sensitive_domain"]:
            parsed["sensitive_domain"] = detect_sensitive_domain(user_input)
        
        return {
            "sanitized_query": parsed["sanitized_query"],
            "query_type": parsed["query_type"],
            "is_valid": parsed["is_valid"],
            "validation_error": parsed["validation_error"],
            "sensitive_domain": parsed["sensitive_domain"],
            "provider": provider,
            "model": model_name,
            "retry_count": 0,
            "should_retry": False,
            ...
        }
        
    except Exception as e:
        logger.error(f"INPUT NODE: Error - {e}")
        # Return safe defaults on error
        return {...}
```

**Key Features:**
- LLM-based query classification
- Regex-based fallback for sensitive domain detection
- Error handling with safe defaults

### 5.3 Node 2: SEARCH

**Purpose:** Generate optimized search query and search DuckDuckGo.

```python
async def search_node(state: AgentState, config: RunnableConfig) -> dict:
    """SEARCH NODE: Search the web."""
    logger.info("SEARCH NODE: Searching")
    
    sanitized_query = state.get("sanitized_query", "")
    retry_count = state.get("retry_count", 0)
    
    # Check if search tool is available
    if web_search is None:
        return {"search_error": "Search unavailable", ...}
    
    try:
        model = get_model(provider, model_name)
        
        # Use LLM to generate optimized search query
        query_response = await model.ainvoke([
            SystemMessage(content=SEARCH_PROMPT.format(query=sanitized_query)),
            HumanMessage(content=sanitized_query)
        ])
        search_query = query_response.content.strip()
        
        logger.info(f"SEARCH NODE: Query: {search_query}")
        
        # Execute DuckDuckGo search
        try:
            search_results = web_search.invoke(search_query)
            
            if not search_results or len(str(search_results)) < 50:
                return {"search_error": "No results", ...}
            
            return {
                "search_query": search_query,
                "search_results": str(search_results),
                "search_error": "",
            }
            
        except Exception as e:
            logger.error(f"SEARCH NODE: Search failed - {e}")
            return {"search_error": str(e), ...}
            
    except Exception as e:
        logger.error(f"SEARCH NODE: Error - {e}")
        return {"search_error": str(e), ...}
```

**Key Features:**
- LLM-optimized search queries
- DuckDuckGo for real-time search (no API key needed)
- Error handling for search failures

### 5.4 Node 3: VERIFY

**Purpose:** Verify search results quality and decide whether to retry.

```python
async def verify_node(state: AgentState, config: RunnableConfig) -> dict:
    """VERIFY NODE: Verify search results."""
    logger.info("VERIFY NODE: Verifying")
    
    search_results = state.get("search_results", "")
    search_error = state.get("search_error", "")
    retry_count = state.get("retry_count", 0)
    
    # Handle search errors
    if search_error and not search_results:
        if retry_count < MAX_RETRIES:
            return {
                "verification_passed": False,
                "should_retry": True,
                "retry_count": retry_count + 1,
            }
        else:
            return {
                "verification_passed": False,
                "should_retry": False,
                "verified_facts": "No search results available.",
            }
    
    try:
        model = get_model(provider, model_name)
        
        # Ask LLM to verify results
        response = await model.ainvoke([
            SystemMessage(content=VERIFY_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nResults:\n{search_results}")
        ])
        
        # Parse verification status
        parsed = parse_verify_response(response.content)
        
        if parsed["verification_passed"]:
            return {
                "verified_facts": parsed["verified_facts"],
                "verification_passed": True,
                "should_retry": False,
            }
        else:
            if retry_count < MAX_RETRIES:
                return {
                    "verification_passed": False,
                    "should_retry": True,
                    "retry_count": retry_count + 1,
                }
            else:
                return {
                    "verification_passed": False,
                    "should_retry": False,
                    "verified_facts": parsed["verified_facts"],
                }
                
    except Exception as e:
        logger.error(f"VERIFY NODE: Error - {e}")
        return {"verification_passed": False, "should_retry": False, ...}
```

**Key Features:**
- LLM-based verification of search results
- Retry mechanism (MAX_RETRIES = 2)
- Explicit `should_retry` flag for routing

### 5.5 Node 4: FINAL

**Purpose:** Generate the final response with appropriate handling for different query types.

```python
async def final_node(state: AgentState, config: RunnableConfig) -> dict:
    """FINAL NODE: Generate response."""
    logger.info("FINAL NODE: Generating response")
    
    sensitive_domain = state.get("sensitive_domain", "")
    query_type = state.get("query_type", "factual")
    verified_facts = state.get("verified_facts", "")
    
    try:
        model = get_model(provider, model_name)
        
        # Handle invalid input
        if not state.get("is_valid", True):
            return {"messages": [AIMessage(content=f"I cannot process this: {error}")]}
        
        # ===== SENSITIVE DOMAIN: Safe refusal =====
        if sensitive_domain:
            logger.info(f"FINAL NODE: Safe refusal for {sensitive_domain}")
            
            # Generate brief general info (non-actionable)
            general_response = await model.ainvoke([
                SystemMessage(content=SENSITIVE_RESPONSE_PROMPT),
                HumanMessage(content=f"Query: {sanitized_query}")
            ])
            
            # Wrap in refusal template
            if sensitive_domain == "medical":
                final_response = MEDICAL_REFUSAL.format(
                    general_info=general_response.content
                )
            elif sensitive_domain == "legal":
                final_response = LEGAL_REFUSAL.format(...)
            elif sensitive_domain == "financial":
                final_response = FINANCIAL_REFUSAL.format(...)
            
            return {"messages": [AIMessage(content=final_response)]}
        
        # ===== CREATIVE/CONVERSATIONAL: Direct response =====
        if query_type in ["creative", "conversational"]:
            response = await model.ainvoke([
                SystemMessage(content=CREATIVE_PROMPT),
                HumanMessage(content=sanitized_query)
            ])
            return {"messages": [response]}
        
        # ===== FACTUAL: Use verified facts =====
        response = await model.ainvoke([
            SystemMessage(content=FINAL_PROMPT),
            HumanMessage(content=f"Facts:\n{verified_facts}\n\nQuery: {query}")
        ])
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"FINAL NODE: Error - {e}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}
```

**Key Features:**
- Different handling for sensitive, creative, and factual queries
- Safe refusal templates with disclaimers
- Uses only verified facts for factual answers

### 5.6 Routing Functions

```python
def route_after_input(state: AgentState) -> Literal["search", "final"]:
    """Route after INPUT node."""
    
    # Invalid input → FINAL (show error)
    if not state.get("is_valid", True):
        logger.info("ROUTER: Invalid → FINAL")
        return "final"
    
    # SENSITIVE DOMAIN → FINAL (skip search, safe refusal)
    if state.get("sensitive_domain"):
        logger.info(f"ROUTER: Sensitive ({state['sensitive_domain']}) → FINAL")
        return "final"
    
    # Creative/conversational → FINAL (no search needed)
    if state.get("query_type") in ["creative", "conversational"]:
        logger.info(f"ROUTER: {state['query_type']} → FINAL")
        return "final"
    
    # Factual → SEARCH
    logger.info("ROUTER: Factual → SEARCH")
    return "search"


def route_after_verify(state: AgentState) -> Literal["search", "final"]:
    """Route after VERIFY node."""
    
    if state.get("should_retry", False):
        logger.info("ROUTER: Retry → SEARCH")
        return "search"  # Retry with different query
    
    logger.info("ROUTER: Done → FINAL")
    return "final"
```

### 5.7 Building the Graph

```python
from langgraph.graph import END, StateGraph

def build_research_agent() -> StateGraph:
    """Build the 4-node research agent."""
    logger.info("Building agent...")
    
    graph = StateGraph(AgentState)
    
    # Add all 4 nodes
    graph.add_node("input", input_node)
    graph.add_node("search", search_node)
    graph.add_node("verify", verify_node)
    graph.add_node("final", final_node)
    
    # Set entry point
    graph.set_entry_point("input")
    
    # Conditional edges from INPUT
    graph.add_conditional_edges(
        "input",
        route_after_input,
        {"search": "search", "final": "final"}
    )
    
    # SEARCH always goes to VERIFY
    graph.add_edge("search", "verify")
    
    # Conditional edges from VERIFY (retry or done)
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"search": "search", "final": "final"}
    )
    
    # FINAL goes to END
    graph.add_edge("final", END)
    
    logger.info("Agent built successfully")
    return graph.compile()

# Export the compiled agent
chatbot = build_research_agent()
```

---

## 6. Sensitive Domain Handling

### 6.1 Detection Keywords

```python
LEGAL_KEYWORDS = [
    r'\b(lawyer|attorney|legal|law|court|judge|lawsuit|sue)\b',
    r'\b(contract|agreement|liability|negligence|malpractice)\b',
    r'\barrested?\b',  # Matches "arrest" and "arrested"
    r'\b(police|criminal|crime|felony|misdemeanor)\b',
    r'\b(divorce|custody|child\s+support|alimony)\b',
    r'\b(will|estate|inheritance|trust|probate)\b',
    r'\b(patent|trademark|copyright)\b',
    r'\b(immigration|visa|deportation|asylum)\b',
    r'\b(my\s+)?rights\b',  # Matches "my rights" and "rights"
]

FINANCIAL_KEYWORDS = [
    r'\b(invest|investment|stock|bond|portfolio|dividend)\b',
    r'\b(tax|taxes|irs|deduction|exemption)\b',
    r'\b(loan|mortgage|debt|credit)\b',
    r'\b(retirement|401k|ira|pension)\b',
    r'\b(insurance|premium|deductible|coverage)\b',
    r'\b(bankruptcy|foreclosure)\b',
    r'\b(should\s+i\s+(buy|sell|invest))\b',
]

MEDICAL_KEYWORDS = [
    r'\b(fever|headache|pain|symptom|disease|illness)\b',
    r'\b(medicine|medication|drug|pill|dose|dosage)\b',
    r'\b(doctor|physician|hospital|clinic|treatment|diagnosis)\b',
    r'\b(cancer|diabetes|heart|infection|virus|bacteria)\b',
    r'\b(pregnant|pregnancy|baby|infant|pediatric)\b',
    r'\b(mental\s+health|depression|anxiety)\b',
    r'\b(vaccine|vaccination|immunization)\b',
    r'\b(prescription|pharmacy|pharmacist)\b',
    r'\b(should\s+i\s+take|can\s+i\s+take)\b',
]
```

### 6.2 Detection Function

```python
import re

def detect_sensitive_domain(query: str) -> str:
    """Detect sensitive domain using regex patterns."""
    query_lower = query.lower()
    
    # Check in order: legal → financial → medical
    # (Order matters to avoid false positives)
    
    for pattern in LEGAL_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "legal"
    
    for pattern in FINANCIAL_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "financial"
    
    for pattern in MEDICAL_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "medical"
    
    return ""  # Not sensitive
```

### 6.3 Safe Refusal Templates

```python
MEDICAL_REFUSAL = """⚠️ **I cannot provide medical advice.**

For health-related concerns like this, please consult a qualified healthcare professional who can:
- Evaluate your specific situation
- Consider your medical history
- Provide appropriate guidance

**General Information (Not Medical Advice):**
{general_info}

---
*For personalized medical guidance, please see a licensed healthcare provider.*
"""

LEGAL_REFUSAL = """⚠️ **I cannot provide legal advice.**

For legal matters like this, please consult a qualified attorney who can:
- Evaluate your specific situation
- Consider applicable laws in your jurisdiction
- Provide appropriate legal guidance

**General Information (Not Legal Advice):**
{general_info}

---
*For personalized legal guidance, please consult a licensed attorney.*
"""

FINANCIAL_REFUSAL = """⚠️ **I cannot provide financial advice.**

For financial decisions like this, please consult a qualified financial advisor who can:
- Evaluate your specific financial situation
- Consider your goals and risk tolerance
- Provide appropriate recommendations

**General Information (Not Financial Advice):**
{general_info}

---
*For personalized financial guidance, please consult a licensed financial advisor.*
"""
```

**Design Decision:** Sensitive queries skip the search node entirely and go directly to FINAL with a safe refusal. This prevents:
1. Searching for potentially harmful information
2. Providing actionable advice in sensitive domains
3. Liability issues

---

## 7. API Service

### 7.1 FastAPI Setup (src/service/service.py)

```python
from fastapi import FastAPI, HTTPException
from agents import chatbot
from schema import ChatRequest, ChatResponse, ServiceInfo
from core import settings

app = FastAPI(
    title="AI Research Agent",
    version="1.0.0"
)

# In-memory conversation storage (per-thread)
conversations: Dict[str, List[BaseMessage]] = {}


@app.get("/", response_model=ServiceInfo)
async def get_info():
    """Return service info and available providers."""
    return ServiceInfo(
        name="AI Research Agent",
        version="1.0.0",
        default_provider=settings.DEFAULT_PROVIDER,
        default_model=settings.DEFAULT_MODEL,
        providers=get_available_providers()
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def get_models():
    """Return available models grouped by provider."""
    return get_available_providers()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the research agent."""
    
    # Get or create thread ID
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Get conversation history for this thread
    messages = conversations.get(thread_id, [])
    
    # Add user message
    messages.append(HumanMessage(content=request.message))
    
    # Configure the agent
    config = {
        "configurable": {
            "provider": request.provider or settings.DEFAULT_PROVIDER,
            "model": request.model or settings.DEFAULT_MODEL,
            "thread_id": thread_id
        }
    }
    
    try:
        # Invoke the LangGraph agent
        result = await chatbot.ainvoke(
            {"messages": messages},
            config
        )
        
        # Extract response message
        response_message = result["messages"][-1].content
        
        # Store updated conversation
        messages.append(AIMessage(content=response_message))
        conversations[thread_id] = messages
        
        return ChatResponse(
            message=response_message,
            thread_id=thread_id,
            provider=config["configurable"]["provider"],
            model=config["configurable"]["model"]
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Get conversation history for a thread."""
    messages = conversations.get(thread_id, [])
    return {
        "thread_id": thread_id,
        "messages": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant",
             "content": m.content}
            for m in messages
        ]
    }
```

### 7.2 Entry Point (src/run_service.py)

```python
import uvicorn
from service.service import app
from core import settings

if __name__ == "__main__":
    print(f"🚀 Starting AI Research Agent")
    print(f"📦 Default provider: {settings.DEFAULT_PROVIDER}")
    print(f"🧠 Default model: {settings.DEFAULT_MODEL}")
    
    uvicorn.run(
        "service.service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True  # Auto-reload on code changes
    )
```

---

## 8. Frontend UI

### 8.1 Streamlit App (src/streamlit_app.py)

```python
import streamlit as st
import requests

API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Sidebar - Settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Check API connection
    try:
        info = requests.get(f"{API_URL}/").json()
        st.success("✅ Connected to API")
    except:
        st.error("❌ Cannot connect to API")
        st.stop()
    
    # Provider selection
    providers = {p["id"]: p["name"] for p in info["providers"]}
    provider = st.selectbox("Provider", options=list(providers.keys()))
    
    # Model selection (dynamic based on provider)
    models_response = requests.get(f"{API_URL}/models").json()
    provider_data = next(p for p in models_response if p["id"] == provider)
    models = {m["id"]: m["name"] for m in provider_data["models"]}
    model = st.selectbox("Model", options=list(models.keys()))
    
    # Agent workflow visualization
    st.markdown("### 🔄 Agent Workflow")
    st.markdown("""
    **INPUT** (Node 1)
      ↓ Sanitize & Validate
    **SEARCH** (Node 2)
      ↓ Query & Search Web
    **VERIFY** (Node 3)
      ├─ ✓ Pass → Final
      └─ ✗ Fail → Retry Search
    **FINAL** (Node 4)
      ↓ Synthesize Answer
    """)
    
    # New conversation button
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.rerun()

# Main area - Chat
st.title("🔍 AI Research Agent")
st.caption("Ask me anything - I'll search the web and verify facts!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "message": prompt,
                    "thread_id": st.session_state.thread_id,
                    "provider": provider,
                    "model": model
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["message"]
                st.session_state.thread_id = data["thread_id"]
                
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.error(f"Error: {response.text}")
```

---

## 9. Data Flow Examples

### Example 1: Factual Query

```
User Input: "What is LangGraph?"

┌─────────────────────────────────────────────────────────────┐
│ NODE 1: INPUT                                               │
│ Output:                                                     │
│   - sanitized_query: "What is LangGraph?"                  │
│   - query_type: "factual"                                  │
│   - sensitive_domain: "" (not sensitive)                   │
│   - is_valid: True                                         │
└─────────────────────────────────────────────────────────────┘
         │ route_after_input() → "search"
         ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 2: SEARCH                                              │
│ Output:                                                     │
│   - search_query: "LangGraph framework documentation 2024" │
│   - search_results: "[{snippet: LangGraph is...}, ...]"    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 3: VERIFY                                              │
│ Output:                                                     │
│   - verification_passed: True                              │
│   - verified_facts: "• LangGraph is... (Source: docs)"     │
│   - should_retry: False                                    │
└─────────────────────────────────────────────────────────────┘
         │ route_after_verify() → "final"
         ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 4: FINAL                                               │
│ Output:                                                     │
│   ## Answer                                                 │
│   LangGraph is a library for building stateful...          │
│                                                             │
│   ## Key Points                                            │
│   - Built by LangChain team (Source: docs)                 │
│   - Enables cyclic graphs (Source: GitHub)                 │
│                                                             │
│   ## Sources                                                │
│   - langgraph.dev                                          │
└─────────────────────────────────────────────────────────────┘
```

### Example 2: Medical Query (Sensitive - Skips Search)

```
User Input: "What should I take for fever?"

┌─────────────────────────────────────────────────────────────┐
│ NODE 1: INPUT                                               │
│ Output:                                                     │
│   - query_type: "factual"                                  │
│   - sensitive_domain: "medical" ← DETECTED!                │
│     (matched "fever" and "should I take")                  │
└─────────────────────────────────────────────────────────────┘
         │ route_after_input() → "final" (SKIP SEARCH!)
         ▼
┌─────────────────────────────────────────────────────────────┐
│ NODE 4: FINAL                                               │
│ Output:                                                     │
│                                                             │
│ ⚠️ **I cannot provide medical advice.**                    │
│                                                             │
│ For health-related concerns, please consult a qualified    │
│ healthcare professional who can:                            │
│ - Evaluate your specific situation                         │
│ - Consider your medical history                            │
│ - Provide appropriate guidance                             │
│                                                             │
│ **General Information (Not Medical Advice):**              │
│ Fever is a natural immune response. General approaches     │
│ typically include rest and staying hydrated.               │
│                                                             │
│ *For personalized guidance, see a healthcare provider.*    │
└─────────────────────────────────────────────────────────────┘
```

### Example 3: Factual Query with Retry

```
User Input: "What is the price of XYZ coin today?"

NODE 1: INPUT → query_type: "factual", sensitive_domain: ""
         ↓ → "search"
NODE 2: SEARCH → Results: sparse, unclear
         ↓
NODE 3: VERIFY → verification_passed: False, should_retry: True, retry_count: 1
         ↓ → "search" (RETRY)
NODE 2: SEARCH → Uses different query approach
         ↓
NODE 3: VERIFY → verification_passed: True
         ↓ → "final"
NODE 4: FINAL → Generates answer with sources
```

---

## 10. Testing

### 10.1 Test Structure

```
tests/
├── __init__.py
├── test_agent.py      # 21 tests
│   ├── TestAgentStructure (3 tests)
│   │   - test_agent_imports
│   │   - test_agent_builds_successfully
│   │   - test_agent_has_correct_nodes
│   ├── TestParsingFunctions (5 tests)
│   │   - test_parse_input_response_valid
│   │   - test_parse_input_response_invalid
│   │   - test_parse_input_response_malformed
│   │   - test_parse_verify_response_pass
│   │   - test_parse_verify_response_fail
│   ├── TestConstants (2 tests)
│   │   - test_max_retries_set
│   │   - test_prompts_defined
│   ├── TestAgentState (1 test)
│   │   - test_agent_state_fields
│   ├── TestRouting (5 tests)
│   │   - test_route_after_input_factual
│   │   - test_route_after_input_creative
│   │   - test_route_after_input_invalid
│   │   - test_route_after_verify_should_retry
│   │   - test_route_after_verify_done
│   └── TestSensitiveDomainDetection (5 tests)
│       - test_detect_medical_query
│       - test_detect_legal_query
│       - test_detect_financial_query
│       - test_detect_non_sensitive_query
│       - test_disclaimers_defined
│
└── test_service.py    # 9 tests
    ├── TestSchemas (4 tests)
    ├── TestCoreModules (3 tests)
    └── TestAPIEndpoints (2 tests)
```

### 10.2 Example Test Cases

```python
class TestSensitiveDomainDetection:
    """Tests for sensitive domain detection."""
    
    def test_detect_medical_query(self):
        from agents.chatbot import detect_sensitive_domain
        
        medical_queries = [
            "What should I take for a fever?",
            "How to treat headache pain?",
            "Is this medication safe during pregnancy?",
        ]
        
        for query in medical_queries:
            result = detect_sensitive_domain(query)
            assert result == "medical", f"Failed for: {query}"
    
    def test_detect_non_sensitive_query(self):
        from agents.chatbot import detect_sensitive_domain
        
        non_sensitive = [
            "What is the weather today?",
            "Who won the World Cup?",
            "How does photosynthesis work?",
        ]
        
        for query in non_sensitive:
            result = detect_sensitive_domain(query)
            assert result == "", f"Should be non-sensitive: {query}"
```

### 10.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with verbose output
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/test_agent.py -v

# Run specific test class
pytest tests/test_agent.py::TestSensitiveDomainDetection -v
```

### 10.4 Test Results

```
================= test session starts =================
tests/test_agent.py::TestAgentStructure::test_agent_imports PASSED
tests/test_agent.py::TestAgentStructure::test_agent_builds_successfully PASSED
tests/test_agent.py::TestAgentStructure::test_agent_has_correct_nodes PASSED
tests/test_agent.py::TestParsingFunctions::test_parse_input_response_valid PASSED
tests/test_agent.py::TestParsingFunctions::test_parse_input_response_invalid PASSED
tests/test_agent.py::TestParsingFunctions::test_parse_input_response_malformed PASSED
tests/test_agent.py::TestParsingFunctions::test_parse_verify_response_pass PASSED
tests/test_agent.py::TestParsingFunctions::test_parse_verify_response_fail PASSED
tests/test_agent.py::TestConstants::test_max_retries_set PASSED
tests/test_agent.py::TestConstants::test_prompts_defined PASSED
tests/test_agent.py::TestAgentState::test_agent_state_fields PASSED
tests/test_agent.py::TestRouting::test_route_after_input_factual PASSED
tests/test_agent.py::TestRouting::test_route_after_input_creative PASSED
tests/test_agent.py::TestRouting::test_route_after_input_invalid PASSED
tests/test_agent.py::TestRouting::test_route_after_verify_should_retry PASSED
tests/test_agent.py::TestRouting::test_route_after_verify_done PASSED
tests/test_agent.py::TestSensitiveDomainDetection::test_detect_medical_query PASSED
tests/test_agent.py::TestSensitiveDomainDetection::test_detect_legal_query PASSED
tests/test_agent.py::TestSensitiveDomainDetection::test_detect_financial_query PASSED
tests/test_agent.py::TestSensitiveDomainDetection::test_detect_non_sensitive_query PASSED
tests/test_agent.py::TestSensitiveDomainDetection::test_disclaimers_defined PASSED
tests/test_service.py::TestSchemas::test_chat_request_schema PASSED
tests/test_service.py::TestSchemas::test_chat_request_minimal PASSED
tests/test_service.py::TestSchemas::test_chat_response_schema PASSED
tests/test_service.py::TestSchemas::test_service_info_schema PASSED
tests/test_service.py::TestCoreModules::test_settings_import PASSED
tests/test_service.py::TestCoreModules::test_get_model_function_exists PASSED
tests/test_service.py::TestCoreModules::test_provider_enum PASSED
tests/test_service.py::TestAPIEndpoints::test_app_exists PASSED
tests/test_service.py::TestAPIEndpoints::test_app_has_routes PASSED
================= 30 passed in 0.63s ==================
```

---

## 11. Evaluation Criteria Mapping

### 11.1 LangGraph Correctness and Design Quality

| Aspect | Implementation | Evidence |
|--------|----------------|----------|
| **Proper StateGraph usage** | ✅ Uses `StateGraph(AgentState)` with `TypedDict` | `chatbot.py:build_research_agent()` |
| **Proper node definition** | ✅ 4 async nodes with proper signatures | `input_node`, `search_node`, `verify_node`, `final_node` |
| **Conditional edges** | ✅ Uses `add_conditional_edges` for routing | 2 conditional edge sets |
| **State management** | ✅ Explicit state with typed fields | `AgentState` TypedDict |
| **Entry/Exit points** | ✅ `set_entry_point`, `add_edge` to `END` | Properly configured |
| **Retry mechanism** | ✅ `should_retry` flag with `MAX_RETRIES=2` | Verify node controls retry |

### 11.2 Code Readability and Maintainability

| Aspect | Implementation | Evidence |
|--------|----------------|----------|
| **Modular structure** | ✅ Separated into `agents/`, `core/`, `service/`, `schema/` | Project structure |
| **Clear naming** | ✅ Descriptive function/variable names | `input_node`, `detect_sensitive_domain` |
| **Type hints** | ✅ Used throughout | All functions typed |
| **Docstrings** | ✅ All functions documented | Purpose explained |
| **Logging** | ✅ Comprehensive logging | `logger.info()` in all nodes |
| **Error handling** | ✅ Try/except with graceful fallbacks | Every node handles errors |
| **Constants separated** | ✅ Prompts and keywords at top | Easy to modify |

### 11.3 Clarity of Reasoning and Outputs

| Aspect | Implementation | Evidence |
|--------|----------------|----------|
| **Clean output format** | ✅ `## Answer`, `## Key Points`, `## Sources` | Not exposing internal reasoning |
| **Source citations** | ✅ Required in prompts | `FINAL_PROMPT` rules |
| **Confidence indication** | ✅ Verification status passed to final | Partial results noted |
| **Sensitive domain handling** | ✅ Safe refusal, no actionable advice | Skip search, use templates |
| **Anti-hallucination** | ✅ Low temp, verified facts only | `temperature=0.3` |

### 11.4 Repository Structure and Documentation

| Aspect | Implementation | Evidence |
|--------|----------------|----------|
| **Clear folder structure** | ✅ Logical separation | 5 modules + tests |
| **README quality** | ✅ Architecture, setup, API docs, examples | ~600 lines |
| **Environment template** | ✅ `.env.example` with all variables | Documented |
| **License** | ✅ MIT License included | `LICENSE` file |
| **Requirements** | ✅ Pinned versions | `requirements.txt` |
| **Tests** | ✅ 30 passing tests | `pytest.ini` configured |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Python files** | 10 |
| **Lines of code** | ~2,550 |
| **Unit tests** | 30 |
| **Test coverage areas** | Agent, parsing, routing, sensitive detection, API |
| **LLM providers supported** | 4 (Groq, OpenAI, Gemini, Anthropic) |
| **LangGraph nodes** | 4 |
| **Conditional routing edges** | 2 |
| **Sensitive domains handled** | 3 (medical, legal, financial) |

---

*Document generated for assessment purposes. This project demonstrates a production-ready AI research agent implementation using LangGraph.*
