"""
FastAPI service for the AI Research Agent.
Provides REST endpoints for chat interactions with multiple LLM providers.
"""
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from agents import chatbot
from core import settings, get_available_models
from schema import ChatRequest, ChatResponse, ServiceInfo, ProviderInfo, ModelInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory saver for conversation persistence
memory = MemorySaver()

# Provider display names
PROVIDER_NAMES = {
    "groq": "Groq (Llama)",
    "openai": "OpenAI (GPT)",
    "gemini": "Google Gemini",
    "anthropic": "Anthropic Claude",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    logger.info(f"🚀 Starting {settings.APP_NAME}")
    logger.info(f"📦 Default provider: {settings.DEFAULT_PROVIDER}")
    logger.info(f"🧠 Default model: {settings.DEFAULT_MODEL}")
    
    # Set up the chatbot with memory
    chatbot.checkpointer = memory
    
    # Log available providers
    available = get_available_models()
    logger.info(f"✅ Available providers: {list(available.keys())}")
    
    yield
    
    logger.info(f"👋 Shutting down {settings.APP_NAME}")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI Research Agent with multiple LLM providers (Groq, OpenAI, Gemini, Claude)",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=ServiceInfo)
async def root() -> ServiceInfo:
    """Get service information including available providers and models."""
    available = get_available_models()
    
    providers = []
    for provider_id, models in available.items():
        providers.append(ProviderInfo(
            id=provider_id,
            name=PROVIDER_NAMES.get(provider_id, provider_id.title()),
            models=[ModelInfo(id=m["id"], name=m["name"]) for m in models]
        ))
    
    return ServiceInfo(
        name=settings.APP_NAME,
        version="1.0.0",
        default_provider=settings.DEFAULT_PROVIDER.value,
        default_model=settings.DEFAULT_MODEL,
        providers=providers,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def get_models() -> dict[str, Any]:
    """Get available models grouped by provider."""
    return get_available_models()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message and get a step-by-step response from the AI agent.
    
    The agent will:
    1. Analyze the question (Router Node)
    2. Search the web if needed (Search Node)
    3. Verify facts from search results (Verifier Node)
    4. Synthesize a step-by-step answer (Synthesizer Node)
    """
    thread_id = request.thread_id or str(uuid4())
    provider = request.provider or settings.DEFAULT_PROVIDER.value
    model = request.model or settings.DEFAULT_MODEL
    
    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "provider": provider,
            "model": model,
        }
    )
    
    try:
        # Invoke the chatbot
        result = await chatbot.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
        
        # Get the last AI message
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            response_text = last_message.content
        else:
            response_text = str(last_message.content)
        
        return ChatResponse(
            message=response_text,
            thread_id=thread_id,
            provider=provider,
            model=model,
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Stream a response from the AI agent.
    
    Args:
        request: Chat request with message and optional thread_id
        
    Returns:
        Server-sent events stream of the response
    """
    thread_id = request.thread_id or str(uuid4())
    provider = request.provider or settings.DEFAULT_PROVIDER.value
    model = request.model or settings.DEFAULT_MODEL
    
    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "provider": provider,
            "model": model,
        }
    )
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for event in chatbot.astream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(event, tuple):
                    message, metadata = event
                    if isinstance(message, AIMessage) and message.content:
                        yield f"data: {message.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/history/{thread_id}")
async def get_history(thread_id: str) -> dict[str, Any]:
    """
    Get conversation history for a thread.
    
    Args:
        thread_id: The thread ID to get history for
        
    Returns:
        List of messages in the conversation
    """
    config = RunnableConfig(configurable={"thread_id": thread_id})
    
    try:
        state = await chatbot.aget_state(config)
        messages = []
        
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        return {"thread_id": thread_id, "messages": messages}
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
