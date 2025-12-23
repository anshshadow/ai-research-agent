"""
Streamlit chat interface for the AI Research Agent.
Supports multiple LLM providers configurable from the UI.
"""
import streamlit as st
import requests
from uuid import uuid4

# Page configuration
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f0f1a 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #a0a0ff;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Provider badges */
    .provider-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .provider-groq {
        background: linear-gradient(90deg, #f97316 0%, #fb923c 100%);
        color: white;
    }
    
    .provider-openai {
        background: linear-gradient(90deg, #10a37f 0%, #34d399 100%);
        color: white;
    }
    
    .provider-gemini {
        background: linear-gradient(90deg, #4285f4 0%, #34a853 100%);
        color: white;
    }
    
    .provider-anthropic {
        background: linear-gradient(90deg, #d97706 0%, #fbbf24 100%);
        color: white;
    }
    
    /* Graph visualization */
    .graph-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }
    
    /* Status indicators */
    .status-active {
        color: #4ade80;
    }
    
    .status-waiting {
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Provider configurations with models
PROVIDERS = {
    "groq": {
        "name": "🟠 Groq (Llama)",
        "models": {
            "Llama 3.3 70B (Best)": "llama-3.3-70b-versatile",
            "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
            "Llama 3.2 90B Vision": "llama-3.2-90b-vision-preview",
            "Mixtral 8x7B": "mixtral-8x7b-32768",
            "Gemma 2 9B": "gemma2-9b-it",
        }
    },
    "openai": {
        "name": "🟢 OpenAI (GPT)",
        "models": {
            "GPT-4o (Best)": "gpt-4o",
            "GPT-4o Mini (Fast)": "gpt-4o-mini",
            "GPT-4 Turbo": "gpt-4-turbo",
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
        }
    },
    "gemini": {
        "name": "🔵 Google Gemini",
        "models": {
            "Gemini 2.0 Flash (Latest)": "gemini-2.0-flash-exp",
            "Gemini 1.5 Pro": "gemini-1.5-pro",
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.0 Pro": "gemini-1.0-pro",
        }
    },
    "anthropic": {
        "name": "🟡 Anthropic Claude",
        "models": {
            "Claude 3.5 Sonnet (Best)": "claude-3-5-sonnet-latest",
            "Claude 3.5 Haiku (Fast)": "claude-3-5-haiku-latest",
            "Claude 3 Opus": "claude-3-opus-latest",
            "Claude 3 Sonnet": "claude-3-sonnet-20240229",
        }
    },
}


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "provider" not in st.session_state:
        st.session_state.provider = "groq"
    if "model" not in st.session_state:
        st.session_state.model = "llama-3.3-70b-versatile"


def check_api_connection() -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_providers() -> dict:
    """Get available providers from the API."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def send_message(message: str, provider: str, model: str) -> str:
    """Send a message to the API and get a response."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "thread_id": st.session_state.thread_id,
                "provider": provider,
                "model": model,
            },
            timeout=180,  # Longer timeout for complex queries
        )
        response.raise_for_status()
        return response.json()["message"]
        
    except requests.exceptions.ConnectionError:
        return """❌ **Cannot connect to the API server**

Please start the backend service:

```bash
cd src
python run_service.py
```

Then refresh this page."""
    except requests.exceptions.Timeout:
        return "⏱️ **Request timed out**. The query might be too complex. Please try a simpler question."
    except Exception as e:
        return f"❌ **Error**: {str(e)}"


def main():
    """Main Streamlit app."""
    init_session_state()
    
    # Check API connection
    api_connected = check_api_connection()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Connection status
        if api_connected:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.markdown("""
            Start the backend:
            ```bash
            cd src
            python run_service.py
            ```
            """)
        
        st.divider()
        
        # Provider selection
        st.markdown("### 🏢 LLM Provider")
        
        provider_options = list(PROVIDERS.keys())
        provider_names = [PROVIDERS[p]["name"] for p in provider_options]
        
        selected_provider_idx = st.selectbox(
            "Select Provider",
            range(len(provider_options)),
            format_func=lambda x: provider_names[x],
            index=provider_options.index(st.session_state.provider) if st.session_state.provider in provider_options else 0,
            help="Choose the AI provider to use",
        )
        selected_provider = provider_options[selected_provider_idx]
        
        # Update provider in session
        if selected_provider != st.session_state.provider:
            st.session_state.provider = selected_provider
            # Reset model to first option of new provider
            first_model = list(PROVIDERS[selected_provider]["models"].values())[0]
            st.session_state.model = first_model
        
        st.divider()
        
        # Model selection
        st.markdown("### 🧠 Model")
        
        model_options = PROVIDERS[selected_provider]["models"]
        model_names = list(model_options.keys())
        model_ids = list(model_options.values())
        
        # Find current model index
        current_model_idx = 0
        if st.session_state.model in model_ids:
            current_model_idx = model_ids.index(st.session_state.model)
        
        selected_model_idx = st.selectbox(
            "Select Model",
            range(len(model_names)),
            format_func=lambda x: model_names[x],
            index=current_model_idx,
            help="Choose the specific model to use",
        )
        selected_model = model_ids[selected_model_idx]
        st.session_state.model = selected_model
        
        st.divider()
        
        # New conversation button
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid4())
            st.rerun()
        
        st.divider()
        
        # Agent flow diagram
        st.markdown("### 🔄 Agent Workflow")
        st.markdown("""
        <div class="graph-container">
        <span class="status-active">●</span> <b>INPUT</b> (Node 1)<br>
        &nbsp;&nbsp;&nbsp;↓ Sanitize & Validate<br>
        <span class="status-active">●</span> <b>SEARCH</b> (Node 2)<br>
        &nbsp;&nbsp;&nbsp;↓ Query & Search Web<br>
        <span class="status-active">●</span> <b>VERIFY</b> (Node 3)<br>
        &nbsp;&nbsp;&nbsp;├─ ✓ Pass → Final<br>
        &nbsp;&nbsp;&nbsp;└─ ✗ Fail → Retry Search<br>
        <span class="status-active">●</span> <b>FINAL</b> (Node 4)<br>
        &nbsp;&nbsp;&nbsp;↓ Synthesize Answer
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Session info
        st.markdown("### 📊 Session")
        st.markdown(f"**Thread**: `{st.session_state.thread_id[:8]}...`")
        st.markdown(f"**Messages**: {len(st.session_state.messages)}")
        st.markdown(f"**Provider**: {selected_provider}")
        st.markdown(f"**Model**: `{selected_model.split('/')[-1]}`")
    
    # Main content
    st.markdown("# 🔍 AI Research Agent")
    st.markdown(f"*Using **{PROVIDERS[selected_provider]['name']}** • Step-by-step answers with citations*")
    
    # Feature badges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("🌐 **Web Search**")
    with col2:
        st.markdown("✅ **Fact Verified**")
    with col3:
        st.markdown("📝 **Step-by-Step**")
    with col4:
        st.markdown("📚 **Cited Sources**")
    
    st.divider()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask any question... (e.g., 'What are the latest AI developments?')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner(f"🔍 Researching with {PROVIDERS[selected_provider]['name']}..."):
                response = send_message(prompt, selected_provider, selected_model)
            st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
