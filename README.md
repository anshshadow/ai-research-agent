# 🔍 AI Research Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-764ABC?style=for-the-badge&logo=langchain&logoColor=white)

**A powerful AI research agent that searches the web, verifies facts, and delivers step-by-step answers with citations.**

[Features](#-features) •
[Architecture](#-4-node-langgraph-architecture) •
[Quick Start](#-quick-start) •
[API Docs](#-api-documentation) •
[Configuration](#%EF%B8%8F-configuration)

</div>

---

## 📖 Overview

This AI Research Agent is built with **LangGraph** and uses a sophisticated **4-node architecture** to process user queries. Unlike simple chatbots, this agent:

1. **Validates and sanitizes** user input
2. **Searches the web** for real-time information
3. **Verifies facts** before using them (with automatic retry on failure)
4. **Synthesizes step-by-step answers** with proper source citations

The agent minimizes hallucination by only using verified facts from web searches.

---

## ✨ Features

### Core Features
| Feature | Description |
|---------|-------------|
| 🛡️ **Input Validation** | Sanitizes queries, blocks harmful content, classifies query types |
| 🌐 **Web Search** | Real-time search using DuckDuckGo (10 results per query) |
| ✅ **Fact Verification** | Verifies search results before using them |
| 🔄 **Smart Retry** | Automatically retries with different queries if verification fails (max 2 retries) |
| 📝 **Step-by-Step Answers** | Structured responses with clear reasoning |
| 📚 **Source Citations** | Every fact is cited with its source |
| 🚫 **Anti-Hallucination** | Uses only verified facts, never makes up information |

### Multi-Provider LLM Support
| Provider | Models Available | Best For |
|----------|------------------|----------|
| 🟠 **Groq** | Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B, Gemma 2 9B | ⚡ Fastest inference |
| 🟢 **OpenAI** | GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo | 🎯 Best overall quality |
| 🔵 **Google** | Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash | 🖼️ Multimodal tasks |
| 🟡 **Anthropic** | Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus | 📖 Long-form reasoning |

---

## 🏗️ 4-Node LangGraph Architecture

The agent uses a **4-node graph** with conditional edges and a retry loop:

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                    LANGGRAPH AGENT                      │
                              │                                                         │
    User Query ──────────────▶│   ┌─────────┐    ┌─────────┐    ┌─────────┐            │
                              │   │  INPUT  │───▶│ SEARCH  │───▶│ VERIFY  │            │
                              │   │ Node 1  │    │ Node 2  │    │ Node 3  │            │
                              │   └────┬────┘    └────▲────┘    └────┬────┘            │
                              │        │              │              │                  │
                              │        │              │    ┌─────────┴─────────┐       │
                              │        │              │    │                   │       │
                              │        │              │    ▼                   ▼       │
                              │        │              │  FAIL              PASS        │
                              │        │              │    │                   │       │
                              │        │              └────┘                   │       │
                              │        │            (retry, max 2)             │       │
                              │        │                                       │       │
                              │        │ (creative/                            │       │
                              │        │  invalid)                             │       │
                              │        │                                       ▼       │
                              │        │                              ┌─────────┐      │
                              │        └─────────────────────────────▶│  FINAL  │      │
                              │                                       │ Node 4  │      │
                              │                                       └────┬────┘      │
                              │                                            │           │
                              └────────────────────────────────────────────┼───────────┘
                                                                           │
                                                                           ▼
                                                                   Step-by-Step Answer
```

### Node Details

#### Node 1: INPUT
| Task | Description |
|------|-------------|
| **Sanitize** | Removes harmful content, fixes obvious typos |
| **Validate** | Checks if query is answerable |
| **Classify** | Categorizes as `factual`, `creative`, or `conversational` |

**Output:**
- `sanitized_query` - Cleaned query
- `query_type` - Classification result
- `is_valid` - Boolean validation status

#### Node 2: SEARCH
| Task | Description |
|------|-------------|
| **Optimize Query** | Creates best search query for the question |
| **Web Search** | Searches DuckDuckGo for 10 results |
| **Handle Retries** | Uses different approach on retry attempts |

**Output:**
- `search_query` - The query used
- `search_results` - Raw search results

#### Node 3: VERIFY
| Task | Description |
|------|-------------|
| **Check Relevance** | Are results relevant to the question? |
| **Check Credibility** | Are sources trustworthy? |
| **Check Consistency** | Do sources agree? |
| **Decide** | PASS → Final, FAIL → Retry (max 2) |

**Output:**
- `verified_facts` - Extracted verified facts with sources
- `verification_passed` - Boolean status
- `retry_count` - Number of retries so far

#### Node 4: FINAL
| Task | Description |
|------|-------------|
| **Synthesize** | Combines verified facts into answer |
| **Format** | Creates step-by-step structure |
| **Cite** | Adds source citations for all facts |

**Output:**
- Step-by-step answer in markdown format

---

## 📋 Answer Format

Every factual answer follows this structured format:

```markdown
## 📋 Summary
[Brief 1-2 sentence direct answer]

## 🔍 Step-by-Step Analysis

### Step 1: Understanding the Question
[What the user is asking and key aspects]

### Step 2: Key Findings from Research
[Main verified facts with source citations]

### Step 3: Detailed Analysis
[In-depth explanation connecting the facts]

### Step 4: Additional Insights
[Related context or implications]

## ✅ Conclusion
[Final synthesized answer with confidence level]

## 📚 Sources Referenced
[List of all sources used]
```

---

## 🗂️ Project Structure

```
ai-agent-toolkit/
│
├── 📁 src/                          # Source code
│   │
│   ├── 📁 agents/                   # LangGraph agent
│   │   ├── __init__.py
│   │   └── chatbot.py               # 4-node graph definition
│   │
│   ├── 📁 core/                     # Core utilities
│   │   ├── __init__.py
│   │   ├── settings.py              # Environment configuration
│   │   └── llm.py                   # Multi-provider LLM factory
│   │
│   ├── 📁 schema/                   # Data models
│   │   └── __init__.py              # Pydantic schemas with validation
│   │
│   ├── 📁 service/                  # API layer
│   │   ├── __init__.py
│   │   └── service.py               # FastAPI endpoints
│   │
│   ├── run_service.py               # Backend entry point
│   └── streamlit_app.py             # Frontend UI
│
├── 📁 tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_agent.py                # Agent graph tests
│   └── test_service.py              # API service tests
│
├── 📄 .env                          # API keys (gitignored)
├── 📄 .env.example                  # Environment template
├── 📄 .gitignore                    # Git ignore rules
├── 📄 pytest.ini                    # Test configuration
├── 📄 requirements.txt              # Python dependencies
├── 📄 LICENSE                       # MIT License
└── 📄 README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** installed
- At least **one LLM API key** (Groq recommended for free tier)

### Step 1: Clone the Repository

```bash
git clone https://github.com/anshshadow/ai-research-agent.git
cd ai-research-agent
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Copy the example environment file and add your API key(s):

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your key(s)
```

**At least ONE API key is required:**

```env
# Groq (FREE - Recommended for testing)
# Get key at: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# OpenAI (Paid)
# Get key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini (FREE tier available)
# Get key at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Anthropic Claude (Paid)
# Get key at: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Step 5: Run the Application

Open **two terminals**:

**Terminal 1 - Backend API:**
```bash
cd src
python run_service.py
```

You should see:
```
INFO:     🚀 Starting AI Research Agent
INFO:     📦 Default provider: groq
INFO:     🧠 Default model: llama-3.3-70b-versatile
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Frontend UI:**
```bash
cd src
streamlit run streamlit_app.py
```

### Step 6: Open the App

Navigate to **http://localhost:8501** in your browser 🎉

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GROQ_API_KEY` | One of these | Groq API key for Llama models | - |
| `OPENAI_API_KEY` | required | OpenAI API key for GPT models | - |
| `GOOGLE_API_KEY` | | Google API key for Gemini models | - |
| `ANTHROPIC_API_KEY` | | Anthropic API key for Claude models | - |
| `HOST` | No | Server host | `0.0.0.0` |
| `PORT` | No | Server port | `8000` |
| `APP_NAME` | No | Application name | `AI Research Agent` |
| `DEBUG` | No | Debug mode | `false` |

### Available Models

#### Groq Models
| Model ID | Description |
|----------|-------------|
| `llama-3.3-70b-versatile` | **Recommended** - Best quality |
| `llama-3.1-8b-instant` | Fastest response |
| `llama-3.2-90b-vision-preview` | Vision capabilities |
| `mixtral-8x7b-32768` | Large context window |
| `gemma2-9b-it` | Efficient & capable |

#### OpenAI Models
| Model ID | Description |
|----------|-------------|
| `gpt-4o` | Latest GPT-4 Omni |
| `gpt-4o-mini` | Fast & affordable |
| `gpt-4-turbo` | GPT-4 with vision |
| `gpt-3.5-turbo` | Fast & cheap |

#### Google Gemini Models
| Model ID | Description |
|----------|-------------|
| `gemini-2.0-flash-exp` | Latest Gemini |
| `gemini-1.5-pro` | Best for complex tasks |
| `gemini-1.5-flash` | Fast & efficient |
| `gemini-1.0-pro` | Stable version |

#### Anthropic Claude Models
| Model ID | Description |
|----------|-------------|
| `claude-3-5-sonnet-latest` | Best overall |
| `claude-3-5-haiku-latest` | Fast & affordable |
| `claude-3-opus-latest` | Most capable |
| `claude-3-sonnet-20240229` | Balanced |

---

## 🌐 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### GET `/` - Service Info
Returns service metadata and available providers.

**Response:**
```json
{
  "name": "AI Research Agent",
  "version": "1.0.0",
  "default_provider": "groq",
  "default_model": "llama-3.3-70b-versatile",
  "providers": [
    {
      "id": "groq",
      "name": "Groq (Llama)",
      "models": [...]
    }
  ]
}
```

#### GET `/health` - Health Check
```json
{"status": "healthy"}
```

#### GET `/models` - Available Models
Returns all available models grouped by provider.

#### POST `/chat` - Send Message
Send a message and get a step-by-step response.

**Request:**
```json
{
  "message": "What are the latest AI developments?",
  "thread_id": "optional-thread-id",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile"
}
```

**Response:**
```json
{
  "message": "## 📋 Summary\n...",
  "thread_id": "abc123",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile"
}
```

#### POST `/chat/stream` - Stream Response
Same as `/chat` but returns Server-Sent Events for streaming.

#### GET `/history/{thread_id}` - Get Conversation History
Returns all messages for a conversation thread.

### Example: Python Client

```python
import requests

# Simple chat
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What is LangGraph and how does it work?",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile"
    }
)

print(response.json()["message"])
```

### Example: cURL

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing",
    "provider": "groq",
    "model": "llama-3.3-70b-versatile"
  }'
```

---

## 💡 Example Queries

The agent excels at questions requiring current, factual information:

### ✅ Great for:
- "What are the latest developments in AI agents?"
- "Compare Tesla Model 3 vs Model Y specifications"
- "What happened in tech news this week?"
- "Explain how LangGraph works with examples"
- "What are the current cryptocurrency prices?"
- "Who won the recent Grammy Awards?"

### 🎨 Also handles creative queries:
- "Write a poem about artificial intelligence"
- "Help me brainstorm startup ideas"
- "Explain this concept to a 5-year-old"

---

## 🛡️ Anti-Hallucination Design

| Layer | Mechanism |
|-------|-----------|
| **INPUT Node** | Validates input, rejects invalid queries |
| **SEARCH Node** | Uses real-time web search, not training data |
| **VERIFY Node** | Strictly verifies facts, retries if needed |
| **FINAL Node** | Uses ONLY verified facts with citations |
| **Temperature** | Set to 0.3 (low creativity, high accuracy) |
| **Prompts** | Include strict instructions to never fabricate |

---

## 🔧 Troubleshooting

### Common Issues

#### "API key not found" error
```
ValueError: At least one LLM API key must be set
```
**Solution:** Add at least one API key to your `.env` file.

#### "Cannot connect to API server"
```
ConnectionError: Cannot connect to the API server
```
**Solution:** Make sure the backend is running:
```bash
cd src
python run_service.py
```

#### "Module not found" error
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

#### Slow responses
**Solution:** 
- Use a faster model like `llama-3.1-8b-instant` or `gpt-4o-mini`
- Check your internet connection (web search required)

---

## 🧪 Testing

The project includes comprehensive unit tests for the agent and API.

### Run All Tests

```bash
# From project root
pytest tests/ -v

# Or with coverage
pytest tests/ -v --tb=short
```

### Test Categories

| Test File | What It Tests |
|-----------|---------------|
| `test_agent.py` | Agent structure, parsing functions, routing logic |
| `test_service.py` | API schemas, endpoints, core modules |

### Example Test Run

```bash
$ pytest tests/ -v

========================= test session starts =========================
tests/test_agent.py::TestAgentStructure::test_agent_imports PASSED
tests/test_agent.py::TestAgentStructure::test_agent_builds_successfully PASSED
tests/test_agent.py::TestAgentStructure::test_agent_has_correct_nodes PASSED
tests/test_agent.py::TestParsingFunctions::test_parse_input_response_valid PASSED
...
========================= 25 passed in 0.76s ==========================
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Agent Framework** | LangGraph 0.2+ |
| **LLM Integration** | LangChain |
| **Backend API** | FastAPI |
| **Frontend UI** | Streamlit |
| **Web Search** | DuckDuckGo |
| **Configuration** | Pydantic Settings |
| **Data Validation** | Pydantic |
| **Testing** | Pytest |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent framework
- [LangChain](https://www.langchain.com/) - LLM integration
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Streamlit](https://streamlit.io/) - Frontend framework
- [Groq](https://groq.com/) - Fast LLM inference

---

<div align="center">

**Built with ❤️ using LangGraph, FastAPI, and Streamlit**

[⬆ Back to Top](#-ai-research-agent)

</div>
