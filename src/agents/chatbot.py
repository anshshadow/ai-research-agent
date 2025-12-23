"""
Research Agent with 4-Node Architecture + Sensitive Domain Handling

SENSITIVE DOMAIN BEHAVIOR:
- Medical/Legal/Financial queries → SKIP SEARCH → Safe refusal + general info

OUTPUT FORMAT:
- Clean, user-facing format
- No internal reasoning exposed
"""
import logging
import re
from datetime import datetime
from typing import Annotated, Literal, Sequence

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from core import get_model, settings

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentState(TypedDict):
    """State that flows through all nodes."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sanitized_query: str
    query_type: str
    is_valid: bool
    validation_error: str
    sensitive_domain: str
    search_query: str
    search_results: str
    search_error: str
    verified_facts: str
    verification_passed: bool
    retry_count: int
    should_retry: bool
    provider: str
    model: str


# =============================================================================
# TOOLS & CONSTANTS
# =============================================================================

try:
    web_search = DuckDuckGoSearchResults(name="web_search", max_results=10)
except Exception as e:
    logger.error(f"Failed to initialize web search: {e}")
    web_search = None

current_date = datetime.now().strftime("%B %d, %Y")
MAX_RETRIES = 2


# =============================================================================
# SENSITIVE DOMAIN DETECTION
# =============================================================================

LEGAL_KEYWORDS = [
    r'\b(lawyer|attorney|legal|law|court|judge|lawsuit|sue|suing)\b',
    r'\b(contract|agreement|liability|negligence|malpractice)\b',
    r'\barrested?\b',  # arrest, arrested
    r'\b(police|criminal|crime|felony|misdemeanor)\b',
    r'\b(divorce|custody|child\s+support|alimony)\b',
    r'\b(will|estate|inheritance|trust|probate)\b',
    r'\b(patent|trademark|copyright|intellectual\s+property)\b',
    r'\b(immigration|visa|deportation|asylum)\b',
    r'\b(my\s+)?rights\b',  # "my rights", "rights"
]

FINANCIAL_KEYWORDS = [
    r'\b(invest|investment|stock|bond|portfolio|dividend)\b',
    r'\b(tax|taxes|irs|deduction|exemption|filing)\b',
    r'\b(loan|mortgage|debt|credit|interest\s+rate)\b',
    r'\b(retirement|401k|ira|pension|social\s+security)\b',
    r'\b(insurance|premium|deductible|coverage|claim)\b',
    r'\b(bankruptcy|foreclosure|collection)\b',
    r'\b(should\s+i\s+(buy|sell|invest))\b',
    r'\b(financial\s+advice|money\s+advice)\b',
]

MEDICAL_KEYWORDS = [
    r'\b(fever|headache|pain|symptom|disease|illness|medicine|medication|drug|pill|dose|dosage)\b',
    r'\b(doctor|physician|hospital|clinic|treatment|diagnosis|cure|therapy|surgery)\b',
    r'\b(cancer|diabetes|heart|blood\s+pressure|infection|virus|bacteria|allergy)\b',
    r'\b(pregnant|pregnancy|baby|infant|pediatric)\b',
    r'\b(mental\s+health|depression|anxiety|psychiatric|psychological)\b',
    r'\b(vaccine|vaccination|immunization)\b',
    r'\b(prescription|pharmacy|pharmacist)\b',
    r'\b(should\s+i\s+take|can\s+i\s+take|how\s+much|what\s+to\s+take)\b',
    r'\b(medical\s+advice|health\s+advice)\b',
]


def detect_sensitive_domain(query: str) -> str:
    """Detect if query falls into a sensitive domain."""
    query_lower = query.lower()
    
    for pattern in LEGAL_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "legal"
    
    for pattern in FINANCIAL_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "financial"
    
    for pattern in MEDICAL_KEYWORDS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "medical"
    
    return ""


# =============================================================================
# SAFE REFUSAL RESPONSES FOR SENSITIVE DOMAINS
# =============================================================================

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


# =============================================================================
# SYSTEM PROMPTS - Clean output format
# =============================================================================

INPUT_PROMPT = f"""You are an input processor. Today is {current_date}.

Analyze the input and output EXACTLY in this format:
SANITIZED_QUERY: [cleaned query]
QUERY_TYPE: [factual/creative/conversational]
IS_VALID: [true/false]
VALIDATION_NOTE: [note or OK]
SENSITIVE_DOMAIN: [medical/legal/financial/none]"""


SEARCH_PROMPT = f"""Create a search query for: {{query}}

Output ONLY the search query. Nothing else."""


VERIFY_PROMPT = f"""Today is {current_date}.

Check if the search results answer the query.

Output format:
VERIFICATION_STATUS: [PASS/FAIL]
VERIFIED_FACTS:
• [Fact] (Source: [source])"""


# Clean output format - no internal reasoning
FINAL_PROMPT = f"""Today is {current_date}.

Provide a clear, helpful answer. Use ONLY the verified facts provided.

FORMAT:
## Answer
[Direct answer to the question]

## Key Points
- [Key point 1 with source]
- [Key point 2 with source]
- [Additional points as needed]

## Sources
[List sources]

RULES:
- Be concise and direct
- Cite sources for facts
- If unsure, say so
- Do NOT include internal reasoning steps
- Do NOT expose verification process"""


SENSITIVE_RESPONSE_PROMPT = f"""Today is {current_date}.

This is a SENSITIVE query ({{}}).

Provide ONLY general, educational information. NO SPECIFIC ADVICE.

Your response should be 2-3 sentences of general context that could apply to anyone.
Do NOT recommend specific actions, medications, investments, or legal strategies.

Example good response: "Fever is a natural immune response. General approaches typically include rest and hydration. Persistent symptoms warrant professional evaluation."

Example bad response: "You should take ibuprofen 400mg every 6 hours" - DO NOT do this."""


CREATIVE_PROMPT = f"""Today is {current_date}.

Provide a helpful, creative response. Be engaging and clear."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_input_response(response_text: str, original_query: str) -> dict:
    """Parse input node response with fallback detection."""
    result = {
        "sanitized_query": "",
        "query_type": "factual",
        "is_valid": True,
        "validation_error": "",
        "sensitive_domain": ""
    }
    
    try:
        lines = response_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("SANITIZED_QUERY:"):
                result["sanitized_query"] = line.split(":", 1)[1].strip()
            elif line.startswith("QUERY_TYPE:"):
                qt = line.split(":", 1)[1].strip().lower()
                if qt in ["factual", "creative", "conversational"]:
                    result["query_type"] = qt
            elif line.startswith("IS_VALID:"):
                result["is_valid"] = "true" in line.lower()
            elif line.startswith("VALIDATION_NOTE:"):
                note = line.split(":", 1)[1].strip()
                if note.lower() != "ok":
                    result["validation_error"] = note
            elif line.startswith("SENSITIVE_DOMAIN:"):
                domain = line.split(":", 1)[1].strip().lower()
                if domain in ["medical", "legal", "financial"]:
                    result["sensitive_domain"] = domain
    except Exception as e:
        logger.error(f"Parse error: {e}")
    
    # Fallback: local detection
    if not result["sensitive_domain"]:
        result["sensitive_domain"] = detect_sensitive_domain(
            result["sanitized_query"] or original_query
        )
    
    return result


def parse_verify_response(response_text: str) -> dict:
    """Parse verification response."""
    result = {"verification_passed": False, "verified_facts": response_text}
    
    try:
        if "VERIFICATION_STATUS:" in response_text.upper():
            if "PASS" in response_text.upper().split("VERIFICATION_STATUS:")[1][:20]:
                result["verification_passed"] = True
    except Exception:
        pass
    
    return result


# =============================================================================
# NODE 1: INPUT NODE
# =============================================================================

async def input_node(state: AgentState, config: RunnableConfig) -> dict:
    """INPUT NODE: Validate, classify, detect sensitive domain."""
    logger.info("=" * 50)
    logger.info("INPUT NODE: Processing")
    
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", settings.DEFAULT_PROVIDER)
    model_name = configurable.get("model", settings.DEFAULT_MODEL)
    
    last_message = state["messages"][-1]
    user_input = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)
    
    logger.info(f"INPUT NODE: Query: {user_input[:80]}...")
    
    try:
        model = get_model(provider, model_name)
        
        messages = [
            SystemMessage(content=INPUT_PROMPT),
            HumanMessage(content=f"Input: {user_input}")
        ]
        
        response = await model.ainvoke(messages)
        parsed = parse_input_response(response.content, user_input)
        
        if not parsed["sanitized_query"]:
            parsed["sanitized_query"] = user_input
        
        logger.info(f"INPUT NODE: Type={parsed['query_type']}, Sensitive={parsed['sensitive_domain']}")
        
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
            "verification_passed": False,
            "search_error": "",
            "verified_facts": "",
        }
        
    except Exception as e:
        logger.error(f"INPUT NODE: Error - {e}")
        sensitive = detect_sensitive_domain(user_input)
        return {
            "sanitized_query": user_input,
            "query_type": "factual",
            "is_valid": True,
            "validation_error": "",
            "sensitive_domain": sensitive,
            "provider": provider,
            "model": model_name,
            "retry_count": 0,
            "should_retry": False,
            "verification_passed": False,
            "search_error": "",
            "verified_facts": "",
        }


# =============================================================================
# NODE 2: SEARCH NODE
# =============================================================================

async def search_node(state: AgentState, config: RunnableConfig) -> dict:
    """SEARCH NODE: Search the web."""
    logger.info("=" * 50)
    logger.info("SEARCH NODE: Searching")
    
    provider = state.get("provider", settings.DEFAULT_PROVIDER)
    model_name = state.get("model", settings.DEFAULT_MODEL)
    sanitized_query = state.get("sanitized_query", "")
    retry_count = state.get("retry_count", 0)
    
    if web_search is None:
        return {"search_query": "", "search_results": "", "search_error": "Search unavailable"}
    
    try:
        model = get_model(provider, model_name)
        
        search_messages = [
            SystemMessage(content=SEARCH_PROMPT.format(query=sanitized_query)),
            HumanMessage(content=sanitized_query)
        ]
        
        query_response = await model.ainvoke(search_messages)
        search_query = query_response.content.strip().strip('"').strip("'")
        
        logger.info(f"SEARCH NODE: Query: {search_query}")
        
        try:
            search_results = web_search.invoke(search_query)
            
            if not search_results or len(str(search_results)) < 50:
                return {
                    "search_query": search_query,
                    "search_results": "",
                    "search_error": "No results",
                }
            
            return {
                "search_query": search_query,
                "search_results": str(search_results),
                "search_error": "",
            }
            
        except Exception as e:
            logger.error(f"SEARCH NODE: Search failed - {e}")
            return {"search_query": search_query, "search_results": "", "search_error": str(e)}
            
    except Exception as e:
        logger.error(f"SEARCH NODE: Error - {e}")
        return {"search_query": "", "search_results": "", "search_error": str(e)}


# =============================================================================
# NODE 3: VERIFY NODE
# =============================================================================

async def verify_node(state: AgentState, config: RunnableConfig) -> dict:
    """VERIFY NODE: Verify search results."""
    logger.info("=" * 50)
    logger.info("VERIFY NODE: Verifying")
    
    provider = state.get("provider", settings.DEFAULT_PROVIDER)
    model_name = state.get("model", settings.DEFAULT_MODEL)
    sanitized_query = state.get("sanitized_query", "")
    search_results = state.get("search_results", "")
    search_error = state.get("search_error", "")
    retry_count = state.get("retry_count", 0)
    
    if search_error and not search_results:
        if retry_count < MAX_RETRIES:
            return {
                "verified_facts": "",
                "verification_passed": False,
                "should_retry": True,
                "retry_count": retry_count + 1,
            }
        else:
            return {
                "verified_facts": "No search results available.",
                "verification_passed": False,
                "should_retry": False,
                "retry_count": retry_count,
            }
    
    try:
        model = get_model(provider, model_name)
        
        verify_messages = [
            SystemMessage(content=VERIFY_PROMPT),
            HumanMessage(content=f"Query: {sanitized_query}\n\nResults:\n{search_results}")
        ]
        
        response = await model.ainvoke(verify_messages)
        parsed = parse_verify_response(response.content)
        
        if parsed["verification_passed"]:
            return {
                "verified_facts": parsed["verified_facts"],
                "verification_passed": True,
                "should_retry": False,
                "retry_count": retry_count,
            }
        else:
            if retry_count < MAX_RETRIES:
                return {
                    "verified_facts": parsed["verified_facts"],
                    "verification_passed": False,
                    "should_retry": True,
                    "retry_count": retry_count + 1,
                }
            else:
                return {
                    "verified_facts": parsed["verified_facts"],
                    "verification_passed": False,
                    "should_retry": False,
                    "retry_count": retry_count,
                }
                
    except Exception as e:
        logger.error(f"VERIFY NODE: Error - {e}")
        return {
            "verified_facts": "",
            "verification_passed": False,
            "should_retry": False,
            "retry_count": retry_count,
        }


# =============================================================================
# NODE 4: FINAL NODE
# =============================================================================

async def final_node(state: AgentState, config: RunnableConfig) -> dict:
    """FINAL NODE: Generate response. Handle sensitive domains with safe refusal."""
    logger.info("=" * 50)
    logger.info("FINAL NODE: Generating response")
    
    provider = state.get("provider", settings.DEFAULT_PROVIDER)
    model_name = state.get("model", settings.DEFAULT_MODEL)
    sanitized_query = state.get("sanitized_query", "")
    query_type = state.get("query_type", "factual")
    is_valid = state.get("is_valid", True)
    validation_error = state.get("validation_error", "")
    sensitive_domain = state.get("sensitive_domain", "")
    verified_facts = state.get("verified_facts", "")
    
    logger.info(f"FINAL NODE: Sensitive={sensitive_domain}")
    
    try:
        model = get_model(provider, model_name)
        
        # Handle invalid input
        if not is_valid:
            return {"messages": [AIMessage(content=f"I cannot process this request: {validation_error}")]}
        
        # =====================================================================
        # SENSITIVE DOMAIN: Safe refusal + general info
        # =====================================================================
        if sensitive_domain:
            logger.info(f"FINAL NODE: Generating safe refusal for {sensitive_domain}")
            
            # Generate brief general info
            general_prompt = SENSITIVE_RESPONSE_PROMPT.format(sensitive_domain)
            general_messages = [
                SystemMessage(content=general_prompt),
                HumanMessage(content=f"Query: {sanitized_query}")
            ]
            
            general_response = await model.ainvoke(general_messages)
            general_info = general_response.content.strip()
            
            # Format with appropriate refusal
            if sensitive_domain == "medical":
                final_response = MEDICAL_REFUSAL.format(general_info=general_info)
            elif sensitive_domain == "legal":
                final_response = LEGAL_REFUSAL.format(general_info=general_info)
            elif sensitive_domain == "financial":
                final_response = FINANCIAL_REFUSAL.format(general_info=general_info)
            else:
                final_response = f"⚠️ This appears to be a sensitive query. Please consult a professional.\n\n{general_info}"
            
            return {"messages": [AIMessage(content=final_response)]}
        
        # =====================================================================
        # CREATIVE/CONVERSATIONAL: Direct response
        # =====================================================================
        if query_type in ["creative", "conversational"]:
            messages = [
                SystemMessage(content=CREATIVE_PROMPT),
                HumanMessage(content=sanitized_query)
            ]
            response = await model.ainvoke(messages)
            return {"messages": [response]}
        
        # =====================================================================
        # FACTUAL: Use verified facts
        # =====================================================================
        context = f"""
Verified Information:
{verified_facts}

Query: {sanitized_query}
"""
        messages = [
            SystemMessage(content=FINAL_PROMPT),
            HumanMessage(content=context)
        ]
        
        response = await model.ainvoke(messages)
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"FINAL NODE: Error - {e}")
        return {"messages": [AIMessage(content=f"Error generating response: {str(e)}")]}


# =============================================================================
# ROUTING LOGIC
# =============================================================================

def route_after_input(state: AgentState) -> Literal["search", "final"]:
    """Route after INPUT node.
    
    SKIP SEARCH for:
    - Invalid input
    - Creative/conversational queries
    - SENSITIVE DOMAINS (medical, legal, financial)
    """
    is_valid = state.get("is_valid", True)
    query_type = state.get("query_type", "factual")
    sensitive_domain = state.get("sensitive_domain", "")
    
    if not is_valid:
        logger.info("ROUTER: Invalid → FINAL")
        return "final"
    
    # SENSITIVE: Skip search, go directly to safe refusal
    if sensitive_domain:
        logger.info(f"ROUTER: Sensitive ({sensitive_domain}) → FINAL (safe refusal)")
        return "final"
    
    if query_type in ["creative", "conversational"]:
        logger.info(f"ROUTER: {query_type} → FINAL")
        return "final"
    
    logger.info("ROUTER: Factual → SEARCH")
    return "search"


def route_after_verify(state: AgentState) -> Literal["search", "final"]:
    """Route after VERIFY node."""
    if state.get("should_retry", False):
        logger.info("ROUTER: Retry → SEARCH")
        return "search"
    
    logger.info("ROUTER: Done → FINAL")
    return "final"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

def build_research_agent() -> StateGraph:
    """Build the 4-node research agent."""
    logger.info("Building agent...")
    
    graph = StateGraph(AgentState)
    
    graph.add_node("input", input_node)
    graph.add_node("search", search_node)
    graph.add_node("verify", verify_node)
    graph.add_node("final", final_node)
    
    graph.set_entry_point("input")
    
    graph.add_conditional_edges(
        "input",
        route_after_input,
        {"search": "search", "final": "final"}
    )
    
    graph.add_edge("search", "verify")
    
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"search": "search", "final": "final"}
    )
    
    graph.add_edge("final", END)
    
    logger.info("Agent built successfully")
    return graph.compile()


# =============================================================================
# EXPORT
# =============================================================================

chatbot = build_research_agent()
