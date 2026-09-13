import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.tools import query_knowledge_base, execute_python_analysis

def initialize_research_agent(provider: str = None, model: str = None, api_key: str = None):
    # Determine provider (auto-detect if not specified)
    if not provider:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider.lower() == "gemini":
        gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        selected_model = model or os.getenv("LLM_MODEL", "gemini-3.6-flash")
        llm = ChatOpenAI(
            model=selected_model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0,
            openai_api_key=gemini_key
        )
    else:
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        selected_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(
            model=selected_model,
            temperature=0,
            openai_api_key=openai_key
        )

    tool_collection = [query_knowledge_base, execute_python_analysis]

    # Establish strict agent operating constraints
    system_instruction = """You are an elite Agentic RAG Research Assistant. 
    Your mission is to resolve user queries with total analytical precision.
    
    CRITICAL BEHAVIORS:
    1. Cross-Reference: Use `query_knowledge_base` to fetch background context before jumping to conclusions.
    2. Data Analysis: Use `execute_python_analysis` whenever calculation or dataset parsing is required.
    3. Self-Correction: If a tool returns an error or incomplete information, dynamically adjust your parameters, fix your code/query, and try a different approach. Do not present error logs as final answers."""

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Compile the active agent graph
    runtime_agent = create_openai_tools_agent(llm, tool_collection, agent_prompt)

    return AgentExecutor(
        agent=runtime_agent,
        tools=tool_collection,
        verbose=True,
        handle_parsing_errors=True
    )
