import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path so 'from src....' imports resolve on any machine/OS
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# If executed directly with `python src/app.py` (e.g. from IDE Run button), delegate to Streamlit CLI
from streamlit.runtime import exists as _streamlit_exists
if not _streamlit_exists():
    from streamlit.web import cli as _stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(_stcli.main())

# Load environment variables (API Keys) from local system configuration
load_dotenv()

from src.agents import initialize_research_agent

# Configure Application Page Settings
st.set_page_config(
    page_title="Agentic RAG Analytics Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Advanced Agentic RAG Research Assistant")
st.caption("Module: INT4203E - Artificial Intelligence Assignment Prototype")

# 1. Sidebar Configuration Layout for Provider and Parameters
st.sidebar.header("⚙️ Model Provider Settings")

has_gemini_env = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
provider = st.sidebar.selectbox(
    "Active LLM Provider",
    ["Google Gemini", "OpenAI"],
    index=0 if has_gemini_env else 1
)

if provider == "Google Gemini":
    provider_code = "gemini"
    default_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY", "")
    api_key_input = st.sidebar.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        help="Get your free key from Google AI Studio (aistudio.google.com)"
    )
    model_choice = st.sidebar.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
    )
else:
    provider_code = "openai"
    default_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value=default_key,
        type="password",
        help="Enter your OpenAI API Key"
    )
    model_choice = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )

# Synchronize API keys and provider environment variables
if api_key_input:
    os.environ["LLM_PROVIDER"] = provider_code
    os.environ["LLM_MODEL"] = model_choice
    os.environ["LLM_API_KEY"] = api_key_input
    if provider_code == "gemini":
        os.environ["GEMINI_API_KEY"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
    else:
        os.environ["OPENAI_API_KEY"] = api_key_input

# 2. Validation Safeguard Layer for API Provisioning
if not api_key_input:
    st.error(f"⚠️ {provider} API Key missing. Please provide it in the sidebar or export it in your environment.")
    st.stop()

# 3. Session Initialization Layer for Core Execution Architecture
config_signature = f"{provider_code}:{model_choice}:{api_key_input}"
if "agent_instance" not in st.session_state or st.session_state.get("active_config") != config_signature:
    st.session_state.agent_instance = initialize_research_agent(
        provider=provider_code,
        model=model_choice,
        api_key=api_key_input
    )
    st.session_state.active_config = config_signature

if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []

st.sidebar.markdown("---")
st.sidebar.header("📊 System Operational Controls")
data_context = st.sidebar.selectbox(
    "Active Analytical Domain Source",
    ["Unstructured Knowledge Repository (PDF Vectors)", "Tabular Matrix Ledger (CSV Database)"]
)
st.sidebar.info(
    "This runtime operates on a deterministic reasoning configuration (Temperature: 0.0) "
    "incorporating a live self-correcting validation Critic node."
)

# Reset Session Actions
if st.sidebar.button("Clear Dynamic Run Logs"):
    st.session_state.execution_logs = []
    st.rerun()

# 4. Core User Prompt Communication Portal
user_query = st.text_input(
    "Submit Analytical Query Request:",
    placeholder="e.g., Search the files to calculate the year-on-year metric variance using Python."
)

if user_query:
    with st.spinner("Agent compiling operational execution graph routes..."):
        try:
            # Execute logic across the operational agent pipeline
            pipeline_payload = {"input": user_query}
            execution_response = st.session_state.agent_instance.invoke(pipeline_payload)

            # Store conversation traces locally for logging purposes
            st.session_state.execution_logs.append({
                "prompt": user_query,
                "output": execution_response.get("output", "Error compiling execution return.")
            })
        except Exception as system_fault:
            st.error(f"Execution Interruption Event: {str(system_fault)}")

# 5. Visual Logs Display Engine
if st.session_state.execution_logs:
    st.subheader("📈 Active Execution Log Records")
    for session_record in reversed(st.session_state.execution_logs):
        with st.container():
            st.markdown(f"**User Prompt Request:** {session_record['prompt']}")
            st.info(f"**Synthesized System Output:**\n\n{session_record['output']}")
            st.markdown("---")
