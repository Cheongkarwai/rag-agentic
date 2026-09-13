import os
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.utilities import PythonREPL

os.environ["ANONYMIZED_TELEMETRY"] = "False"

class GeminiEmbeddings(Embeddings):
    """Native Gemini embeddings supporting models/gemini-embedding-001."""
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        self.api_key = api_key
        self.model = model if model.startswith("models/") else f"models/{model}"

    def embed_documents(self, texts: list) -> list:
        results = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents?key={self.api_key}"
            payload = {
                "requests": [
                    {"model": self.model, "content": {"parts": [{"text": t}]}}
                    for t in batch
                ]
            }
            resp = requests.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini Embedding Error ({resp.status_code}): {resp.text}")
            for e in resp.json().get("embeddings", []):
                results.append(e["values"])
        return results

    def embed_query(self, text: str) -> list:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent?key={self.api_key}"
        payload = {"model": self.model, "content": {"parts": [{"text": text}]}}
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini Embedding Error ({resp.status_code}): {resp.text}")
        return resp.json()["embedding"]["values"]

def get_embeddings():
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            provider = "openai"

    if provider == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found. Please set GEMINI_API_KEY.")
        return GeminiEmbeddings(api_key=gemini_key)
    else:
        openai_key = os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )

@tool
def query_knowledge_base(query: str) -> str:
    """Searches the document vector store repository to retrieve contextual definitions,
    background information, guidelines, and textual research records."""
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    chroma_dir = base_dir / "chroma_db"
    persist_dir = str(chroma_dir) if chroma_dir.exists() else "./chroma_db"

    if not os.path.exists(persist_dir):
        return "Error: The vector knowledge store directory has not been initialized."

    embeddings = get_embeddings()
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    docs = db.similarity_search(query, k=3)

    context_output = []
    for i, doc in enumerate(docs):
        context_output.append(f"[Document Source {i+1}]:\n{doc.page_content}")
    return "\n\n".join(context_output)

@tool
def execute_python_analysis(code: str) -> str:
    """Executes structural Python code locally to handle complex calculations,
    process pandas DataFrames, or generate statistical values. Inputs must be pure code."""
    try:
        repl = PythonREPL()
        execution_result = repl.run(code)
        if not execution_result.strip():
            return "Execution completed successfully, but no console output was printed."
        return f"Execution Success:\n{execution_result}"
    except Exception as error:
        return f"Execution Error: {str(error)}. Review code variables, resolve syntax, and retry."
