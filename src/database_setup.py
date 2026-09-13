import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.tools import get_embeddings

def run_database_ingestion():
    """Reads raw document entries, processes structural chunk distributions,
    and commits vector metrics to local disk storage."""

    source_directory = str(project_root / "data" / "raw_documents")
    persist_db_directory = str(project_root / "chroma_db")

    # 1. Verify existence of raw document drop directory
    if not os.path.exists(source_directory):
        os.makedirs(source_directory)
        print(f"[*] Created target directory: '{source_directory}'. Place your downloaded PDFs/Txt files here.")
        return

    # 2. Iterate and ingest target files dynamically
    loaded_documents = []
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)

        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            loaded_documents.extend(loader.load())
        elif file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path)
            loaded_documents.extend(loader.load())

    if not loaded_documents:
        print("[!] Ingestion aborted: No valid source .pdf or .txt records found in data folder.")
        return

    # 3. Implement strict recursive token boundaries matching Section 5 specifications
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )
    processed_chunks = splitter.split_documents(loaded_documents)
    print(f"[*] Successfully processed text segments into {len(processed_chunks)} isolated chunks.")

    # 4. Generate embeddings and persist text arrays locally
    print("[*] Generating text embeddings and saving to database indices...")
    embedding_engine = get_embeddings()

    Chroma.from_documents(
        documents=processed_chunks,
        embedding=embedding_engine,
        persist_directory=persist_db_directory
    )
    print(f"[✓] Database indexing complete. Saved to: '{persist_db_directory}'")

if __name__ == "__main__":
    run_database_ingestion()
