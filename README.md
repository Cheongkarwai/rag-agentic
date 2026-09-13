# Agentic RAG Research Assistant Deployment Guideline

## Project Initialization Guide

### 1. Extract Archive Package
Extract the `.zip` archive package into your preferred working directory.

### 2. Configure Python Virtual Environment
Set up a clean virtual environment (Python 3.9+ recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
The platform supports both **Google Gemini** (Free tier via Google AI Studio) and **OpenAI**. Configure either key in your environment or in a `.env` file:

- **Option A: Google Gemini (Recommended / Free Tier)**
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  ```
- **Option B: OpenAI**
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  ```
*(Alternatively, create a `.env` file from `.env.example`, or paste the API key directly into the sidebar of the web application).*

### 5. Ingest Raw Documents into Vector Database (Run First)
> [!IMPORTANT]
> **You must run `database_setup.py` before starting the web dashboard.**
> This script parses the source documents in `data/raw_documents/`, divides them into overlapping text chunks, generates vector embeddings, and creates the persistent local Chroma vector database (`chroma_db/`). Without this step, the RAG agent will have no document knowledge base to query.

Run the ingestion script:
```bash
python src/database_setup.py
```
*(Or `./venv/bin/python src/database_setup.py`)*

Upon successful completion, you will see:
```text
[*] Successfully processed text segments into chunks.
[*] Generating text embeddings and saving to database indices...
[✓] Database indexing complete. Saved to: '.../chroma_db'
```

### 6. Launch the Interactive Application Dashboard
Once the database has been indexed, launch the Streamlit interface:
```bash
python -m streamlit run src/app.py
```
*(Or `./venv/bin/streamlit run src/app.py`)*

**Running inside IntelliJ IDEA / PyCharm:**
- Right-click `src/app.py` and choose **Run 'app'** (direct IDE execution is fully supported).