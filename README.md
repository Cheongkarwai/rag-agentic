# Agentic RAG Research Assistant Deployment Guideline

## Project Initialization Guide
1. Extract the `.zip` archive package content layout.
2. Configure a clean Python virtual infrastructure execution environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up an API key (Google Gemini or OpenAI):
   - **Google Gemini (Recommended / Free Tier via [Google AI Studio](https://aistudio.google.com)):**
     ```
     export GEMINI_API_KEY="your-gemini-api-key"
     ```
   - **OpenAI:**
     ```
     export OPENAI_API_KEY="your-openai-api-key"
     ```
   *(Alternatively, you can configure the API key in `.env` file in root directory, or simply enter the key directly in the web dashboard sidebar!)*
5. Initialize the interactive application dashboard:
   ```bash
   ./venv/bin/streamlit run src/app.py
   ```
   *(Or in IntelliJ IDEA: right-click `src/app.py` and select **Run 'app'**)*