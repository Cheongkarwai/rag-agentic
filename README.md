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
     ```bash
     export GEMINI_API_KEY="your-gemini-api-key"
     ```
   - **OpenAI:**
     ```bash
     export OPENAI_API_KEY="your-openai-api-key"
     ```
   *(Alternatively, you can create a `.env` file from `.env.example`, or simply enter the key directly in the web dashboard sidebar!)*
5. Initialize the interactive application dashboard:
   ```bash
   streamlit run src/app.py
   ```
   *(Or in IntelliJ IDEA: right-click `src/app.py` and select **Run 'app'**)*