# 🩺 Advanced Medical AI - "Chat with your PDFs"

This is an end-to-end **Retrieval-Augmented Generation (RAG)** application that builds a private "knowledge base" from a collection of PDF documents and allows you to "chat" with them. This demo uses medical case reports, but it can work with any set of text-based PDFs!

### Key Features

- **Chat History:** Remembers the context of your conversation.
- **Source Citations:** Proves its answers by citing the exact PDF filenames it used.
- **Live File Upload:** You can "teach" the AI new information by uploading new PDFs directly in the app.
- **100% Private Embeddings:** Uses a local `all-MiniLM-L6-v2` model to create embeddings, so your documents *never* leave your machine for this step.
- **Powerful AI:** Uses the Google Gemini API for state-of-the-art answer generation.

## Demo: How it Works

This demo shows all three advanced features in action.

**(1) RAG + Source Citations:** We can ask a complex question that requires information from all the *initial* documents. The app synthesizes a single answer and correctly cites all its sources.

**(2) Live File Upload (Before & After):** We can teach the AI new information live.

- **Before:** We ask about a topic *not* in the database ("MDMA and Parkinson's"). The app correctly states it doesn't have the information.
- **After:** We upload a new file (`newdoc_parkinsons.pdf`) using the sidebar. We ask the *exact same question*, and the app now provides a perfect answer, citing the new file as its source.

**(3) Chat History:** The app remembers the context of the conversation. We can ask a follow-up question ("which of those was in the 9-year-old child") that refers to our *very first question* ("complications"). The app understands "those" and provides the correct answer from the file we uploaded.

## Tech Stack

- **Python 3.9+**
- **Streamlit:** For the interactive web UI.
- **LangChain:** To build the complete RAG pipeline.
- **Gemini API:** For the generative "brain" (`gemini-2.5-flash-preview-09-2025`).
- **ChromaDB:** A local vector database to store document embeddings.
- **HuggingFace Embeddings:** A free, local model (`all-MiniLM-L6-v2`) to create vector embeddings privately.

## How to Run This Project

**Step 1: Setup Environment**
Clone this repository and navigate into it. Then, create and activate a virtual environment:

```
git clone [URL_TO_YOUR_REPO]
cd rag_medical_project

python3 -m venv venv
source venv/bin/activate

```

**Step 2: Install All Dependencies**
Install all required packages in one command using the `requirements.txt` file:

```
pip install -r requirements.txt

```

**Step 3: Add API Key**
Create a file named `.env` in the project root. Add your Google API key to it:

```
GOOGLE_API_KEY=YOUR_API_KEY_HERE

```

**Step 4: (Optional) Create Initial Database**
You can pre-load the app with a set of documents.

1. Place your PDFs into the `./documents` folder.
2. Run the `ingest_advanced.py` script *once*:
    
    ```
    python ingest_advanced.py
    
    ```
    

*(You can also skip this and build your database 100% live via the app's uploader.)*

**Step 5: Run the Advanced App**
Launch the Streamlit application:

```
streamlit run app_advanced.py

```

Your browser will open to `http://localhost:8501`. You can now chat with your documents and upload new ones via the sidebar.
