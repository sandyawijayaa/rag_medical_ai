import os
import streamlit as st
from dotenv import load_dotenv

# Import functions from our refactored ingest_advanced.py
from ingest_advanced import (
    get_vector_store,
    load_pdf_from_upload,
    split_documents,
    add_docs_to_store  # <-- This is the correct function name
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser 

# --- 1. SETTINGS & INITIALIZATION ---

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
    st.stop()

# Define the prompt template for the RAG chain
RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUESTION:
{question}

Based on the context provided, please answer the question clearly and concisely.
If you don't know the answer, just say that you don't know.
"""

# --- 2. LOADERS (Cached for performance) ---

@st.cache_resource
def load_llm():
    """Initializes and caches the Google Generative AI model."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()

@st.cache_resource
def load_retriever():
    """Initializes and caches the vector database retriever."""
    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        return retriever
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        st.info("Have you run `python ingest.py` or `python ingest_advanced.py` to create the initial database?")
        st.stop()

# --- 3. RAG CHAIN (With Source Citations) ---

def format_docs(docs):
    """Helper function to format retrieved documents into a string."""
    return "\n\n---\n\n".join([d.page_content for d in docs])

llm = load_llm()
retriever = load_retriever()

prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

rag_chain = (
    RunnableMap({"context": retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

retriever_chain = RunnableMap({
    "context": retriever,
    "question": RunnablePassthrough()
})

# --- 4. STREAMLIT APP UI ---

st.title("🩺 Advanced Medical AI (v2)")
st.subheader("With Chat History, Source Citations, and Live File Upload")

# --- FEATURE 3: File Uploader (in the sidebar) ---
with st.sidebar:
    st.header("Upload New PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF to add it to the knowledge base:", 
        type="pdf"
    )
    
    if st.button("Add Document"):
        if uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    # 1. Load the in-memory file
                    doc_list = [load_pdf_from_upload(uploaded_file)]
                    
                    # 2. Split it
                    chunks = split_documents(doc_list)
                    
                    # 3. Get the vector store and add to it
                    vector_store = get_vector_store()
                    
                    # --- THIS IS THE FIX ---
                    # Changed 'add_docs_to__store' (two underscores)
                    # to 'add_docs_to_store' (one underscore)
                    add_docs_to_store(chunks, vector_store) 
                    
                    st.success(f"Successfully added '{uploaded_file.name}' to the database!")
                    # We clear the retriever cache to force it to reload
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        else:
            st.warning("Please upload a PDF file first.")

# --- FEATURE 2: Chat History ---
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("Ask your question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Run the RAG chains to get answer and sources ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Get the answer
                answer = rag_chain.invoke(prompt)
                
                # 2. Get the sources (run the parallel chain)
                retrieved_docs = retriever_chain.invoke(prompt)
                
                # --- FEATURE 1: Source Citations ---
                sources_content = "### Sources:\n\n"
                unique_sources = set()
                # Access the "context" key from the retrieved_docs dictionary
                for doc in retrieved_docs["context"]:
                    unique_sources.add(doc.metadata["source"])
                
                if unique_sources:
                    for i, source in enumerate(unique_sources):
                        sources_content += f"{i+1}. {source}\n"
                else:
                    sources_content = "No sources found."

                # Combine answer and sources
                full_response = f"{answer}\n\n---\n{sources_content}"
                
                st.markdown(full_response)
                # Add assistant's full response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                # Add error to history
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})