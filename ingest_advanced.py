import os
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants ---
DOCUMENTS_DIR = "./documents"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Embedding Model Initialization (Initialized once) ---
try:
    EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
except Exception as e:
    print(f"Error initializing embedding model: {e}")
    EMBEDDING_MODEL = None

# --- Core Functions (Meant to be imported) ---

def load_pdfs_from_directory(directory_path):
    """Loads and extracts text from all PDFs in a given directory."""
    documents_list = []
    print(f"Loading documents from {directory_path}...")
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            try:
                loader = pypdf.PdfReader(filepath)
                pdf_text = ""
                for page in loader.pages:
                    pdf_text += page.extract_text()
                
                documents_list.append({
                    "text": pdf_text,
                    "metadata": {"source": filename}
                })
                print(f"  - Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return documents_list

def load_pdf_from_upload(uploaded_file):
    """Loads and extracts text from a single uploaded PDF file."""
    if not uploaded_file:
        return None
    
    try:
        # pypdf can read from a file-like object
        loader = pypdf.PdfReader(uploaded_file)
        pdf_text = ""
        for page in loader.pages:
            pdf_text += page.extract_text()
        
        return {
            "text": pdf_text,
            "metadata": {"source": uploaded_file.name}
        }
    except Exception as e:
        print(f"Error loading uploaded file {uploaded_file.name}: {e}")
        return None

def split_documents(documents_list):
    """Splits loaded documents into smaller, overlapping chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_chunks = []
    for doc in documents_list:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })
            
    print(f"Created {len(all_chunks)} text chunks.")
    return all_chunks

def get_vector_store():
    """Initializes and returns the Chroma vector store."""
    if not EMBEDDING_MODEL:
        raise ValueError("Embedding model could not be initialized.")
        
    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=EMBEDDING_MODEL
    )
    return vector_db

def create_new_vector_store(chunks):
    """Creates a new vector database from text chunks."""
    if not chunks:
        print("No chunks to create database from.")
        return
    if not EMBEDDING_MODEL:
        raise ValueError("Embedding model could not be initialized.")

    print(f"Creating new vector database at {CHROMA_DB_DIR}...")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=EMBEDDING_MODEL,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_DIR
    )
    print("Vector database created successfully!")
    return vector_db

def add_docs_to_store(chunks, vector_store):
    """Adds new document chunks to an existing vector store."""
    if not chunks:
        print("No chunks to add.")
        return

    print(f"Adding {len(chunks)} new chunks to the database...")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print("Documents added successfully!")

# --- Main script (This part runs if you execute `python ingest_advanced.py`) ---
if __name__ == "__main__":
    if not EMBEDDING_MODEL:
        print("Could not initialize embedding model. Exiting.")
    else:
        # Check if DB already exists
        if os.path.exists(CHROMA_DB_DIR):
            print(f"Database already exists at {CHROMA_DB_DIR}. Skipping initial creation.")
            print("To force a rebuild, delete the 'chroma_db' folder and run this script again.")
        else:
            print("No database found. Building new one...")
            raw_documents = load_pdfs_from_directory(DOCUMENTS_DIR)
            
            if not raw_documents:
                print("No documents found. Exiting.")
            else:
                text_chunks = split_documents(raw_documents)
                if text_chunks:
                    create_new_vector_store(text_chunks)
                else:
                    print("No text could be extracted. Exiting.")