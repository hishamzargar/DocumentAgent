import os
import sentence_transformers
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
#Import LCEL components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI # Or your chosen LLM
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI

import traceback

# Load environment variables
load_dotenv()

# --- Azure OpenAI Configuration ---
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_KEY")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_api_version = "2024-12-01-preview"

if not all([azure_endpoint, azure_key, azure_deployment_name]):
    print("Warning: Azure OpenAI environment variables (ENDPOINT, KEY, DEPLOYMENT_NAME) not fully set.")

# Constants
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "docuagent_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

#check if llm key exists
if os.getenv("OPENAI_API_KEY") is None:
    print("Warning: LLM API Key environment variable not set.")


# --- Singleton Instances (manage resources efficiently) ---
_embedding_function = None
_vector_store = None
_rag_chain = None

def get_embedding_function():
    """ Initialize and return singleton embedding function"""
    global _embedding_function
    if _embedding_function is None:
       print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
       _embedding_function = HuggingFaceEmbeddings(
           model_name=EMBEDDING_MODEL_NAME, 
           model_kwargs = {'device': 'mps'})
       print("Embedding model initialized.")
    return _embedding_function

def get_vector_store():
    """Initializes and returns a singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        print(f"Accessing ChromaDB persistence directory: {CHROMA_DB_DIR}")
        os.makedirs(CHROMA_DB_DIR, exist_ok=True) # Ensure directory exists
        embedding_function = get_embedding_function()
        _vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        print(f"Vector store collection '{COLLECTION_NAME}' accessed/created.")
        try:
            print(f"Initial vector store count: {_vector_store._collection.count()}")
        except Exception as e:
            print(f"Could not get initial vector store count: {e}")
    return _vector_store

def get_rag_chain():
    """Initializes and returns a singleton LCEL RAG chain."""
    global _rag_chain
    if _rag_chain is None:
        print("Initializing LCEL RAG chain")

        if not all([azure_endpoint, azure_key, azure_deployment_name]):
             raise ValueError("Azure OpenAI environment variables not configured.")
        
        print(f"Initializing LCEL RAG chain with Azure OpenAI (Deployment: {azure_deployment_name})...")

        #1. Define prompt template
        template = """Answer the following question based only on the provided context:
                    Context:
                    {context}
                    Question: {question}
                    Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        #2. Get Components
        try:
            llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                azure_deployment=azure_deployment_name,
                api_version=azure_api_version,
                temperature=0
            )
            print("AzureChatOpenAI client initialized.")
        except Exception as e:
            print(f"Error initializing AzureChatOpenAI: {e}")
            raise e # Re-raise the error to be caught by caller
        vector_store = get_vector_store()
        if vector_store is None:
            raise ValueError("Vector store not initialized. Cannot create RAG chain.")
        retriever = vector_store.as_retriever()

        #3 format retrieved documents 
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 4. Construct the LCEL chain
        #    - Retrieve documents based on the question.
        #    - Format the documents into a single context string.
        #    - Pass the context and original question to the prompt.
        #    - Pass the formatted prompt to the LLM.
        #    - Parse the LLM output as a string.

        _rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("LCEL RAG chain initialized")
    return _rag_chain

# Processing Function
def load_and_split_document(file_path, chunk_size=1000, chunk_overlap=150):
    """Loads a document and splits it into chunks."""
    print(f"Processing document: {file_path}")
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    loader = None
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        print(f"Unsupported file format: {file_extension}")
        return None # Indicate failure

    if loader:
        try:
            documents = loader.load()
            if not documents:
                print("No content loaded from document.")
                return None
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Loaded and split into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            print(f"Error loading/splitting document {file_path}: {e}")
            return None
    return None

def add_document_to_store(file_path):
    """Loads, splits and adds documnet to the vector store"""
    chunks = load_and_split_document(file_path)
    if chunks:
        vector_store = get_vector_store()
        print(f"Adding {len(chunks)} chunks to vector store...")
        returned_ids = None
        try:
            current_count = vector_store._collection.count()
            print(f"Vector store count after add: {current_count}")
            print(f"Document {os.path.basename(file_path)} processed successfully up to count.")
        except Exception as e:
            print(f"ERROR occurred *during* or *immediately after* vector_store.add_documents: {e}")
            return False 
        return True
    else:
        print(f"Failed to process document {os.path.basename(file_path)}.")
        return False
    

def query_documnents(query_text):
    """Queries the documents using the QA chain"""
    print(f"Received query: '{query_text}'")
    rag_chain = get_rag_chain()
    vector_store  = get_vector_store()

    # Optional: Check if vector store is empty before querying
    if vector_store._collection.count() == 0:
        print("Vector store is empty. Cannot answer query.")
        return {"answer": "I haven't processed any documents yet. Please upload a document first.", "source_documents": []}
    
    #Invoke LCEL chain
    try:
        print(f"Invoking LCEL RAG chain with query: '{query_text}'")
        # The LCEL chain as defined expects the query string directly as input
        # because RunnablePassthrough() is used for the "question" field.
        answer = rag_chain.invoke(query_text)
        print(f"LCEL RAG chain executed. Answer: {answer[:100]}...")

        # --- Handling Source Documents (LCEL Basic) ---
        # The basic LCEL chain above doesn't easily return source documents.
        # To return sources, the chain needs modification (e.g., using RunnableParallel
        # to pass retriever output through). For now, returning empty list.
        formatted_sources = []
        print(f"Query answered (sources not retrieved in this basic LCEL setup).")

        return {f"answer": answer, "source_documents": formatted_sources}
    
    except Exception as e:
        print(f"Error during LCEL RAG chain execution: {e}")
        traceback.print_exc() # Print full traceback
        # Return structure consistent with expected QueryResponse
        return {"answer": f"An error occurred during RAG chain execution: {e}", "source_documents": []}



       