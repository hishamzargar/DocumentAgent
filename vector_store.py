import chromadb
import os
from langchain.vectorstores import Chroma
from embedding_generator import get_embedding_function
from document_loader import load_document
from text_splitter import split_documents



CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME= "documentagent_collection"

def get_vector_store(embedding_function):
    """Initializes or loads the ChromaDB vector store."""
    print(f"Accessing ChromaDB persistence directory: {CHROMA_DB_DIR}")
    # Ensure the directory exists
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Get or create the collection with the specified embedding function
    vector_store = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Vector store collection '{COLLECTION_NAME}' accessed/created.")

    return vector_store, client

def add_chunks_to_store(vector_store, chunks):
    if not chunks:
        print("No chunks to add.")
        return
    
    print(f"Adding {len(chunks)} chunks to the vector store...")

    # Prepare data for ChromaDB: IDs, documents (text), and metadatas
    ids = [f"chunks_{i}" for i in range(len(chunks))]
    documents= [chunk.page_content for chunk in chunks]
    metadata= [chunk.metadata for chunk in chunks]

    # Add to the collection (ChromaDB will use its configured embedding function)
    # If using ChromaDB directly *without* LangChain's wrapper, you'd configure
    # the collection to use an embedding function on creation, or embed manually.
    # For simplicity here, let's assume we use LangChain's Chroma wrapper later,
    # which handles embedding transparently. This function shows the direct add structure.
    try:
        vector_store.add(
            ids=ids,
            documents=documents,
            metadata=metadata,
        )
    except Exception as e:
        print(f"Error adding chunks: {e}")



def query_vector_store(query, embedding_function, k=3):
    """Queries the vector store for relevant chunks."""
    print(f"\nQuerying store for: '{query}'")
    # Initialize the LangChain Chroma wrapper to load the existing store
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    # Perform similarity search
    results = vector_store.similarity_search(query, k=k)

    if results:
        print(f"Found {len(results)} relevant chunks:")
    else:
        print("No relevant chunks found.")
    
    return results




if __name__ == '__main__':


    #  Get Embedding Function
    embed_func = get_embedding_function()

    #Load and Split
    if not os.path.exists("example.txt"):
        with open("example.txt", 'w') as f:
            f.write("This is a document about ChromaDB vector store.\nIt helps in storing and retrieving vector embeddings for text data using similarity search.")

        docs = load_document("example.txt")
        if docs:
            chunks = split_documents(docs)

            if chunks:
                # Get Vector Store
                vector_store, client = get_vector_store(embed_func)

                # Add chunks
                # Clear collection before adding if running repeatedly for testing

                try:
                    print(f"Clearing collection '{COLLECTION_NAME}' before adding new chunks...")
                    client.delete_collection(name=COLLECTION_NAME)
                    vector_store, client = get_vector_store(embed_func) #recreate after delete
                    print("Collection cleared and recreated.")
                except Exception as e:
                    print(f"Could not delete collection (may not exist yet): {e}")

                add_chunks_to_store(vector_store, chunks)
                print(f"\nNumber of items in collection: {vector_store.count()}")

    if vector_store.count > 0:
        query = "What is ChromaDB?"
        relevent_chunks = query_vector_store(query, k=2)
    else:
        print("\nSkipping query as vector store is empty.")

    












    





