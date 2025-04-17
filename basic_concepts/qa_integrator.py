import os
from langchain.chains import retrieval_qa
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from basic_concepts.embedding_generator import get_embedding_function
from dotenv import load_dotenv

# load api keys
load_dotenv()

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "docuagent_collection"


def setup_qa_chain():
    """Sets up the RetrievalQA chain."""
    # initialize embedding function
    embedding_function = get_embedding_function()

    # load vector store
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME,
    )

    # check if vector store has documents

    try:
        print(f"Vector store count: {vector_store._collection.count()}")
        if vector_store._collection.count() == 0:
            print(
                "Warning: Vector store is empty. QA chain may not function correctly."
            )
    except Exception as e:
        print(f"Warning: Could not check vector store count: {e}")

    # initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # create retrievalQa chain
    qa_chain = retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retrieval=vector_store.as_retriever(),
        return_source_document=True,
    )
    print("RetrievalQA chain setup complete.")
    return qa_chain


if __name__ == "__main__":
    qa_chain = setup_qa_chain()
    query = "What is ChromaDB used for?"
    print(f"\nExecuting QA chain for query: '{query}'")

    try:
        result = qa_chain.invoke({"query": query})
        print("\n--- QA Result ---")
        print(f"Query: {query}")
        print(f"Answer: {result.get('result', 'N/A')}")

    except Exception as e:
        print(f"Error during QA chain execution: {e}")
