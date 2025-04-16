from langchain.embeddings import SentenceTransformerEmbeddings


def get_embedding_function(model_name="all-MiniLM-L6-v2"):
    """Initializes and returns the embedding function."""
    print(f"Initializing embedding model: {model_name}")
    # device='mps'
    embedding_function = SentenceTransformerEmbeddings(
        model_name=model_name
    )
    print("Embedding model initialized.")
    return embedding_function



if __name__ == '__main__':
    embed_func = get_embedding_function()
    sample_text = 'This is a test sentence'
    embedding = embed_func.embed_query(sample_text)
    print(f"\nEmbedding for {sample_text}: ")
    print(f"\nVector dimension: {len(embedding)}")
    print(f"First 5 dimensions: {embedding[:5]}")

          
