from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_loader import load_document
import os


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Split loaded documnets into smaller chunks"""
    print(f"Splitting {len(documents)} documnets sections into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # helpful for context
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    if not os.path.exists("example.txt"):
        with open("example.txt", "w") as f:
            f.write(
                "This is line one.\nThis is line two.\nThis is the third line, which is a bit longer.\nAnd the fourth line. "
            )

    docs = load_document("example.txt")

    if docs:
        chunks = split_documents(docs)
        if chunks:
            print(f"\nFirst chunk metadata. {chunks[0].metadata}")
            print(f"\nFirst chunk content. {chunks[0].page_content}")
