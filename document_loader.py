import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()

def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    print(f"Loading Documents: {file_path}")

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        print(f"Unsupported file format: {file_extension}")
        return None
    
    try:
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} document sections.")
        return documents
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return None
    

    
if __name__ == '__main__':
    if not os.path.exists("example.txt"):
        with open("example.txt", 'w') as f:
            f.write("This is a sample text document.\nIt has multiple lines.")
    
    txt_docs = load_document("example.txt")
    if txt_docs:
        print(f"\nLoaded TXT Content (first doc): {txt_docs[0].page_content[:100]}...")

    



    





