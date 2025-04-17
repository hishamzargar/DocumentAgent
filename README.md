# DocuAgent 🤖📄

DocuAgent is an AI-powered application that allows you to upload documents (PDF and TXT) and ask questions about their content. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide contextually relevant answers based on the uploaded information.

## Features

* **Document Upload:** Supports uploading `.pdf` and `.txt` files via a REST API.
* **Content Querying:** Ask questions about the content of uploaded documents via a REST API.
* **RAG Pipeline:** Uses LangChain to orchestrate a RAG pipeline involving:
    * Document loading (PyPDF, TextLoader)
    * Text splitting (RecursiveCharacterTextSplitter)
    * Embeddings generation (Sentence Transformers via HuggingFaceEmbeddings)
    * Vector storage (ChromaDB)
    * Retrieval and Large Language Model (LLM) interaction (using Azure OpenAI GPT-4o via LCEL)
* **API:** Built with FastAPI, providing interactive documentation via Swagger UI.
* **Containerized:** Dockerfile provided for easy packaging and deployment.
* **CI/CD:** Basic Continuous Integration setup using GitHub Actions (linting, testing).

## Technology Stack

* **Backend:** Python 3.x, FastAPI
* **AI/LLM:** LangChain, Azure OpenAI (GPT-4o), Sentence Transformers (all-MiniLM-L6-v2)
* **Vector Database:** ChromaDB
* **Containerization:** Docker
* **CI/CD:** GitHub Actions
* **Testing:** Pytest

## Project Structure
DocuAgentProject/
├── .github/            # GitHub Actions workflows
│   └── workflows/
│       └── ci.yaml
├── app/                # Core application source code
│   ├── init.py
│   ├── main.py         # FastAPI endpoints
│   └── rag_processor.py  # RAG pipeline logic
├── tests/              # Tests
│   ├── init.py
│   ├── test_api.py     # API tests
│   └── test_data/      # Sample data for tests
│       └── test_upload.txt
├── .dockerignore       # Files ignored by Docker build
├── .env.example        # Example environment variables
├── .gitignore          # Files ignored by Git
├── Dockerfile          # Docker build instructions
├── launch.json         # VS Code debug config (optional)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── venv/               # Python virtual environment (ignored)

chroma_db/ <-- Created at runtime (ignored)
temp_uploads/ <-- Created at runtime (ignored)

## Setup and Installation

**Prerequisites:**

* Python (version 3.9 or later recommended)
* pip (Python package installer)
* Git
* Docker Desktop (or Docker Engine)

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd DocuAgentProject
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    * Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
        *(You should create a `.env.example` file listing the needed variables without their values, and add it to git).*
    * Edit the `.env` file and add your actual API keys and endpoints:
        ```dotenv
        # Example .env content
        AZURE_OPENAI_ENDPOINT="YOUR_AZURE_ENDPOINT_URL"
        AZURE_OPENAI_KEY="YOUR_AZURE_KEY"
        AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o" # Or your deployment name
        # Add other keys if needed
        ```
        **Note:** The `.env` file is included in `.gitignore` and should **never** be committed to version control.

## Running the Application

**1. Locally (without Docker):**

```bash
# Ensure venv is active and .env file is populated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Using Docker

1.  **Build the image:**
    ```bash
    docker build -t docuagent .
    ```

2.  **Run the container:**
    ```bash
    docker run --rm -p 8000:8000 --env-file .env -v ./chroma_db:/app/chroma_db --name docuagent-container docuagent
    ```
    The API will be available at `http://localhost:8000`. The `chroma_db` data will persist in your local `./chroma_db` directory mapped into the container.

## API Endpoints

Access the interactive Swagger UI documentation by navigating to `http://localhost:8000/docs` in your browser when the application is running.

* **`POST /upload`**: Upload a `.txt` or `.pdf` file.
    * **Request:** `multipart/form-data` with a `file` field containing the document.
    * **Response:** `200 OK` with success message or `4xx/5xx` on error.

* **`POST /query`**: Ask a question about the uploaded documents.
    * **Request Body (JSON):** `{ "query": "Your question here" }`
    * **Response Body (JSON):** `{ "answer": "LLM response", "source_documents": [] }`
        **Note: Retrieval of detailed source document information in the response is not currently implemented.**

**Example using `curl`:**

```bash
# Upload a file (replace with your file path)
curl -X POST -F 'file=@./path/to/your/document.pdf' http://localhost:8000/upload

# Query the content
curl -X POST -H "Content-Type: application/json" -d '{"query": "What is the main topic?"}' http://localhost:8000/query
```

## Testing

```bash
pytest
```

## CI/CD

**A basic Continuous Integration (CI) pipeline is configured using GitHub Actions (.github/workflows/ci.yaml). It automatically runs linters (flake8, black) and tests (pytest) on pushes and pull requests targeting the main branch to ensure code quality and functionality.**
