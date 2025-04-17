import os
import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import your FastAPI app instance

# Create a client instance based on your app
client = TestClient(app)


# --- Test Fixtures (Optional but useful) ---
@pytest.fixture(scope="module")
def test_txt_file_path():
    test_dir = "test_api_docs"
    os.makedirs(test_dir, exist_ok=True)
    file_path = os.path.join(test_dir, "test_upload.txt")
    with open(file_path, "w") as f:
        f.write("This document contains information about testing FastAPI endpoints.")
    yield file_path  # Provide the path to the test
    # Teardown
    # os.remove(file_path)
    # if not os.listdir(test_dir): # Remove dir if empty
    #     os.rmdir(test_dir)
    # Note: Cleanup might interfere if vector store persists between tests unexpectedly.
    # For true isolation, mocking the RAG backend or clearing the store might be needed.


# --- Test Functions ---


def test_root_path():
    """Test if the root path (if defined, e.g., for health check) is reachable."""
    # Assuming you add a @app.get("/") later:
    # response = client.get("/")
    # assert response.status_code == 200
    # assert response.json() == {"message": "Welcome to DocuAgent!"}
    pass  # No root path defined yet


def test_upload_txt_file(test_txt_file_path):
    """Test uploading a valid .txt file."""
    with open(test_txt_file_path, "rb") as f:
        response = client.post(
            "/upload", files={"file": ("test_upload.txt", f, "text/plain")}
        )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Document 'test_upload.txt' processed and added successfully."
    }
    # Add assertion to check if vector store count increased if possible/reliable


def test_upload_unsupported_file():
    """Test uploading an unsupported file type."""
    # Create a dummy file content in memory
    dummy_content = b"This is a dummy docx content."
    response = client.post(
        "/upload",
        files={
            "file": (
                "test_unsupported.docx",
                dummy_content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_query_after_upload(test_txt_file_path):
    """Test querying after a relevant document has been uploaded."""
    # Ensure the file is uploaded first (consider test order or fixtures)
    # Note: This test depends on the state left by test_upload_txt_file if run sequentially
    # Better approach: Upload within this test or use session-scoped fixtures.
    with open(test_txt_file_path, "rb") as f:
        client.post(
            "/upload", files={"file": ("test_upload.txt", f, "text/plain")}
        )  # Re-upload for test isolation if needed

    query = "What is this document about?"
    response = client.post("/query", json={"query": query})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # Basic check if the answer seems relevant (depends heavily on LLM)
    assert "testing" in data["answer"].lower() or "endpoints" in data["answer"].lower()
    assert "source_documents" in data
    # If sources are returned, check they look reasonable
    # assert len(data["source_documents"]) > 0
    # assert data["source_documents"][0]["source"] == "test_upload.txt" # Check if source metadata is correct


def test_query_empty_store():
    """Test querying when the vector store is expected to be empty (hard to guarantee without cleanup)."""
    # This test is difficult without proper state management (clearing the store)
    # Assuming the store might be empty initially or after cleanup:
    # query = "Query on potentially empty store"
    # response = client.post("/query", json={"query": query})
    # assert response.status_code == 200 # Or appropriate error code
    # assert "haven't processed any documents" in response.json()["answer"] or "store is empty" in response.json()["answer"]
    pass  # Skip or implement with store cleanup/mocking


def test_query_empty_input():
    """Test sending an empty query."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 400  # Should be caught by FastAPI/Pydantic
    assert "Query cannot be empty" in response.json()["detail"]
