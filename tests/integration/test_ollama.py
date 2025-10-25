# tests/integration/test_ollama.py

import pytest
import requests

@pytest.mark.integration
def test_ollama_chat_model():
    """Ensure Ollama chat model responds."""
    payload = {"model": "llama3.1:8b", "prompt": "Hello", "stream": False}
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)
    assert r.status_code == 200
    data = r.json()
    assert "response" in data and len(data["response"]) > 0


@pytest.mark.integration
def test_ollama_embeddings():
    """Ensure Ollama embeddings endpoint returns valid vector."""
    payload = {"model": "mxbai-embed-large", "prompt": "This is a test sentence."}
    r = requests.post("http://localhost:11434/api/embeddings", json=payload, timeout=20)
    assert r.status_code == 200
    data = r.json()
    embedding = data.get("embedding", [])
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    assert len(embedding) > 100
