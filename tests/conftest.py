# tests/conftest.py

import pytest
import requests
from neo4j import GraphDatabase

@pytest.fixture(scope="session")
def ollama_available():
    """Check if local Ollama server is running."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_available():
    """Check if Neo4j is reachable on bolt://localhost:7687"""
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "test"))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def mcp_available():
    """Check if local MCP service responds (adjust port)."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(("localhost", 8000))
    sock.close()
    return result == 0  # 0 means open port


@pytest.fixture(autouse=True)
def skip_if_ollama_not_running(request, ollama_available):
    if "integration" in request.keywords and not ollama_available:
        pytest.skip("Ollama not running at http://localhost:11434")
