# tests/integration/test_mcp.py
import os
import pytest
import requests

@pytest.mark.integration
def test_mcp_service_running():
    """Check that MCP API responds on configured port."""
    port = os.getenv("NEO4J_MCP_PORT", "8080")
    url = f"http://localhost:{port}/api/mcp/health"

    try:
        resp = requests.get(url, timeout=5)
        assert resp.status_code in (200, 404), f"Unexpected status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        pytest.fail(f"MCP service not reachable on localhost:{port}")