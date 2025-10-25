# tests/unit/test_app_imports.py

from unittest.mock import patch
import streamlit as st

def test_app_imports_cleanly():
    """Ensure Streamlit app imports safely even if MCP service is offline."""
    st.session_state.clear()
    # Mock out the network-heavy call
    with patch("src.app.init_mcp_agent", side_effect=Exception("Simulated MCP offline")):
        import importlib
        app = importlib.import_module("src.app")
        # Ensure app handled the exception gracefully
        assert "mcp_initialized" in st.session_state
        assert st.session_state["mcp_initialized"] is False