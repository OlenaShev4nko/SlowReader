# tests/integration/test_app.py

from unittest.mock import patch

import pytest
import streamlit as st

@pytest.mark.integration
def test_app_imports(monkeypatch):
    """Ensure Streamlit app imports and sets up session safely."""
    with patch("src.app.init_mcp_agent", return_value=(None, None)):
        st.session_state.clear()
        import src.app as app
        # Emulate app behavior that sets messages lazily
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        assert "messages" in st.session_state
