# tests/test_pipeline.py

import pytest
from src import pipeline
from langchain.schema.runnable import RunnableSequence


def test_run_query_returns_string():
    """Ensure run_query invokes Ollama correctly."""
    result = pipeline.run_query("Say hello")
    assert isinstance(result, str)
    assert len(result.strip()) > 0

def test_create_chain_contains_question():
    """Unit test for prompt chain structure."""
    chain = pipeline.create_chain()
    assert isinstance(chain, RunnableSequence)
    schema = chain.input_schema.model_json_schema()
    assert "question" in schema["properties"]
