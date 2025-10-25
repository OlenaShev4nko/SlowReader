# srs/pipeline.py
import os
import logging
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


def get_llm():
    """Return a configured Ollama LLM instance."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    logger.info(f"get_llm: Using Ollama model: {model_name} at {ollama_host}")

    return OllamaLLM(
        model=model_name,
        base_url=ollama_host,
        timeout=180,
    )


def create_chain():
    """Create a Runnable pipeline: PromptTemplate → OllamaLLM."""
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer clearly and concisely:\n\n{question}",
    )
    return prompt | llm


def run_query(user_input: str) -> str:
    """Send a text query to the Ollama LLM and return its response."""
    try:
        chain = create_chain()
        result = chain.invoke({"question": user_input})
        text = result if isinstance(result, str) else result.get("text", str(result))
        logger.info("run_query: LLM responded successfully")
        return text.strip()
    except Exception as e:
        logger.error(f"run_query: LLM query failed: {e}")
        return f"[Error] {e}"


async def run_query_async(user_input: str) -> str:
    """Async version of run_query using ainvoke."""
    try:
        chain = create_chain()
        result = await chain.ainvoke({"question": user_input})
        text = result if isinstance(result, str) else result.get("text", str(result))
        logger.info("run_query_async: LLM responded successfully")
        return text.strip()
    except Exception as e:
        logger.error(f"run_query_async: LLM query failed: {e}")
        return f"[Error] {e}"

