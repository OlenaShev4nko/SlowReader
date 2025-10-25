import os
import logging
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "mxbai-embed-large")

def get_embedding_ollama(text: str, model="mxbai-embed-large"):
    """Generate a dense embedding for the given text using Ollama."""
    try:
        emb_model = OllamaEmbeddings(model=model, base_url=OLLAMA_HOST)
        return emb_model.embed_query(text)
    except Exception as e:
        logger.error(f"get_embedding_ollama: Embedding failed for text: {text[:40]}... → {e}")
        return []
