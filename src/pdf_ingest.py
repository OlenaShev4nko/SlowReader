import uuid
from pathlib import Path
from pypdf import PdfReader
import logging
import os
import asyncio
from datetime import datetime, timezone
from src.graph_pipeline import init_mcp_client
from src.utils_embeddings import get_embedding_ollama

logger = logging.getLogger(__name__)


def read_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50):
    """Split long text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


async def ingest_pdf_async(pdf_path: str, doc_id: str = None, meta: dict = None):
    """Full async pipeline for PDF ingestion → Neo4j via MCP."""
    path = Path(pdf_path)
    doc_id = doc_id or str(uuid.uuid4())
    meta = meta or {}
    logger.info(f"ingest_pdf_async: Ingesting '{path.name}' (id={doc_id})...")

    # Step 1 - Connect MCP client
    client = await init_mcp_client()
    logger.info("ingest_pdf_async: Connected to MCP client")

    # Step 2 - Create or reuse Document node
    doc_meta = {
        "id": doc_id,
        "name": meta.get("name", path.stem),
        "path": str(path.resolve()),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "author": meta.get("author", "Unknown"),
        "year": meta.get("year", "Unknown"),
    }

    cypher_doc = """
    MERGE (d:Document {name: $name, author: $author, year: $year})
    ON CREATE SET
        d.id = $doc_id,
        d.path = $path,
        d.uploaded_at = datetime($uploaded_at)
    RETURN d
    """

    params = {
        "doc_id": doc_meta["id"],
        "name": doc_meta["name"],
        "path": doc_meta["path"],
        "uploaded_at": doc_meta["uploaded_at"],
        "author": doc_meta["author"],
        "year": doc_meta["year"],
    }

    try:
        async with client.session("neo4j") as session:
            result = await session.call_tool(
                "write_neo4j_cypher", {"query": cypher_doc, "params": params}
            )

        raw = getattr(result, "content", [])
        logger.info(f"ingest_pdf_async: Created or reused Document node for '{path.name}' → {raw}")

    except Exception as e:
        logger.error(f"ingest_pdf_async: Failed to create/reuse Document node: {e}", exc_info=True)
        return {"error": str(e)}

    # Step 3 - Read and chunk text
    text = read_pdf(str(path))
    chunks = chunk_text(text)
    logger.info(f"ingest_pdf_async: Extracted {len(chunks)} chunks")

    # Step 4 - Embed and insert chunks
    for i, chunk in enumerate(chunks, 1):
        try:
            emb = get_embedding_ollama(chunk)
            chunk_id = str(uuid.uuid4())

            # ensure the index exists (first run only)
            create_index_cypher = """
            CREATE VECTOR INDEX chunk_embeddings
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
              }
            }
            """

            # Try creating once (ignore if exists)
            try:
                async with client.session("neo4j") as session:
                    await session.call_tool(
                        "write_neo4j_cypher",
                        {"query": create_index_cypher, "params": {"dim": len(emb)}}
                    )
            except Exception as idx_err:
                if "already exists" in str(idx_err):
                    logger.info("ingest_pdf_async: Vector index 'chunk_embeddings' already exists — skipping.")
                else:
                    logger.warning(f"ingest_pdf_async: Index creation skipped due to unexpected error: {idx_err}")

            # create chunk node + relationship
            cypher_chunk = """
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text, c.embedding = $embedding
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:HAS_CHUNK]->(c)
            RETURN c
            """

            params = {
                "chunk_id": chunk_id,
                "text": chunk,
                "embedding": emb,
                "doc_id": doc_id,
            }

            async with client.session("neo4j") as session:
                result = await session.call_tool(
                    "write_neo4j_cypher",
                    {"query": cypher_chunk, "params": params}
                )

            raw = getattr(result, "content", [])
            logger.info(f"ingest_pdf_async: [{i}/{len(chunks)}] Inserted chunk {chunk_id} ({len(emb)} dims) → {raw}")

        except Exception as e:
            logger.error(f"ingest_pdf_async: Failed to insert chunk {i}: {e}", exc_info=True)

    logger.info(f"ingest_pdf_async: Finished ingesting '{path.name}' ({len(chunks)} chunks total)")
    return {"doc_id": doc_id, "chunks": len(chunks)}


def ingest_pdf(pdf_path: str, doc_id: str = None, meta: dict = None):
    """Sync wrapper for compatibility with Streamlit app."""
    return asyncio.run(ingest_pdf_async(pdf_path, doc_id=doc_id, meta=meta))