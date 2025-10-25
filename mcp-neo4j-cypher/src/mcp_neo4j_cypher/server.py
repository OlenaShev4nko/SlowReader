# --------------------------------------------------------------------------
# Modified version of mcp-neo4j-cypher server from:
# https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher
#
# Original code © Neo4j Contrib contributors, licensed under Apache License 2.0.
# Modifications by Olena Shevchenko, 2025:
# - Added three custom MCP tools: link_concept_chunks, search_chunks_by_embedding, get_concept_subgraph
#    for hybrid RAG graph integration
# - Adjusted initialization logic for local embedding service
#
# This modified version remains under the Apache License 2.0.
# --------------------------------------------------------------------------

import json
import logging
import re
from typing import Any, Literal, Optional

from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import TextContent, ToolResult
from mcp.types import ToolAnnotations
from neo4j import AsyncDriver, AsyncGraphDatabase, Query, RoutingControl
from neo4j.exceptions import ClientError, Neo4jError
from pydantic import Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .utils import _truncate_string_to_tokens, _value_sanitize

logger = logging.getLogger("mcp_neo4j_cypher")


def _format_namespace(namespace: str) -> str:
    if namespace:
        if namespace.endswith("-"):
            return namespace
        else:
            return namespace + "-"
    else:
        return ""


def _is_write_query(query: str) -> bool:
    """Check if the query is a write query."""
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )


def create_mcp_server(
    neo4j_driver: AsyncDriver,
    database: str = "neo4j",
    namespace: str = "",
    read_timeout: int = 30,
    token_limit: Optional[int] = None,
    read_only: bool = False,
) -> FastMCP:
    mcp: FastMCP = FastMCP(
        "mcp-neo4j-cypher", dependencies=["neo4j", "pydantic"], stateless_http=True
    )

    namespace_prefix = _format_namespace(namespace)
    allow_writes = not read_only

    @mcp.tool(
        name=namespace_prefix + "get_neo4j_schema",
        annotations=ToolAnnotations(
            title="Get Neo4j Schema",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_neo4j_schema() -> list[ToolResult]:
        """
        List all nodes, their attributes and their relationships to other nodes in the neo4j database.
        This requires that the APOC plugin is installed and enabled.
        """

        get_schema_query = """
        CALL apoc.meta.schema();
        """

        def clean_schema(schema: dict) -> dict:
            cleaned = {}

            for key, entry in schema.items():
                new_entry = {"type": entry["type"]}
                if "count" in entry:
                    new_entry["count"] = entry["count"]

                labels = entry.get("labels", [])
                if labels:
                    new_entry["labels"] = labels

                props = entry.get("properties", {})
                clean_props = {}
                for pname, pinfo in props.items():
                    cp = {}
                    if "indexed" in pinfo:
                        cp["indexed"] = pinfo["indexed"]
                    if "type" in pinfo:
                        cp["type"] = pinfo["type"]
                    if cp:
                        clean_props[pname] = cp
                if clean_props:
                    new_entry["properties"] = clean_props

                if entry.get("relationships"):
                    rels_out = {}
                    for rel_name, rel in entry["relationships"].items():
                        cr = {}
                        if "direction" in rel:
                            cr["direction"] = rel["direction"]
                        # nested labels
                        rlabels = rel.get("labels", [])
                        if rlabels:
                            cr["labels"] = rlabels
                        # nested properties
                        rprops = rel.get("properties", {})
                        clean_rprops = {}
                        for rpname, rpinfo in rprops.items():
                            crp = {}
                            if "indexed" in rpinfo:
                                crp["indexed"] = rpinfo["indexed"]
                            if "type" in rpinfo:
                                crp["type"] = rpinfo["type"]
                            if crp:
                                clean_rprops[rpname] = crp
                        if clean_rprops:
                            cr["properties"] = clean_rprops

                        if cr:
                            rels_out[rel_name] = cr

                    if rels_out:
                        new_entry["relationships"] = rels_out

                cleaned[key] = new_entry

            return cleaned

        try:
            results_json_str = await neo4j_driver.execute_query(
                get_schema_query,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            logger.debug(f"Read query returned {len(results_json_str)} rows")

            schema_clean = clean_schema(results_json_str[0].get("value"))

            schema_clean_str = json.dumps(schema_clean, default=str)

            return ToolResult(content=[TextContent(type="text", text=schema_clean_str)])

        except ClientError as e:
            if "Neo.ClientError.Procedure.ProcedureNotFound" in str(e):
                raise ToolError(
                    "Neo4j Client Error: This instance of Neo4j does not have the APOC plugin installed. Please install and enable the APOC plugin to use the `get_neo4j_schema` tool."
                )
            else:
                raise ToolError(f"Neo4j Client Error: {e}")

        except Neo4jError as e:
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error retrieving Neo4j database schema: {e}")
            raise ToolError(f"Unexpected Error: {e}")

    @mcp.tool(
        name=namespace_prefix + "read_neo4j_cypher",
        annotations=ToolAnnotations(
            title="Read Neo4j Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def read_neo4j_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: dict[str, Any] = Field(
            dict(), description="The parameters to pass to the Cypher query."
        ),
    ) -> list[ToolResult]:
        """Execute a read Cypher query on the neo4j database."""

        if _is_write_query(query):
            raise ValueError("Only MATCH queries are allowed for read-query")

        try:
            query_obj = Query(query, timeout=float(read_timeout))
            results = await neo4j_driver.execute_query(
                query_obj,
                parameters_=params,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )
            sanitized_results = [_value_sanitize(el) for el in results]
            results_json_str = json.dumps(sanitized_results, default=str)
            if token_limit:
                results_json_str = _truncate_string_to_tokens(
                    results_json_str, token_limit
                )

            logger.debug(f"Read query returned {len(results_json_str)} rows")

            return ToolResult(content=[TextContent(type="text", text=results_json_str)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error executing read query: {e}\n{query}\n{params}")
            raise ToolError(f"Neo4j Error: {e}\n{query}\n{params}")

        except Exception as e:
            logger.error(f"Error executing read query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")

    @mcp.tool(
        name=namespace_prefix + "write_neo4j_cypher",
        annotations=ToolAnnotations(
            title="Write Neo4j Cypher",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        enabled=allow_writes,
    )
    async def write_neo4j_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: dict[str, Any] = Field(
            dict(), description="The parameters to pass to the Cypher query."
        ),
    ) -> list[ToolResult]:
        """Execute a write Cypher query on the neo4j database."""

        if not _is_write_query(query):
            raise ValueError("Only write queries are allowed for write-query")

        try:
            _, summary, _ = await neo4j_driver.execute_query(
                query,
                parameters_=params,
                routing_control=RoutingControl.WRITE,
                database_=database,
            )

            counters_json_str = json.dumps(summary.counters.__dict__, default=str)

            logger.debug(f"Write query affected {counters_json_str}")

            return ToolResult(
                content=[TextContent(type="text", text=counters_json_str)]
            )

        except Neo4jError as e:
            logger.error(f"Neo4j Error executing write query: {e}\n{query}\n{params}")
            raise ToolError(f"Neo4j Error: {e}\n{query}\n{params}")

        except Exception as e:
            logger.error(f"Error executing write query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")


    @mcp.tool(
        name=namespace_prefix + "link_concept_chunks",
        annotations=ToolAnnotations(
            title="Link Concept to Similar Chunks",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        ),
        enabled=allow_writes,
    )
    async def link_concept_chunks(
            concept_id: str = Field(..., description="The unique concept identifier (or name)."),
            embedding: list[float] = Field(..., description="Embedding vector for the concept."),
            top_n: int = Field(5, description="Number of most similar chunks to link."),
    ) -> list[ToolResult]:
        """
        Finds top-N similar Chunk nodes by vector similarity and links them
        to the Concept node via (:Concept)-[:RELATED_TO_CHUNK]->(:Chunk).
        Requires a vector index `chunk_embeddings` on (:Chunk {embedding}).
        """

        # Step 1: Ensure vector index exists (ignore error if already created)
        ensure_index_query = """
        CREATE VECTOR INDEX chunk_embeddings
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine'
          }
        }
        """
        try:
            await neo4j_driver.execute_query(
                ensure_index_query,
                parameters_={"dim": len(embedding)},
                routing_control=RoutingControl.WRITE,
                database_=database,
            )
            logger.info("Neo4j Ensured vector index on Chunk(embedding).")
        except Exception as e:
            if "EquivalentSchemaRuleAlreadyExists" in str(e):
                logger.info("Neo4j Vector index already exists — skipping creation.")
            else:
                logger.warning(f"Neo4j Vector index check failed: {e}")

        # Step 2: Find top-N similar chunks
        query_find_similar = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_n, $embedding)
        YIELD node, score
        RETURN node.id AS chunk_id, node.text AS text, score
        ORDER BY score DESC
        """

        try:
            records, _, _ = await neo4j_driver.execute_query(
                query_find_similar,
                parameters_={"embedding": embedding, "top_n": top_n},
                routing_control=RoutingControl.READ,
                database_=database,
            )

            similar_chunks = [
                {"chunk_id": r["chunk_id"], "text": r.get("text", ""), "score": r["score"]}
                for r in records
            ]
            logger.info(f"Neo4j Found {len(similar_chunks)} similar chunks for concept {concept_id}")

        except Exception as e:
            logger.error(f"Neo4j Similarity query failed: {e}")
            raise ToolError(f"Neo4j Error finding similar chunks: {e}")

        # Step 3: Link concept to similar chunks
        link_query = """
        MATCH (c:Concept {name: $concept_id})
        UNWIND $chunks AS chunk
        MATCH (ch:Chunk {id: chunk.chunk_id})
        MERGE (c)-[r:RELATED_TO_CHUNK]->(ch)
        SET r.score = chunk.score
        RETURN count(r) AS links_created
        """

        try:
            _, summary, _ = await neo4j_driver.execute_query(
                link_query,
                parameters_={"concept_id": concept_id, "chunks": similar_chunks},
                routing_control=RoutingControl.WRITE,
                database_=database,
            )
            counters_json = json.dumps(summary.counters.__dict__, default=str)
            logger.info(f"Neo4j Linked {len(similar_chunks)} chunks to concept {concept_id}.")
            return [ToolResult(content=[TextContent(type="text", text=counters_json)])]

        except Exception as e:
            logger.error(f"Neo4j Failed to create concept-chunk links: {e}")
            raise ToolError(f"Neo4j Error creating links: {e}")


    @mcp.tool(
        name=namespace_prefix + "search_chunks_by_embedding",
        annotations=ToolAnnotations(
            title="Search Chunks by Embedding",
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def search_chunks_by_embedding(embedding: list[float], top_n: int = 5) -> list[ToolResult]:
        """
        Search text chunks by embedding similarity in Neo4j.
        Returns top-N chunks as JSON, in a format app-side can parse.
        """
        try:
            query = """
            CALL db.index.vector.queryNodes('chunk_embeddings', $top_n, $embedding)
            YIELD node, score
            RETURN node, score
            """
            results = await neo4j_driver.execute_query(
                query,
                parameters_={"embedding": embedding, "top_n": top_n},
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            chunks = []
            for r in results:
                node = r.get("node", {})
                props = node.get("properties", node)
                chunks.append({
                    "id": str(node.get("id", "")),
                    "text": props.get("text", ""),
                    "score": r.get("score", 0.0),
                })
            return chunks

        except Exception as e:
            logger.error(f"Neo4j Unexpected error in search_chunks_by_embedding: {e}", exc_info=True)
            raise ToolError(f"Neo4j Unexpected error: {e}")

    @mcp.tool(
        name=namespace_prefix + "get_concept_subgraph",
        annotations=ToolAnnotations(
            title="Get Concept Subgraph",
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_concept_subgraph(concept_name: str, depth: int = 2):
        """
        Retrieve a local subgraph around a given Concept.
        Only returns: id, labels, name, code_samples (no embeddings or large fields).
        """
        try:
            query = f"""
            MATCH path = (c:Concept {{name: $concept_name}})-[*1..{depth}]-(n)
            RETURN path
            """
            results, _, _ = await neo4j_driver.execute_query(
                query,
                parameters_={"concept_name": concept_name},
                routing_control=RoutingControl.READ,
                database_=database,
            )

            def make_json_safe(obj):
                """Recursively ensure JSON-safe serialization."""
                try:
                    json.dumps(obj)
                    return obj
                except TypeError:
                    if isinstance(obj, list):
                        return [make_json_safe(i) for i in obj]
                    if isinstance(obj, dict):
                        return {k: make_json_safe(v) for k, v in obj.items()}
                    if hasattr(obj, "__dict__"):
                        return make_json_safe(vars(obj))
                    return str(obj)

            nodes, relationships, node_ids = [], [], set()

            for record in results:
                path = record["path"]

                for node in path.nodes:
                    if node.element_id not in node_ids:
                        props = dict(node._properties)
                        # Extract only selected keys
                        filtered_props = {
                            k: v for k, v in props.items()
                            if k in {"name", "code_samples"}
                        }
                        safe_props = make_json_safe(filtered_props)

                        nodes.append({
                            "id": node.element_id,
                            "labels": list(node.labels),
                            "properties": safe_props,
                        })
                        node_ids.add(node.element_id)

                for rel in path.relationships:
                    relationships.append({
                        "id": rel.element_id,
                        "start": rel.start_node.element_id,
                        "end": rel.end_node.element_id,
                        "type": rel.type,
                    })

            payload = {"nodes": nodes, "relationships": relationships}
            return payload

        except Exception as e:
            logger.error(f"Neo4j get_concept_subgraph failed: {e}", exc_info=True)
            raise ToolError(f"Neo4j Unexpected error: {e}")

    return mcp


async def main(
    db_url: str,
    username: str,
    password: str,
    database: str,
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    allow_origins: list[str] = [],
    allowed_hosts: list[str] = [],
    read_timeout: int = 30,
    token_limit: Optional[int] = None,
    read_only: bool = False,
) -> None:
    logger.info("Starting MCP neo4j Server")

    neo4j_driver = AsyncGraphDatabase.driver(
        db_url,
        auth=(
            username,
            password,
        ),
    )
    custom_middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        ),
        Middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts),
    ]

    mcp = create_mcp_server(
        neo4j_driver, database, namespace, read_timeout, token_limit, read_only
    )

    # Run the server with the specified transport
    match transport:
        case "http":
            logger.info(
                f"Running Neo4j Cypher MCP Server with HTTP transport on {host}:{port}..."
            )
            await mcp.run_http_async(
                host=host, port=port, path=path, middleware=custom_middleware
            )
        case "stdio":
            logger.info("Running Neo4j Cypher MCP Server with stdio transport...")
            await mcp.run_stdio_async()
        case "sse":
            logger.info(
                f"Running Neo4j Cypher MCP Server with SSE transport on {host}:{port}..."
            )
            await mcp.run_http_async(
                host=host,
                port=port,
                path=path,
                middleware=custom_middleware,
                transport="sse",
            )
        case _:
            logger.error(
                f"Invalid transport: {transport} | Must be either 'stdio', 'sse', or 'http'"
            )
            raise ValueError(
                f"Invalid transport: {transport} | Must be either 'stdio', 'sse', or 'http'"
            )


if __name__ == "__main__":
    main()
