# src/graph_pipeline.py
import logging
import json
import os
import numpy as np
from typing import List, Dict
from langchain_mcp_adapters.client import MultiServerMCPClient
import re
from collections import defaultdict

from src.pipeline import run_query_async
from src.utils_embeddings import get_embedding_ollama

logger = logging.getLogger(__name__)

MCP_URL = os.getenv("NEO4J_MCP_URL", "http://neo4j_mcp:8080/api/mcp/")
TRANSPORT = "sse"

async def init_mcp_client():
    """Initialize MCP client and return instance."""
    logger.info(f"init_mcp_client: Connecting to MCP at {MCP_URL} via {TRANSPORT}")
    client = MultiServerMCPClient({
        "neo4j": {"url": MCP_URL, "transport": TRANSPORT}
    })
    await client.get_tools()
    logger.info("init_mcp_client: MCP connection initialized successfully.")
    return client


async def get_all_documents_async(limit: int = 50):
    """Fetch all Document nodes from Neo4j via MCP."""
    client = await init_mcp_client()

    cypher_query = f"""
    MATCH (d:Document)
    RETURN
      d.id AS id,
      coalesce(d.name, "") AS name,
      coalesce(d.author, "") AS author,
      coalesce(d.year, "") AS year,
      coalesce(d.uploaded_at, "") AS uploaded_at
    ORDER BY d.uploaded_at DESC
    LIMIT {limit}
    """

    try:
        async with client.session("neo4j") as session:
            result = await session.call_tool(
                "read_neo4j_cypher", {"query": cypher_query}
            )

        text_blocks = [c.text for c in getattr(result, "content", []) if hasattr(c, "text")]
        logger.info(f"get_all_documents_async: MCP returned {len(text_blocks)} text blocks")

        if not text_blocks:
            return []

        documents = []
        for block in text_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, list):
                    documents.extend(data)
                elif isinstance(data, dict):
                    documents.append(data)
                else:
                    logger.warning(f"get_all_documents_async: Unexpected block format: {type(data)}")
            except Exception:
                # Fallback: plain-text parsing
                for line in block.splitlines():
                    if line.strip():
                        documents.append({"name": line.strip()})

        logger.info(f"get_all_documents_async: Parsed {len(documents)} documents from MCP response")
        return documents

    except Exception as e:
        logger.error(f"get_all_documents_async: Failed to fetch documents via MCP: {e}", exc_info=True)
        return []


async def extract_triplets_locally(text: str):
    """
    Use LLM to:
      1. Separate code snippets from natural-language explanation.
      2. Extract semantic triplets only from the descriptive part.
      3. Return both triplets and code blocks for graph integration.
    """

    prompt = f"""
    You are a **semantic parser for technical documentation**.

    Your goal is to extract structured knowledge as **semantic triplets**
    and detect **code examples**.

    ---

    ### 🧩 TASKS
    1. If the text contains **code examples**, extract them under "code" (as valid JSON-safe strings).
    2. From the remaining text, extract conceptual definitions as **triplets**.

    ---

    ### 🧾 OUTPUT FORMAT
    Return STRICT JSON only:
    {{
      "triplets": [{{"subject": "...", "relation": "...", "object": "..."}}],
      "code": [{{"language": "python", "content": "escaped code string"}}]
    }}

    ---

    ### RULES
    - All line breaks in code must be escaped as `\\n` (not raw newlines).
    - Do not include markdown fences (```) or extra commentary around code.
    - **All triplet text — both "subject" and "object" — must be lowercase, except for acronyms**  
      (e.g., keep "API", "HTTP", "SQL" in uppercase if they appear).
    - Convert capitalized concept names like "Data Sampling" → `"data sampling"`.
    - Normalize concept names even if they appear in title case or sentence case.
    - The "relation" field remains in uppercase (e.g. `"IS"`, `"EXPLAINS"`).

    ---

    ### GUIDELINES
    - If the text defines or explains a concept, create a triplet:
      - `"subject"` = the concept being defined (in lowercase)
      - `"relation"` = one of ["IS", "IS_WHEN", "EXPLAINS", "REFERS_TO", "IS_CONCEPT"]
      - `"object"` = the short, natural-language definition (also lowercase)
    - If multiple concepts are defined, include multiple triplets.
    - Always use natural language for objects — never variable names or formulas.
    - Ignore procedural steps, function names, or code-specific logic.

    ---

    ### EXAMPLES

    **Example 1**
    Text:  
    "Explanation of Data Sampling: It is the process of selecting a representative sample."

    Output:
    {{
      "triplets": [
        {{
          "subject": "data sampling",
          "relation": "IS",
          "object": "the process of selecting a representative sample"
        }}
      ],
      "code": []
    }}

    ---

    **Example 2**
    Text:  
    "The HTTP protocol defines how messages are formatted and transmitted."

    Output:
    {{
      "triplets": [
        {{
          "subject": "http protocol",
          "relation": "EXPLAINS",
          "object": "how messages are formatted and transmitted"
        }}
      ],
      "code": []
    }}

    ---

    ### 🚀 IMPORTANT
    - Always output valid JSON (no extra text, comments, or markdown).
    - Every subject and object must be lowercase unless it’s a known acronym.

    Now parse this text:

    {text}
    """

    response = await run_query_async(prompt)
    raw = str(getattr(response, "content", response))
    logger.info(f"extract_triplets_locally: extract_triplets_locally raw LLM output: {raw[:300]}")

    # Clean markdown and extract JSON-like part
    cleaned = re.sub(r"```(json)?", "", raw).strip()
    json_start, json_end = cleaned.find("{"), cleaned.rfind("}")
    cleaned = cleaned[json_start:json_end + 1] if json_start != -1 else "{}"

    # Try parsing JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"extract_triplets_locally: Raw JSON parse failed ({e}). Attempting to repair formatting...")

        # Escape unescaped newlines in code blocks
        repaired = re.sub(r'("content"\s*:\s*")([^"]*?)"',
                          lambda m: m.group(1) + m.group(2).replace("\n", "\\n").replace("\r", "") + '"',
                          cleaned)
        try:
            data = json.loads(repaired)
        except Exception as e2:
            logger.error(f"extract_triplets_locally: JSON repair failed: {e2}\nRaw: {cleaned[:400]}")
            return {"triplets": [], "code": []}

    # Extract results
    triplets = data.get("triplets", [])
    code_blocks = data.get("code", [])
    logger.info(f"extract_triplets_locally: Extracted {len(triplets)} triplets and {len(code_blocks)} code blocks.")
    return {"triplets": triplets, "code": code_blocks}


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


async def insert_triplets(text: str, link_chunks: bool = True, top_n: int = 5):
    """
    1 Extract semantic triplets and code snippets from text (via LLM).
    2 Merge Concept nodes and relations into Neo4j.
    3 Attach each code snippet to its most semantically similar concept.
    4 Optionally link each new Concept to top-N similar Chunk nodes
        using the MCP vector search tool link_concept_chunks.
    """

    # STEP 0 — Init MCP Client

    try:
        client = await init_mcp_client()
    except Exception as e:
        logger.error(f"insert_triplets: MCP client initialization failed: {e}", exc_info=True)
        return {"error": f"MCP init failed: {e}"}

    # STEP 1 — Extract triplets and code via LLM

    try:
        data = await extract_triplets_locally(text)
        triplets = data.get("triplets", [])
        code_blocks = data.get("code", [])
        logger.info(f"insert_triplets: Extracted {len(triplets)} triplets and {len(code_blocks)} code blocks.")
    except Exception as e:
        logger.error(f"insert_triplets: Triplet extraction failed: {e}", exc_info=True)
        return {"error": f"Triplet extraction failed: {e}"}

    if not triplets:
        logger.warning("insert_triplets: No triplets extracted from text.")
        return {"triplets": []}

    created = []

    # STEP 2 — Insert triplets into Neo4j and embed concepts

    concept_embeddings: Dict[str, List[float]] = {}
    for idx, t in enumerate(triplets, 1):
        try:
            subj = t.get("subject") if isinstance(t, dict) else None
            rel = (t.get("relation", "RELATED_TO") if isinstance(t, dict) else "RELATED_TO")
            obj = t.get("object") if isinstance(t, dict) else None

            if not subj or (obj is None and rel != "IS_CONCEPT"):
                logger.warning(f"insert_triplets: Skipping malformed triplet #{idx}: {t}")
                continue

            subj_emb = get_embedding_ollama(subj)
            obj_emb = get_embedding_ollama(obj) if obj else None
            concept_embeddings[subj] = subj_emb
            if obj:
                concept_embeddings[obj] = obj_emb

            cypher = f"""
            MERGE (a:Concept {{name: $subj}})
            ON CREATE SET a.embedding = $subj_emb
            """

            if obj:
                cypher += f"""
                MERGE (b:Concept {{name: $obj}})
                ON CREATE SET b.embedding = $obj_emb
                MERGE (a)-[:{rel.upper().replace(' ', '_')}]->(b)
                """

            cypher += " RETURN a"

            params = {"subj": subj, "obj": obj, "subj_emb": subj_emb, "obj_emb": obj_emb}

            async with client.session("neo4j") as session:
                await session.call_tool("write_neo4j_cypher", {"query": cypher, "params": params})

            created.append({"subject": subj, "relation": rel, "object": obj})
            logger.info(f"insert_triplets: Inserted #{idx}: ({subj})-[:{rel}]->({obj})")

        except Exception as e:
            logger.error(f"insert_triplets: Error processing triplet #{idx} ({t}): {e}", exc_info=True)

    # STEP 3 — Link code snippets to most similar concept

    if code_blocks and concept_embeddings:
        for code_entry in code_blocks:
            code_text = code_entry.get("content", "")
            if not code_text.strip():
                continue

            try:
                code_emb = get_embedding_ollama(code_text)
                # Find the most similar concept
                best_concept = max(
                    concept_embeddings.keys(),
                    key=lambda name: cosine_similarity(concept_embeddings[name], code_emb)
                )
                best_score = cosine_similarity(concept_embeddings[best_concept], code_emb)
                logger.info(f"insert_triplets: Best concept for code: '{best_concept}' (similarity={best_score:.3f})")

                cypher_code = """
                MATCH (c:Concept {name: $concept})
                SET c.code_samples = coalesce(c.code_samples, []) + $code_entry
                RETURN c.name
                """
                params = {"concept": best_concept, "code_entry": json.dumps(code_entry, ensure_ascii=False)}

                async with client.session("neo4j") as session:
                    await session.call_tool("write_neo4j_cypher", {"query": cypher_code, "params": params})

                logger.info(f"insert_triplets: Linked code snippet to concept '{best_concept}'")

            except Exception as e:
                logger.error(f"insert_triplets: Failed to link code snippet: {e}", exc_info=True)

    # STEP 4 — Link concepts to top-N chunks

    if link_chunks:
        for concept_name, concept_emb in concept_embeddings.items():
            try:
                async with client.session("neo4j") as session:
                    link_result = await session.call_tool(
                        "link_concept_chunks",
                        {
                            "concept_id": concept_name,
                            "embedding": concept_emb,
                            "top_n": top_n,
                        },
                    )
                logger.info(f"insert_triplets: Linked concept '{concept_name}' to chunks (top_n={top_n}) → {link_result}")
            except Exception as e:
                logger.error(f"insert_triplets: Failed to link concept '{concept_name}' to chunks: {e}")

    logger.info(f"insert_triplets: Inserted {len(created)} triplets total and linked {len(code_blocks)} code snippets.")
    return {"inserted": created, "linked_code": len(code_blocks)}


def list_uploaded_documents(limit: int = 50):
    """Synchronous wrapper for Streamlit UI."""
    import asyncio
    return asyncio.run(get_all_documents_async(limit))


async def link_concept_to_chunks(concept_name: str, concept_text: str, top_n: int = 3):
    """
    Create a Concept node and connect it to the top-N similar Chunks
    based on vector similarity using the chunk_embeddings index.
    """

    client = await init_mcp_client()
    embedding = get_embedding_ollama(concept_text)
    dim = len(embedding)

    logger.info(f"link_concept_to_chunks: Linking concept '{concept_name}' to top {top_n} chunks (dim={dim})")

    # 1 Ensure Concept node exists
    cypher_create_concept = """
    MERGE (c:Concept {name: $name})
    ON CREATE SET c.description = $text, c.embedding = $embedding
    RETURN c
    """
    async with client.session("neo4j") as session:
        await session.call_tool(
            "write_neo4j_cypher",
            {"query": cypher_create_concept, "params": {
                "name": concept_name,
                "text": concept_text,
                "embedding": embedding
            }}
        )

    # 2 Retrieve top-N similar chunks
    cypher_query_chunks = """
    CALL db.index.vector.queryNodes('chunk_embeddings', $top_n, $embedding)
    YIELD node, score
    RETURN node.id AS id, node.text AS text, score
    """
    async with client.session("neo4j") as session:
        result = await session.call_tool(
            "neo4j.read_neo4j_cypher",
            {"query": cypher_query_chunks, "params": {"embedding": embedding, "top_n": top_n}}
        )

    text_blocks = [c.text for c in getattr(result, "content", []) if hasattr(c, "text")]
    logger.info(f"link_concept_to_chunks: Found {len(text_blocks)} similar chunks for concept '{concept_name}'")

    # 3 Create relationships
    cypher_link = """
    MATCH (concept:Concept {name: $name})
    UNWIND $chunks AS chunk
    MATCH (c:Chunk {id: chunk.id})
    MERGE (concept)-[:RELATED_TO {score: chunk.score}]->(c)
    """
    chunk_data = []
    for block in text_blocks:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                chunk_data.extend(parsed)
            elif isinstance(parsed, dict):
                chunk_data.append(parsed)
        except Exception:
            continue

    if chunk_data:
        async with client.session("neo4j") as session:
            await session.call_tool(
                "write_neo4j_cypher",
                {"query": cypher_link, "params": {"name": concept_name, "chunks": chunk_data}}
            )
        logger.info(f"link_concept_to_chunks: Linked concept '{concept_name}' to {len(chunk_data)} chunks.")
    else:
        logger.warning(f"link_concept_to_chunks: No chunks found to link with concept '{concept_name}'.")

    return {"concept": concept_name, "linked_chunks": len(chunk_data)}


async def extract_concepts_from_text(text: str) -> list[str]:
    """
    Extracts conceptual terms or entities from a question or paragraph.
    """

    try:

        prompt = f"""
        You are a **concept extraction assistant**.

        Your task is to identify **key technical concepts or named entities**
        from the following sentence or paragraph.

        ---

        ### OUTPUT REQUIREMENT
        Return a **pure JSON list of lowercase strings only** — no explanations, no prose, no markdown.

        ---

        ### RULES
        - Always convert all extracted concept names to **lowercase**, even if they appear capitalized in the text.
        - Preserve acronyms (e.g. "API", "HTTP", "SQL") in their **original uppercase form**.
        - Remove punctuation, articles ("the", "a", "an"), and extra words not part of the core concept.
        - Never invent concepts that aren’t explicitly or implicitly mentioned.
        - Output must be **valid JSON**, e.g. `["concept one", "concept two"]`.

        ---

        ### GOOD EXAMPLES
        Input: "Remind me about Design Patterns"
        Output: ["design patterns"]

        Input: "Explain how the Observer Pattern supports Data Consistency."
        Output: ["observer pattern", "data consistency"]

        Input: "Please tell what is Load Balancing"
        Output: ["load balancing"]

        Input: "What does the HTTP protocol define?"
        Output: ["http protocol"]

        ---

        ### BAD EXAMPLES
        Incorrect: ["Design Patterns", "Observer Pattern"]  ← Capitalized
        Incorrect: ["Design Patterns.", "Load balancing concept"]  ← Punctuation & extra words
        Incorrect: ["API", "data streaming", "NewTerm"]  ← Invented "NewTerm"

        ---

        Now extract all key concepts from the text below, returning them strictly as **lowercase JSON**:

        {text}
        """

        # 1 Ask the model
        # response = await model.ainvoke(prompt)
        response = await run_query_async(prompt)
        raw = str(getattr(response, "content", response)).strip()
        logger.info(f"extract_concepts_from_text: Raw LLM concept response: {raw}")
        logger.debug(f"extract_concepts_from_text: Raw LLM concept response: {raw}")

        # 2 Clean Markdown wrappers
        cleaned = re.sub(r"```(json|python)?", "", raw, flags=re.IGNORECASE).strip()
        logger.info(f"extract_concepts_from_text: Raw LLM concept cleaned: {cleaned}")

        # 3 Try JSON parse first
        start, end = cleaned.find("["), cleaned.rfind("]")
        json_text = cleaned[start:end + 1] if start != -1 and end != -1 else None

        concepts = []
        if json_text:
            try:
                concepts = json.loads(json_text)
                if not isinstance(concepts, list):
                    raise ValueError("Not a list.")
            except Exception:
                concepts = []

        # 4 Fallback: if still empty, ask a mini-prompt
        if not concepts:
            logger.warning("extract_concepts_from_text: Primary extraction failed — retrying with simplified prompt.")
            simple_prompt = f"List the main technical term(s) from this text as JSON array only:\n{text}"
            # retry = await model.ainvoke(simple_prompt)
            retry = await run_query_async(simple_prompt)
            logger.info(f"extract_concepts_from_text: Raw LLM concept retry: {retry}")
            retry_text = re.sub(r"```(json)?", "", str(retry)).strip()
            s, e = retry_text.find("["), retry_text.rfind("]")
            if s != -1 and e != -1:
                try:
                    concepts = json.loads(retry_text[s:e + 1])
                except Exception:
                    pass

        # 5 Normalize + deduplicate
        normalized = list({c.strip() for c in concepts if isinstance(c, str) and c.strip()})
        logger.info(f"extract_concepts_from_text: Extracted {len(normalized)} key concepts: {normalized}")
        return normalized

    except Exception as e:
        logger.error(f"extract_concepts_from_text: Concept extraction failed: {e}", exc_info=True)
        return []



def unwrap_text(result) -> str:
    """Unwrap text from ToolResult."""
    if not result:
        return ""
    # Direct ToolResult or list of TextContent
    if hasattr(result, "content") and result.content:
        first = result.content[0]
        if hasattr(first, "text") and first.text:
            return first.text.strip()
        if hasattr(first, "content") and first.content:
            inner = first.content[0]
            if hasattr(inner, "text") and inner.text:
                return inner.text.strip()
    # Raw string fallback
    if isinstance(result, str):
        return result.strip()
    return ""


def filter_connected_nodes(graph: dict, root_id: str) -> dict:
    """Keep only nodes and edges reachable from the root node."""
    visited = set()
    stack = [root_id]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            for rel in graph["edges"].get(current, []):
                stack.append(rel["target"])

    filtered_nodes = {
        nid: graph["node_index"][nid]
        for nid in visited if nid in graph["node_index"]
    }
    filtered_edges = {
        nid: [r for r in rels if r["target"] in visited]
        for nid, rels in graph["edges"].items()
        if nid in visited
    }
    return {"node_index": filtered_nodes, "edges": filtered_edges}


async def extract_text_from_subgraphs(concepts: list[str], depth: int = 2) -> dict:
    """
    Retrieves connected concept subgraphs (excluding standalone or chunk-only nodes).
    Returns: {'concept_text': str, 'chunks': list[str]}.
    """
    if not concepts:
        logger.warning("extract_text_from_subgraphs: No concepts provided for subgraph extraction.")
        return {"concept_text": "", "chunks": []}

    client = await init_mcp_client()
    all_concept_texts, all_chunk_texts = [], []

    # Local helper: DFS traversal for connected concepts
    def dfs(node_id: str, graph: dict, visited: set, path_text: str) -> str:
        visited.add(node_id)
        node = graph["node_index"].get(node_id)
        if not node:
            return path_text

        props = node.get("properties", {})
        name = props.get("name", "")
        local_text = path_text

        # Root concept phrasing
        if not path_text:
            local_text += f"{name}"
        else:
            local_text += f" {name}"

        # Traverse relationships
        for rel in graph["edges"].get(node_id, []):
            rel_type = rel["type"].replace("_", " ").lower()
            target_id = rel["target"]
            if target_id not in visited:
                target_node = graph["node_index"].get(target_id)
                if target_node:
                    target_name = target_node.get("properties", {}).get("name", "")
                    if target_name:
                        local_text += f" {rel_type} {target_name}."
                local_text = dfs(target_id, graph, visited, local_text)

        # Append code samples after description
        if "code_samples" in props and isinstance(props["code_samples"], list):
            for sample in props["code_samples"]:
                if isinstance(sample, str):
                    try:
                        sample = json.loads(sample)
                    except Exception:
                        continue
                if isinstance(sample, dict):
                    lang = sample.get("language", "python")
                    content = sample.get("content", "").strip()
                    if content:
                        local_text += f"\n\nCode example ({lang}):\n```{lang}\n{content}\n```\n"

        return local_text

    # Process all input concepts
    for concept in concepts:
        try:
            logger.info(f"extract_text_from_subgraphs: Retrieving subgraph for concept: {concept}")
            async with client.session("neo4j") as session:
                result = await session.call_tool(
                    "get_concept_subgraph",
                    {"concept_name": concept, "depth": depth},
                )

            raw_text = unwrap_text(result)
            if not raw_text:
                logger.warning(f"extract_text_from_subgraphs: Empty response for '{concept}'.")
                continue

            try:
                graph_json = json.loads(raw_text)
            except Exception as e:
                logger.error(f"extract_text_from_subgraphs: Invalid JSON for '{concept}': {e}\nRaw: {raw_text[:200]}")
                continue

            nodes = graph_json.get("nodes", [])
            rels = graph_json.get("relationships", [])
            if not nodes:
                logger.warning(f"extract_text_from_subgraphs: No nodes found for '{concept}'.")
                continue

            # Separate concept vs. chunk nodes
            concept_nodes, chunk_nodes = [], []
            for n in nodes:
                labels = n.get("labels", [])
                if any("Chunk" in lbl for lbl in labels):
                    chunk_nodes.append(n)
                else:
                    concept_nodes.append(n)

            logger.info(f"extract_text_from_subgraphs: extract_text_from_subgraphs concept_nodes: {concept_nodes}")

            # Build concept-only graph (preserve name + code_samples)
            node_index = {}

            for n in concept_nodes:
                props = n.get("properties", {}) or {}
                clean_props = {}

                # Preserve name
                if "name" in props:
                    clean_props["name"] = props["name"]

                # Normalize and preserve code_samples
                if "code_samples" in props:
                    raw_samples = props["code_samples"]
                    parsed_samples = []

                    if isinstance(raw_samples, list):
                        for sample in raw_samples:
                            if isinstance(sample, str):
                                try:
                                    sample = json.loads(sample)
                                except Exception:
                                    continue
                            if isinstance(sample, dict):
                                parsed_samples.append(sample)
                    clean_props["code_samples"] = parsed_samples

                # Keep node entry clean and consistent
                node_index[n["id"]] = {
                    "id": n["id"],
                    "labels": n.get("labels", []),
                    "properties": clean_props,
                }

            # Build edges between concepts
            edges = defaultdict(list)
            for r in rels:
                if r["start"] in node_index and r["end"] in node_index:
                    edges[r["start"]].append({"target": r["end"], "type": r["type"]})
                    edges[r["end"]].append({"target": r["start"], "type": r["type"]})

            graph = {"node_index": node_index, "edges": edges}
            logger.info(f"extract_text_from_subgraphs: extract_text_from_subgraphs graph: {graph}")

            # Find root node
            root_id = next(
                (n["id"] for n in concept_nodes if n["properties"].get("name") == concept),
                None,
            )
            if not root_id:
                logger.warning(f"extract_text_from_subgraphs: Root node '{concept}' not found in subgraph.")
                continue

            # Keep only connected concepts
            graph = filter_connected_nodes(graph, root_id)

            # Build text via DFS
            visited = set()
            logger.info(f"extract_text_from_subgraphs: extract_text_from_subgraphs graph: {graph}")
            concept_text = dfs(root_id, graph, visited, "")
            all_concept_texts.append(concept_text.strip())

            # Extract chunk texts (if any)
            for chunk in chunk_nodes:
                props = chunk.get("properties", {})
                chunk_text = props.get("text") or props.get("name", "")
                if chunk_text:
                    all_chunk_texts.append(chunk_text.strip())

        except Exception as e:
            logger.error(f"extract_text_from_subgraphs: Subgraph extraction failed for '{concept}': {e}", exc_info=True)
            continue

    combined_concept_text = "\n\n".join(all_concept_texts)
    unique_chunks = list({t for t in all_chunk_texts if t})
    logger.info(
        f"extract_text_from_subgraphs: Extracted {len(all_concept_texts)} concept blocks and {len(unique_chunks)} unique chunks."
    )

    return {"concept_text": combined_concept_text, "chunks": unique_chunks}


async def extract_text_from_chunks(question: str, top_n: int = 3) -> str:
    """
    Retrieves the text of the top-N most relevant chunks for a user question.
    Uses question embedding → vector search → text aggregation.
    """
    logger.info("extract_text_from_chunks: extract_text_from_chunks")
    try:
        client = await init_mcp_client()
        embedding = get_embedding_ollama(question)

        async with client.session("neo4j") as session:
            result = await session.call_tool(
                "search_chunks_by_embedding",
                {"embedding": embedding, "top_n": top_n},
            )

        # Unwrap JSON text from MCP
        raw_text = ""
        if hasattr(result, "content") and result.content:
            first = result.content[0]
            if hasattr(first, "text") and first.text:
                raw_text = first.text.strip()

        if not raw_text:
            logger.warning("extract_text_from_chunks: Empty chunk search response.")
            return ""

        logger.info(f"extract_text_from_chunks: extract_text_from_chunks raw: {raw_text[:200]}")

        try:
            chunks = json.loads(raw_text)
            if not isinstance(chunks, list):
                raise ValueError("Expected a list of chunk dicts.")
        except Exception as e:
            logger.error(f"extract_text_from_chunks: Invalid JSON from chunk search: {e}\nRaw: {raw_text[:200]}")
            return ""

        if not chunks:
            logger.warning("extract_text_from_chunks: No chunks retrieved.")
            return ""

        # Aggregate text content
        text_context = "\n".join(c.get("text", "") for c in chunks if c.get("text"))
        logger.info(f"extract_text_from_chunks: Retrieved {len(chunks)} closest chunks.")
        return text_context.strip()

    except Exception as e:
        logger.error(f"extract_text_from_chunks: Chunk extraction failed: {e}", exc_info=True)
        return ""


async def build_rag_context(question: str) -> str:
    """
    Combines subgraph (symbolic) and chunk (semantic) context into one text block.
    """
    logger.info(f"build_rag_context: Building RAG context for: {question}")

    concepts = await extract_concepts_from_text(question)
    logger.info(f"build_rag_context: Extracted key concepts: {concepts}")

    if not concepts:
        logger.warning("build_rag_context: No key concepts found — fallback to chunk-only context.")
        return await extract_text_from_chunks(question)

    # Retrieve subgraph context (concepts + chunks)
    subgraph_result = await extract_text_from_subgraphs(concepts)
    concept_text = subgraph_result["concept_text"]
    subgraph_chunks = subgraph_result["chunks"]

    # Retrieve vector-similar chunks
    chunk_text = await extract_text_from_chunks(question)
    chunk_list = chunk_text.split("\n\n") if chunk_text else []

    # Merge and deduplicate all chunks
    all_chunks = list({c.strip() for c in subgraph_chunks + chunk_list if c.strip()})
    merged_chunk_text = "\n\n".join(all_chunks)

    combined = (
        f"--- GRAPH CONCEPTS ---\n{concept_text}\n\n"
        f"--- RELEVANT CHUNKS ---\n{merged_chunk_text}"
    )

    logger.info("build_rag_context: RAG context built successfully.")
    return combined.strip()
