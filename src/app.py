import os
import re
import json
import uuid
import asyncio
import pandas as pd

import streamlit as st
import logging
from pathlib import Path
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from src.pdf_ingest import ingest_pdf
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from src.graph_pipeline import list_uploaded_documents, build_rag_context
from src.pipeline import run_query, run_query_async

load_dotenv()

# -------------------- Toast Utility --------------------
def notify(message: str, icon: str = "✅"):
    """Display a transient toast message."""
    st.toast(f"{icon} {message}")


# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Config --------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MCP_URL = os.getenv("NEO4J_MCP_URL", "http://neo4j_mcp:8080/api/mcp/")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")

# -------------------- MCP Agent Init --------------------
_client_agent_cache = None


def init_mcp_agent():
    """Lazy init — only create once per process."""
    global _client_agent_cache
    if _client_agent_cache is not None:
        return _client_agent_cache

    async def _async_init():
        try:
            transport = "sse"
            logger.info(f"init_mcp_agent: Using MCP transport: {transport}")
            client = MultiServerMCPClient({"neo4j": {"url": MCP_URL, "transport": transport}})
            tools = await asyncio.wait_for(client.get_tools(), timeout=10)
            logger.info(f"init_mcp_agent: Loaded {len(tools)} tools from MCP")
            model = ChatOllama(
                model=CHAT_MODEL,
                base_url=OLLAMA_HOST,
            )
            agent = create_react_agent(model, tools)
            return client, agent
        except Exception as e:
            # Log and re-raise as RuntimeError so caller can handle gracefully
            logger.exception("init_mcp_agent: MCP initialization failed:")
            raise RuntimeError(f"init_mcp_agent: MCP init failed: {e}")

    _client_agent_cache = asyncio.run(_async_init())
    return _client_agent_cache

client = agent = None

try:
    if "mcp_initialized" not in st.session_state:
        with st.spinner("🔌 Connecting to MCP server..."):
            try:
                client, agent = init_mcp_agent()
                st.session_state["mcp_initialized"] = True
                st.session_state["client"] = client
                st.session_state["agent"] = agent
                notify("Connected to MCP server", "🧩")
            except Exception as e:
                # Graceful fallback — app still imports
                st.warning(f"init_mcp_agent: MCP not initialized: {e}")
                logger.warning(f"init_mcp_agent: MCP init skipped or failed: {e}")
                st.session_state["mcp_initialized"] = False
    else:
        client = st.session_state.get("client")
        agent = st.session_state.get("agent")
except Exception as e:
    logger.warning(f"init_mcp_agent: Skipping MCP init (non-Streamlit context): {e}")
    if hasattr(st, "session_state"):
        st.session_state["mcp_initialized"] = False

# -------------------- Streamlit Layout --------------------
st.set_page_config(page_title="SlowReader", layout="wide")
st.title("📚 SlowReader — Graph-Augmented Reading Assistant")

# --- Persistent tab selection across reruns ---
TAB_LABELS = ["📥 Upload Book", "🧠 Build Graph", "💬 Chat", "📚 Library"]

# Initialize active tab once
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = TAB_LABELS[0]

# Draw all tabs
tabs = st.tabs(TAB_LABELS)

# Helper to preserve the active tab after reruns
def remember_tab(label: str):
    st.session_state["active_tab"] = label


# Apply tab header + memory
for i, label in enumerate(TAB_LABELS):
    with tabs[i]:
        if st.session_state["active_tab"] == label:
            st.markdown(f"### {label}")
        if st.button(f"Go to {label}", key=f"goto_{i}"):
            remember_tab(label)


def clean_context(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = [s for s in sentences if not (s.strip() in seen or seen.add(s.strip()))]
    return " ".join(unique)


# ------------------------------------------------------------------
# TAB 1: PDF Upload & Ingestion
# ------------------------------------------------------------------
with tabs[0]:
    remember_tab(TAB_LABELS[0])
    st.header("Upload a PDF Book")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        file_path = UPLOAD_DIR / uploaded_file.name
        marker_path = file_path.with_suffix(".ingested")

        # Keep track of current file in session state
        st.session_state["current_file"] = str(file_path)

        # Save uploaded file only once
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"✅ Saved: {uploaded_file.name}")
            logger.info(f"PDF Upload Tab: Saved uploaded file: {file_path}")

        # If not yet ingested
        if not marker_path.exists():
            st.info("Please describe the book (name, author, year).")

            # Persist the text box content across reruns
            if "book_meta" not in st.session_state:
                st.session_state["book_meta"] = ""

            st.session_state["book_meta"] = st.text_area(
                "Book description",
                value=st.session_state["book_meta"],
                key="meta_input",
            )

            if st.session_state["book_meta"]:
                user_meta = st.session_state["book_meta"]

                # --- Run metadata extraction
                with st.spinner("🔍 Extracting metadata via Ollama..."):

                    prompt = f"""
                    You are an information extraction model.
                    Extract the book's metadata from this text and return **ONLY** valid JSON with keys:
                    name, author, year.

                    Example:
                    {{
                      "name": "Book Title",
                      "author": "Author Name",
                      "year": 2024
                    }}

                    Text:
                    {user_meta}
                    """

                    logger.info("PDF Upload Tab: [LLM INPUT PROMPT] → %s", prompt)
                    response = run_query(prompt)
                    logger.info("PDF Upload Tab: [LLM RAW OUTPUT] ← %s", response)

                try:
                    # cleaned = re.sub(r"```(json)?", "", response).strip()
                    # cleaned = cleaned.split("}", 1)[0] + "}" if "}" in cleaned else cleaned
                    cleaned = re.sub(r"```(json)?", "", response)
                    cleaned = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", cleaned)  # remove invisible chars
                    cleaned = cleaned.strip()
                    json_start = cleaned.find("{")
                    json_end = cleaned.rfind("}")
                    if json_start == -1 or json_end == -1:
                        logger.warning(f"⚠️ No JSON braces found in LLM output. Cleaned={repr(cleaned[:200])}")
                        raise ValueError("No valid JSON object found in model output")
                    json_text = cleaned[json_start:json_end + 1]
                    logger.info(f"RAW repr: {repr(json_text)}" )
                    meta = json.loads(json_text)
                    st.json(meta)
                    logger.info("PDF Upload Tab: [PARSED META] %s", meta)

                    doc_id = str(uuid.uuid4())
                    with st.spinner("⚙️ Ingesting PDF..."):
                        result = ingest_pdf(str(file_path), doc_id, meta)
                        st.success(f"📚 Ingested {result['chunks']} chunks for {result['doc_id']}")
                        marker_path.touch()
                        st.session_state.pop("book_meta", None)
                        st.session_state["last_uploaded_book"] = meta["name"]
                        st.toast(f"📘 Book '{meta['name']}' successfully ingested!")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    logger.exception("PDF Upload Tab: Metadata or ingestion failed")

        else:
            st.info(f"ℹ️ Already ingested: {uploaded_file.name}")

# ------------------------------------------------------------------
# TAB 2: Concept Graph Builder (Triplet Extraction + Linking)
# ------------------------------------------------------------------
with tabs[1]:
    remember_tab(TAB_LABELS[1])
    st.header("Build Knowledge Graph")

    user_text = st.text_area("✏️ Paste notes or text to add to graph", height=200)

    if st.button("🔗 Build Graph", key="build_graph_btn"):
        st.session_state["active_tab"] = TAB_LABELS[1]
        if not user_text.strip():
            st.warning("⚠️ Please enter some text before building the graph.")
        else:
            with st.spinner("🧠 Extracting concepts and updating Neo4j graph..."):
                try:
                    from src.graph_pipeline import insert_triplets

                    result = asyncio.run(insert_triplets(user_text, link_chunks=True, top_n=5))
                    st.info("ℹ️ insert_triplets result: {result}")

                    if "error" in result:
                        st.error(f"❌ Failed: {result['error']}")
                    elif "inserted" in result and len(result["inserted"]) > 0:
                        st.success(f"✅ Added {len(result['inserted'])} new concepts and relations.")
                        st.json(result["inserted"])
                        notify("🧩 New concepts linked to chunks", "🌐")
                    else:
                        st.info("ℹ️ No valid triplets were extracted from the text.")

                except Exception as e:
                    st.error(f"❌ Graph build failed: {e}")
                    import traceback
                    st.text(traceback.format_exc())

# ------------------------------------------------------------------
# TAB 3: Chat with Ollama
# ------------------------------------------------------------------
with tabs[2]:
    remember_tab(TAB_LABELS[2])
    st.header("Chat with Ollama")

    chat_input = st.text_area("💬 Ask your question:", height=150, key="chat_input_box")

    col1, col2 = st.columns(2)

    # ---------------------------------------------------------------
    # Option 1 — Direct LLM answer (no retrieval)
    # ---------------------------------------------------------------
    if col1.button("Ask (LLM only)", key="chat_llm_btn"):
        st.session_state["active_tab"] = TAB_LABELS[2]
        if not chat_input.strip():
            st.warning("⚠️ Please enter a question first.")
        else:
            with st.spinner("🤖 Thinking (LLM only)..."):
                try:
                    # model = ChatOllama(
                    #     model=CHAT_MODEL,
                    #     base_url=OLLAMA_HOST,
                    # )
                    # response = asyncio.run(model.ainvoke(chat_input))
                    response = asyncio.run(run_query_async(chat_input))
                    answer = getattr(response, "content", str(response))
                    st.markdown(answer)
                    notify("Response received from Ollama", "🤖")
                except Exception as e:
                    st.error(f"❌ Direct LLM query failed: {e}")
                    import traceback
                    st.text(traceback.format_exc())

    # ---------------------------------------------------------------
    # Option 2 — Ask with Hybrid RAG
    # ---------------------------------------------------------------
    if col2.button("Ask with RAG 🧠", key="chat_rag_btn"):
        st.session_state["active_tab"] = TAB_LABELS[2]
        if not chat_input.strip():
            st.warning("⚠️ Please enter a question first.")
        else:
            with st.spinner("🧠 Searching graph and chunks..."):
                try:
                    # Step 1 — Build hybrid RAG context
                    rag_context = asyncio.run(build_rag_context(chat_input))
                    rag_context = rag_context.replace("\\n", "\n")
                    rag_context = clean_context(rag_context)

                    # Step 2 — Compose context-aware prompt for LLM
                    prompt = f"""
                    You are a precise and technically competent AI assistant writing for engineers.

                    Your task is to read and consolidate the provided context retrieved from a knowledge graph.
                    The retrieved text may contain duplicate or fragmented sentences. 
                    You must merge, clean, and rewrite it into a concise, grammatically correct, and logically coherent explanation, 
                    while staying as close as possible to the factual meaning of the context.

                    Rules:
                    1. Keep all factual and technical details from the context, but **remove repeated or redundant sentences**.
                    2. Rewrite fragmented or repetitive statements into a **single, well-formed technical paragraph**.
                    3. Ensure the result reads like a **documentation or textbook-style explanation** that an engineer can quickly understand.
                    4. Do **not invent** or add facts that are not supported by the context.
                    5. If the context includes **code examples** (fenced with triple backticks ```), 
                       preserve them **exactly** as they appear — including language tags, indentation, and line breaks.
                       Replace any literal '\n' sequences with actual newlines before output.
                       Do not add or remove code comments, variable names, or function definitions.
                    6. When relevant, you may refer to the code or describe what it does in your answer, 
                       but never modify the code itself, **include it verbatim** within your final response.
                    7. If there is not enough information to fully answer the question, say:
                       "The provided context does not contain enough information to answer this question."

                    🧩 Context:
                    --- CONTEXT START ---
                    {rag_context}
                    --- CONTEXT END ---

                    Question:
                    {chat_input}

                    Your Answer:
                    """

                    # Step 3 — Call Ollama for contextual reasoning
                    # model = ChatOllama(
                    #     model=CHAT_MODEL,
                    #     base_url=OLLAMA_HOST,
                    # )
                    logger.info(f"RAG Tab: RAG prompt= {prompt}")
                    # response = asyncio.run(model.ainvoke(prompt))
                    response = asyncio.run(run_query_async(prompt))
                    answer = getattr(response, "content", str(response))
                    logger.info(f"RAG Tab: RAG answer= {answer}")

                    # Step 4️⃣ — Display result
                    st.markdown("### 🧠 Contextual Answer")
                    st.markdown(answer)
                    st.markdown("---")
                    st.markdown("### 📚 Context Used")
                    st.text_area("RAG Context", rag_context, height=250)
                    notify("Answer generated with RAG context", "🧠")

                except Exception as e:
                    st.error(f"❌ RAG pipeline failed: {e}")
                    import traceback
                    st.text(traceback.format_exc())

# ------------------------------------------------------------------
# TAB 4: List Uploaded Books
# ------------------------------------------------------------------
with tabs[3]:
    remember_tab(TAB_LABELS[3])
    st.header("📚 Library")
    try:
        docs = list_uploaded_documents()
    except Exception as e:
        logger.error(f"Library Tab: Error fetching library: {e}")
        st.error("Failed to connect to Neo4j MCP server.")
        docs = []
    if not docs:
        st.info("The library is empty — please upload your first PDF book!")
    else:
        df = pd.DataFrame(docs)
        # Ensure all expected columns exist
        for col in ["name", "author", "year", "uploaded_at"]:
            if col not in df.columns:
                df[col] = ""
        if df.empty:
            st.info("The library is empty — please upload your first PDF book!")
        else:
            st.dataframe(
                df[["name", "author", "year", "uploaded_at"]],
                width="stretch",
                hide_index=True,
            )
        notify(f"Loaded {len(docs)} books from Neo4j", "📚")




