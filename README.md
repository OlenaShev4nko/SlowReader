# SlowReader: Graph-Augmented Reading Assistant

SlowReader is an intelligent **hybrid RAG (Retrieval-Augmented Generation)** application designed to help users read, understand, and organize complex technical documents. It combines **Streamlit**, **Neo4j**, and **Ollama** to extract semantic triplets, build knowledge graphs, and generate precise context-aware responses.

---

##  Components

- **Frontend interface:** [Streamlit](https://streamlit.io/)
- **Graph Database:** [Neo4j](https://neo4j.com/)
- **LLM Provider:** [Ollama](https://ollama.com/) (Local Llama 3.1)
- **LLM Model:** llama3.1:8b
- **Embeddings:** mxbai-embed-large
- **MCP Server:** [mcp-neo4j-cypher](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher) 

---

## User Flow

This guide describes how to set up and run the **SlowReader Hybrid RAG App** locally.



### 1️⃣ Install Ollama and Models

Install [Ollama](https://ollama.com/) and pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

---

### 2️⃣ Install Docker

Install [Docker Desktop](https://www.docker.com/) for your operating system.

---

### 3️⃣ Clone this Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

---

### 4️⃣ Create a `.env` File

At the project root, create a file named `.env` and add the following configuration:

```dotenv
# Ollama configuration
OLLAMA_HOST=http://host.docker.internal:11434
EMBEDDINGS_MODEL=mxbai-embed-large
CHAT_MODEL=llama3.1:8b

# Neo4j + MCP configuration
NEO4J_MCP_URL=http://neo4j_mcp:8080/api/mcp/
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=LetsNeo4j
```

---

### 5️⃣ Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/unit -q
```

### 6️⃣ Run the Application

```bash
docker-compose up -d --build
pytest -m integration -v
```

✅ You should see:
- Streamlit app imports fine
- Ollama chat + embeddings endpoints work
- Neo4j is reachable
- MCP (microservice) is responding

If all tests pass, your environment is ready.

### 7️⃣ Access Interfaces

- **Streamlit App:** [http://localhost:8501/](http://localhost:8501/)
- **Neo4j Browser:** [http://localhost:7474/browser/](http://localhost:7474/browser/)

To stop everything:
```bash
docker-compose down
deactivate
```

### 8️⃣ Clean or Reset Neo4j Database

If you want to erase all data and start with a clean graph, open the Neo4j Browser at http://localhost:7474/browser/ and execute the following Cypher commands:

```sypher
// Delete all nodes and relationships
MATCH (n)
DETACH DELETE n;

// Show all existing indexes
SHOW INDEXES;

// Drop the vector index if it exists
DROP INDEX chunk_embeddings IF EXISTS;
```
This will:

- Remove all nodes (Documents, Chunks, Concepts, etc.)
- Remove relationships between them
- Drop the vector index used for embeddings (chunk_embeddings)

You can now re-upload PDFs or re-ingest data from scratch.

---

## Screenshots & Workflow


<img src="assets/app_architecture.png" width="100%" alt="App Architecture Overview">


---

### Tab 1 — Upload a PDF

<img src="assets/Tab_1.jpg" width="100%" alt="Tab 1 — Upload a PDF">
<img src="assets/pdf_upload.png" width="100%" alt="PDF Upload BPMN Diagram">
<img src="assets/Book_in_neo4j.png" width="100%" alt="Book in neo4j">

---

### Tab 2 — Build Knowledge Graph
<img src="assets/Tab_2_1.png" width="100%" alt="Knowledge Graph Construction1">
<img src="assets/Tab_2_2.png" width="100%" alt="Knowledge Graph Construction2">
<img src="assets/build_Knowledge_graph.png" width="100%" alt="Knowledge Graph Build Step 1">
<img src="assets/graph.png" width="100%" alt="Graph Visualization">


---

### Tab 3 — Chat with Hybrid RAG
<img src="assets/Tab_3_chat.png" width="100%" alt="Chat Interface">
<img src="assets/Tab_3_chat_1.png" width="100%" alt="Chat Answer Example">

<img src="assets/hybrid_RAG.png" width="100%" alt="Hybrid RAG Diagram">


---

### Tab 4 — Library & Saved Documents

<img src="assets/Tab_4_Library.png" width="100%" alt="Library of Uploaded Books">

---

## 🪪 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

---


## Acknowledgements

This project builds upon the excellent [mcp-neo4j-cypher](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher)
server by the Neo4j Contrib community (Apache 2.0 License).

The current version has been extended and adapted by **Olena Shevchenko (2025)**  
to include additional MCP tools for hybrid RAG and local embedding workflows:
Added three custom MCP tools: link_concept_chunks, search_chunks_by_embedding, get_concept_subgraph