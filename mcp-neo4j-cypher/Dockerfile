FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir hatchling

COPY pyproject.toml /app/

RUN pip install --no-cache-dir neo4j>=5.26.0 fastmcp>=2.10.5

RUN pip install --no-cache-dir langchain-ollama

RUN pip install --no-cache-dir langchain-core

COPY src/ /app/src/
COPY README.md /app/

RUN pip install --no-cache-dir -e .

# Default configuration
ENV NEO4J_URI="bolt://host.docker.internal:7687"
ENV NEO4J_USERNAME="neo4j"
ENV NEO4J_PASSWORD="password"
ENV NEO4J_DATABASE="neo4j"
ENV NEO4J_NAMESPACE=""
ENV NEO4J_TRANSPORT="sse"
ENV NEO4J_MCP_SERVER_HOST="0.0.0.0"
ENV NEO4J_MCP_SERVER_PORT="8080"
ENV NEO4J_MCP_SERVER_PATH="/api/mcp/"
ENV NEO4J_MCP_SERVER_ALLOW_ORIGINS="*"
ENV NEO4J_MCP_SERVER_ALLOWED_HOSTS="*"

EXPOSE 8080

CMD sh -c 'python -m mcp_neo4j_cypher \
    --db-url $NEO4J_URI \
    --username $NEO4J_USERNAME \
    --password $NEO4J_PASSWORD \
    --database $NEO4J_DATABASE \
    --transport $NEO4J_TRANSPORT \
    --server-host $NEO4J_MCP_SERVER_HOST \
    --server-port $NEO4J_MCP_SERVER_PORT \
    --server-path $NEO4J_MCP_SERVER_PATH \
    --allowed-hosts "$NEO4J_MCP_SERVER_ALLOWED_HOSTS" \
    --allow-origins "$NEO4J_MCP_SERVER_ALLOW_ORIGINS"'
