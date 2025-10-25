# tests/integration/test_neo4j.py

import pytest
from neo4j import GraphDatabase

@pytest.mark.integration
def test_neo4j_connection():
    """Check Neo4j is running and accepts Bolt connections."""
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "LetsNeo4j")

    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            value = result.single()["ok"]
        driver.close()
        assert value == 1, "Unexpected Neo4j response"
    except Exception as e:
        pytest.fail(f"Neo4j connection failed: {e}")