# src/neo4j_client.py
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
    auth=(
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "your_password")
    ),
)

def test_connection():
    with driver.session() as s:
        res = s.run("RETURN 1 AS ok").single()
        return res["ok"]
