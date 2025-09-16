"""Graph store implementation for Neo4j database operations.

This module provides a GraphStore class that handles connections to Neo4j
and operations for storing and retrieving graph data from technology radar documents.
"""

import os
from typing import Any, Dict, List, Optional

from click import password_option

from langchain_neo4j import Neo4jGraph

from config.app_config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)
from src.utils.logger import logger
from src.llm.model_manager import LLMModelManager
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer


class GraphStore:
    """Neo4j graph database store for technology radar data."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """Initialize GraphStore with Neo4j connection parameters.
        
        Args:
            uri: Neo4j database URI (defaults to NEO4J_URI env var or bolt://localhost:7687)
            username: Neo4j username (defaults to NEO4J_USERNAME env var or 'neo4j')
            password: Neo4j password (defaults to NEO4J_PASSWORD env var)
            database: Neo4j database name (defaults to NEO4J_DATABASE env var or 'neo4j')
        """
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.embedding_model = LLMModelManager().get_embedding_model()
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database
            )

        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            raise
    
    def create(self, query, node_data):
        self.graph.query(query, node_data)

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.graph:
            self.graph.close()
            logger.info("Neo4j connection closed")
    
    # def clear_database(self) -> None:
    #     """Clear all nodes and relationships from the database."""
    #     if not self.graph:
    #         raise RuntimeError("Neo4j driver not connected")
    #     with self.graph.session(database=self.database) as session:
    #         session.run("MATCH (n) DETACH DELETE n")
    #         logger.info("Database cleared")
    
    