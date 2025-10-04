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
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate


class TechGraphStore:
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
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.database = database or NEO4J_DATABASE
        self.embedding_model = LLMModelManager().get_embedding_model()
        self.load()
    
    def load(self):
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
    
    def build_vector_index(self, index_name, node_label, text_node_properties):
        return Neo4jVector.from_existing_graph(
            self.embedding_model,
            url=self.uri,
            username=self.username,
            password=self.password,
            index_name=index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property="embedding",
        )
    
    def as_retriever(self):
        self.graph.refresh_schema()

        cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph, 
            llm=LLMModelManager().get_chat_model(), 
            verbose=True, 
            allow_dangerous_requests=True,
            cypher_prompt=self._get_cypher_prompt(),
            qa_prompt=self._get_qa_prompt(),
        )
        return cypher_chain
    
    def _get_cypher_prompt(self):
        return PromptTemplate(
            input_variables=["schema", "question"],
            template="""
    You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query.

    Schema: {schema}

    Important Notes:
    - Ring titles are exactly: "Adopt", "Trial", "Assess", "Hold"
    - Quadrant titles are exactly: "Techniques", "Platforms", "Tools", "Languages and Frameworks"
    - Use MATCH clauses to connect Blip nodes to Ring and Quadrant nodes
    - For time-based queries, use TechRadar.year or TechRadar.period properties
    - Always return meaningful properties like b.title, b.content

    Question: {question}
    Cypher Query:"""
        )

    def _get_qa_prompt(self):
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
    Use the following context from the TechRadar graph database to answer the question.
    If you don't know the answer, just say you don't know.

    IMPORTANT: When listing technologies/blips, include ALL items from the context. Do not summarize or limit the list unless specifically asked to do so.

    Context: {context}
    Question: {question}

    Answer: List all the technologies/blips mentioned in the context."""
        )


    # def clear_database(self) -> None:
    #     """Clear all nodes and relationships from the database."""
    #     if not self.graph:
    #         raise RuntimeError("Neo4j driver not connected")
    #     with self.graph.session(database=self.database) as session:
    #         session.run("MATCH (n) DETACH DELETE n")
    #         logger.info("Database cleared")
    
    