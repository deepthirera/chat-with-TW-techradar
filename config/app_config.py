"""Base configuration for the RAG application.

This module contains the base configuration settings that are
common across all environments. Environment-specific settings
should override these in their respective config files.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
RAW_DATA_DIR = ROOT_DIR / "data"

# Config path
LLM_CONFIG_FILE = ROOT_DIR / "config" / "llm_config.yaml"

# Vector database
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tech_radar_store")
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "nomic-embed-text")

# Graph database (Neo4j)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Basic application settings
DEBUG = False
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Document validation settings
TECH_RADAR_FILENAME_PATTERN = r"tr_technology_radar"
PDF_FILE_PATTERN = "*.pdf"
REQUIRED_METADATA_FIELDS = ["creationdate"]
REGEX_PATTERN = r'\d{1,3}\. [^"\n]+\n(?:Adopt|Trial|Hold|Assess)'

SYSTEM_PROMPT = """You are an assistant for question-answering queries related to ThoughtWorks TechRadar.
Use the following pieces of retrieved context to answer the question. If you don't know the answer,
just say that you don't know. Use ten sentences maximum and keep the answer concise.

Context: {context}

Answer:
"""

GRAPH_SYSTEM_PROMPT = """
You are a query routing assistant for a TechRadar graph database system. 
Analyze the user's question and determine the most appropriate search strategy.

**Search Type Guidelines:**

**graph_semantic_search** - Use for content-based queries:
- Questions about what technologies do, their descriptions, or detailed content
- Conceptual queries about best practices, recommendations, or explanations
- Examples: "What are the blips about AI?", "What are the best practices in AWS cloud?", "Tell me about microservices approaches"

**graph_cypher_search** - Use for structural/metadata queries:
- Questions about counts, categories, positions, or organizational structure
- Queries filtering by ring (Adopt/Trial/Assess/Hold), quadrant (Tools/Techniques/Platforms/Languages & Frameworks), or volume/year
- Examples: "What are the blips under Hold in radar 32?", "How many blips are in Adopt this year?", "List all Tools in the Trial ring"

Question: {question}
Search type:"""

GRAPH_CYPHER_PROMPT = """You are an assistant for question-answering queries related to ThoughtWorks TechRadar.
Use the following pieces of retrieved context to answer the question. If you don't know the answer,
just say that you don't know. Use ten sentences maximum and keep the answer concise.

## Graph Schema Information:
The TechRadar data is stored in a Neo4j graph with the following structure:
Schema: {schema}
Question: {question}
**Nodes:**
- TechRadar: {{title, volume, period, year, creationdate, filename}}
- Quadrant: {{title}} - Values: "Techniques", "Platforms", "Tools", "Languages and Frameworks"
- Ring: {{title}} - Values: "Adopt", "Trial", "Assess", "Hold"
- Blip: {{title, content}} - Individual technologies/practices

**Relationships:**
- (Blip)-[:PUBLISHED_IN {{volume, period}}]->(TechRadar)
- (Blip)-[:CATEGORIZED_IN]->(Quadrant)
- (Blip)-[:POSITIONED_IN]->(Ring)

## Query Translation Guide:
- "assess", "trial", "adopt", "hold" → Ring nodes
- "techniques", "platforms", "tools", "languages and frameworks" → Quadrant nodes
- Years (2024, 2025) → TechRadar.year or TechRadar.period
- "technologies", "blips", "items" → Blip nodes
- Always capitalise the title of rings and quadrants like assess to "Assess", languages and frameworks to "Languages and \nFrameworks"

## Common Query Patterns:
- Technologies in specific ring: MATCH (b:Blip)-[:POSITIONED_IN]->(r:Ring {{title: "Assess"}})
- Technologies by quadrant: MATCH (b:Blip)-[:CATEGORIZED_IN]->(q:Quadrant {{title: "Tools"}})
- Technologies by time: MATCH (b:Blip)-[:PUBLISHED_IN]->(tr:TechRadar) WHERE tr.year = 2025
- Combined filters: Use multiple MATCH clauses for complex queries
- Correct pattern for technologies in Hold ring from volume 32
MATCH (b:Blip)-[:POSITIONED_IN]->(r:Ring {{title: "Hold"}})
MATCH (b)-[:PUBLISHED_IN]->(tr:TechRadar {{volume: "32"}})
RETURN b.title, b.content


Important Notes:
- Ring titles are exactly: Adopt, Trial, Assess, Hold
- Quadrant titles are exactly: Techniques, Platforms, Tools, Languages and Frameworks
- Use MATCH clauses to connect Blip nodes to Ring and Quadrant nodes
- For time-based queries, use TechRadar.year or TechRadar.period properties
- Always return meaningful properties like b.title, b.content

 
Cypher Query:
"""
