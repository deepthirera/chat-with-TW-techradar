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

GRAPH_SYSTEM_PROMPT = """You are an assistant for question-answering queries related to ThoughtWorks TechRadar.
Use the following pieces of retrieved context to answer the question. If you don't know the answer,
just say that you don't know. Use ten sentences maximum and keep the answer concise.

## Graph Schema Information:
The TechRadar data is stored in a Neo4j graph with the following structure:

**Nodes:**
- TechRadar: {title, volume, period, year, creationdate, filename}
- Quadrant: {title} - Values: "Techniques", "Platforms", "Tools", "Languages and Frameworks"
- Ring: {title} - Values: "Adopt", "Trial", "Assess", "Hold"
- Blip: {title, content} - Individual technologies/practices

**Relationships:**
- (Blip)-[:PUBLISHED_IN {volume, period}]->(TechRadar)
- (Blip)-[:CATEGORIZED_IN]->(Quadrant)
- (Blip)-[:POSITIONED_IN]->(Ring)

## Query Translation Guide:
- "assess", "trial", "adopt", "hold" → Ring nodes
- "techniques", "platforms", "tools", "languages and frameworks" → Quadrant nodes
- Years (2024, 2025) → TechRadar.year or TechRadar.period
- "technologies", "blips", "items" → Blip nodes
- Always capitalise the title of rings and quadrants like assess to "Assess", languages and frameworks to "Languages and \nFrameworks"

## Common Query Patterns:
- Technologies in specific ring: MATCH (b:Blip)-[:POSITIONED_IN]->(r:Ring {title: "Assess"})
- Technologies by quadrant: MATCH (b:Blip)-[:CATEGORIZED_IN]->(q:Quadrant {title: "Tools"})
- Technologies by time: MATCH (b:Blip)-[:PUBLISHED_IN]->(tr:TechRadar) WHERE tr.year = 2025
- Combined filters: Use multiple MATCH clauses for complex queries

Context: {context}

Answer:
"""
