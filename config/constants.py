"""Base configuration for the RAG application.

This module contains the base configuration settings that are
common across all environments. Environment-specific settings
should override these in their respective config files.
"""

from pathlib import Path
import os 

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Basic application settings
DEBUG = False
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Document validation settings
TECH_RADAR_FILENAME_PATTERN = r'tr_technology_radar'
PDF_FILE_PATTERN = "*.pdf"
REQUIRED_METADATA_FIELDS = ['creationdate']
