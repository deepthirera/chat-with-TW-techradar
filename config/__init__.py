# """Configuration management for the RAG application.

# This module handles loading the appropriate configuration
# based on the environment (development/production).
# """
# from typing import Dict, Any
# from config.constants import *

# def get_config() -> Dict[str, Any]:
#     """Get the configuration based on the environment."""
#     config_dict = {k: v for k, v in globals().items() if k.isupper()}
#     return config_dict

# config = get_config()
