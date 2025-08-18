"""LLM configuration for the RAG application.

This module contains the models and their aliases for the RAG application. 
"""

DEFAULT_MODEL = "ollama-mistral"
DEFAULT_EMBEDDING_MODEL = "ollama-nominic"
OPENAI = "openai"
GEMINI = "gemini"
OLLAMA = "ollama"

MODELS_CONFIG = [
    {
        "model_alias": "gpt-5-mini",
        "model": "gpt-5-mini-2025-08-07"
    },
    {
        "model_alias": "gpt-5-nano",
        "model": "gpt-5-nano-2025-08-07"
    },
    {
        "model_alias": "gemini-2.0-flash",
        "model": "gemini/gemini-2.0-flash"
    },
    {
        "model_alias": "ollama-mistral",
        "model": "ollama/mistral",
        "api_base": "http://localhost:11434"
    },
]

EMBEDDING_MODELS_CONFIG = [
    {
        "model_alias": "openai-ada",
        "model": "text-embedding-ada-002",
        "provider": OPENAI
    },
    {
        "model_alias": "openai-embedding",
        "model": "text-embedding-3-small",
        "provider": OPENAI
    },
    {
        "model_alias": "ollama-nominic",
        "model": "nomic-embed-text",
        "api_base": "http://localhost:11434",
        "provider": OLLAMA
    },
    {
        "model_alias": "gemini-embedding-001",
        "model": "models/gemini-embedding-001",
        "provider": GEMINI
    }
]

LLM_COMMON_PARAMETERS = {
    "temperature": 0.7,
    "max_tokens": 4096
}
