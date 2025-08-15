"""LLM configuration for the RAG application.

This module contains the models and their aliases for the RAG application. 
"""

DEFAULT_MODEL = "ollama-mistral"
DEFAULT_EMBEDDING_MODEL = "ollama-nominic"
OPENAI = "openai"
GOOGLE = "google"
OLLAMA = "ollama"

MODELS_CONFIG = [
    {
        "model_alias": "openai-gpt4",
        "model": "gpt-4-1106-preview"
    },
    {
        "model_alias": "openai-gpt35",
        "model": "gpt-3.5-turbo"
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
        "model_alias": "ollama-nominic",
        "model": "ollama/nomic-embed-text",
        "api_base": "http://localhost:11434",
        "provider": OLLAMA
    },
    {
        "model_alias": "gemini-embedding-001",
        "model": "models/gemini-embedding-001",
        "provider": GOOGLE
    }
]

LLM_COMMON_PARAMETERS = {
    "temperature": 0.7,
    "max_tokens": 4096
}
