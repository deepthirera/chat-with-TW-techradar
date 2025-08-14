from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config.app_config import (
    CHROMA_PATH,
    COLLECTION_NAME,
)
from src.llm.model_manager import LLMModelManager

def get_vector_db(documents):
    embedding = LLMModelManager().get_embedding_model()
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    ).from_texts(documents, embedding=embedding)

    return db
