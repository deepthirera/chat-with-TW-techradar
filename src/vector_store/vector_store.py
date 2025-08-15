from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config.app_config import (
    CHROMA_PATH,
    COLLECTION_NAME
)
from src.llm.model_manager import LLMModelManager

class VectorStore:
    def __init__(self, collection_name=COLLECTION_NAME, persist_directory=CHROMA_PATH):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = LLMModelManager().get_embedding_model()

    def create(self, documents):
        db = self.load()
        db.reset_collection()
        db.add_texts(documents, embedding=self.embedding_model,
                        collection_name=self.collection_name,
                        persist_directory=self.persist_directory
                    )
        return db
    
    def load(self):
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
    