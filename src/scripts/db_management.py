from dotenv import load_dotenv

from config import RAW_DATA_DIR
from src.data_ingestion.doc_processor_with_metadata import DocProcessorWithMetadata
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.document_processor import DocumentProcessor
from src.utils.logger import logger
from src.vector_store.vector_store import VectorStore


def vector_migrate_and_seed():
    """Migrate and seed the vector database with documents."""
    RAGDataManager().migrate_and_seed(RAGDataManager.VECTOR_BASIC)

def vector_metadata_migrate_and_seed():
    """Migrate and seed the vector database with documents and related metadata."""
    RAGDataManager().migrate_and_seed(RAGDataManager.VECTOR_METADATA)

class RAGDataManager:
    VECTOR_BASIC = "vector_basic"
    VECTOR_METADATA = "vector_with_metadata"
    def __init__(self) -> None:
        load_dotenv()
        self.loaded_docs = DocumentLoader(str(RAW_DATA_DIR)).load_radar_files()

    def migrate_and_seed(self, processor_type=VECTOR_BASIC):
        if processor_type == RAGDataManager.VECTOR_BASIC:
            processor = DocumentProcessor()
        elif processor_type == RAGDataManager.VECTOR_METADATA:
            processor = DocProcessorWithMetadata()

        chunked_docs = processor.chunk_pdfs(self.loaded_docs)
        self._store_in_vectordb(chunked_docs)

    def _store_in_vectordb(self, chunked_docs):
        try:
            logger.info("\nEmbedding and storing in database")
            VectorStore().create(documents=chunked_docs)
            logger.info("\nOne time migration process complete!")
        except Exception as e:
            logger.error(f"Failed to store documents in vector database: {e}")
            raise
