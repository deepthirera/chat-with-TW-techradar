from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.document_processor import DocumentProcessor
from config import RAW_DATA_DIR
from src.vector_store.vector_store import VectorStore
from dotenv import load_dotenv
from src.utils.logger import logger

def migrate_and_seed():
    """Migrate and seed the vector database with documents."""
    load_dotenv()
    loaded_docs = DocumentLoader(RAW_DATA_DIR).load_radar_files()
    chunked_docs = DocumentProcessor().chunk_pdfs(loaded_docs)
    logger.info("\nEmbedding and storing in database")
    db = VectorStore().create(documents=chunked_docs)
    logger.info("\nOne time migration process complete!")

if __name__ == "__main__":
    migrate_and_seed()
