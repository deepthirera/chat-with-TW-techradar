from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.document_processor import DocumentProcessor
from config.constants import RAW_DATA_DIR

def main():
    docs_by_created_date = DocumentLoader(RAW_DATA_DIR).load_radar_files()
    DocumentProcessor().chunk_pdfs(docs_by_created_date)
    
if __name__ == "__main__":
    main()
