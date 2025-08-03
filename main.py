from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.document_processor import DocumentProcessor
from config.constants import RAW_DATA_DIR
from src.llm.model_manager import LLMModelManager

def main():
    docs_by_created_date = DocumentLoader(RAW_DATA_DIR).load_radar_files()
    chunked_docs_by_created_date = DocumentProcessor().chunk_pdfs(docs_by_created_date)
    llm_manager = LLMModelManager()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    
    response = llm_manager.chat_completion(messages)
    print("\nResponse:")
    print(response['choices'][0]['message']['content'])
    
    
if __name__ == "__main__":
    main()
