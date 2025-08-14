from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.document_processor import DocumentProcessor
from config import RAW_DATA_DIR, SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_litellm import ChatLiteLLM
from src.vector_store.get_vector_db import get_vector_db
from src.llm.model_manager import LLMModelManager
from dotenv import load_dotenv


def main():
    load_dotenv()
    loaded_docs = DocumentLoader(RAW_DATA_DIR).load_radar_files()
    chunked_docs = DocumentProcessor().chunk_pdfs(loaded_docs)
    llm = LLMModelManager().get_chat_model()
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    print("\nEmbed Response:")
    
    retriever = get_vector_db(documents=chunked_docs).as_retriever()

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    result = rag_chain.invoke("What is the volume number of the latest Radar? List down the trial topics")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(result)

if __name__ == "__main__":
    main()
