
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger


class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_using_lib(self, docs):
        pattern = r'\d{1,3}\. [^"\n]+\n(?:Adopt|Trial|Hold|Assess)'
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, is_separator_regex=True, separators=[pattern])
        return splitter.split_text(docs)

    def chunk_pdfs(self, loaded_docs):
        """Process each document and split into chunks at title boundaries."""
        chunked_docs = []
        logger.info("Chunking documents...")
        for doc_dict in loaded_docs:
            chunks = self.split_using_lib(doc_dict.page_content)
            [ chunked_docs.extend([Document(page_content=chunk, metadata=doc_dict.metadata) ]) for chunk in chunks ]
        return chunked_docs
