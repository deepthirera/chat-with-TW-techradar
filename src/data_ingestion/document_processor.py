import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import logger
from functools import reduce
class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_by_titles(self, text):
        """Split text at numbered titles followed by Adopt/Trial/Hold/Assess."""
        pattern = r'\d{1,3}\. [^"\n]+\n(?:Adopt|Trial|Hold|Assess)'
        
        # Find all title positions
        matches = list(re.finditer(pattern, text))
        if not matches:
            return [text]
            
        # Get start positions and add end of text
        positions = [m.start() for m in matches]
        positions.append(len(text))
        
        # Split at each position
        chunks = []
        for i in range(len(positions)-1):
            start = positions[i]
            end = positions[i+1]
            chunk = text[start:end].strip()
            chunks.append(chunk)
        
        return chunks

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
            chunk_metadata = reduce(lambda result_string, metadata_tuple: f"{result_string}{metadata_tuple[0]}: {metadata_tuple[1]} " if metadata_tuple[0] in ['creationdate', 'source'] else f"{result_string}", doc_dict.metadata.items(), "")
            rich_chunks = [ chunk_metadata + "\n" + chunk for chunk in chunks]
            chunked_docs.extend(rich_chunks)
        return chunked_docs
