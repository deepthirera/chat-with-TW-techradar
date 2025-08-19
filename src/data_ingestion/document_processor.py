import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger


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
            final_metadata = self._process_metadata(doc_dict.metadata)
            chunks = self.split_using_lib(doc_dict.page_content)
            [ chunked_docs.extend([Document(page_content=chunk, metadata=final_metadata) ]) for chunk in chunks ]
        return chunked_docs

    def _process_metadata(self, metadata):
        source = metadata.get("source")
        filename, title_parts = "", ""
        if source:
            filename = source.split("/")[-1]
            title_parts = filename.title().split("_")[1:-1]
        return  {
            "creationdate": metadata.get("creationdate", ""),
            "filename": filename,
            "title": " ".join(title_parts),
            "volume": title_parts[-1][-2:] if title_parts else "",
            "period": "April 2025",
        }
