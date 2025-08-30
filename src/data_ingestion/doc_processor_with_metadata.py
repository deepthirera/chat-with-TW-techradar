import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger


class DocProcessorWithMetadata:
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
        logger.info(f"Chunking documents...{len(loaded_docs)}")
        for doc_dict in loaded_docs:
            base_metadata = self._process_base_metadata(doc_dict.metadata)
            cleaned_page_content = self._cleanup_page_content(doc_dict.page_content)
            final_metadata = self._process_metadata(cleaned_page_content, base_metadata)
            chunks = self.split_using_lib(cleaned_page_content)
            for chunk in chunks:
                chunk_title_response = re.match(r'\d{1,3}\. [^"\n]+', chunk)
                if chunk_title_response:
                    chunk_title = chunk_title_response.group(0).strip()
                    chunked_docs.extend([Document(page_content=chunk, metadata=final_metadata.get(chunk_title, base_metadata)) ])
                else:
                    chunked_docs.extend([Document(page_content=chunk, metadata=base_metadata) ])
        return chunked_docs

    def _cleanup_page_content(self, raw_page_content):
        mega_pattern = (
            r"(Hold\s+HoldAssess\s+AssessTrial\s+TrialAdopt\s+Adopt\s*\n(?:\s*\d+(?:\s+\d+)*\s*\n?)*)"
            r"|(©\s*Thoughtworks,\s*Inc\.\s*All\s*Rights\s*Reserved\.(?:\s*\n\s*\d+)?)"
            r"|(?:New\s+Moved\s+in/out\s+No\s+change)"
        )
        cleaned = re.sub(mega_pattern, "", raw_page_content, flags=re.MULTILINE)
        return cleaned.strip()

    def _process_metadata(self, cleaned_text, base_metadata):
        result = {}
        for quadrant_title in ["Techniques", "Platforms", "Tools", "Languages and \nFrameworks"]:
            quadrant_search_response = self._extract_whole_quadrant(quadrant_title, cleaned_text)

            if quadrant_search_response:
                quadrant_data = quadrant_search_response.group(0)
                ring_ends = {
                    "Adopt": "Trial",
                    "Trial": "Assess",
                    "Assess": "Hold",
                    "Hold": r"\s*\n|\Z",
                }
                for ring_title in ["Adopt", "Trial", "Assess", "Hold"]:
                    ring_search_response = self._extract_ring(ring_title, ring_ends[ring_title], quadrant_data)
                    if ring_search_response:
                        ring_data = ring_search_response.group(0)
                        for searched_ring_item in re.findall(r"\d+\..*?(?=\n\d+\.|\n*$)", ring_data, re.DOTALL):
                            ring_item = searched_ring_item.strip()
                            if ring_item and re.match(r"\d+\.", ring_item):
                                cleaned_item = re.sub(r"\s+", " ", ring_item).strip()
                                result[cleaned_item] = {
                                    **base_metadata,
                                    "quadrant": quadrant_title,
                                    "ring": ring_title,
                                }
        return result

    def _extract_whole_quadrant(self, quadrant_name, cleaned_text):
        quadrant_pattern = rf"{quadrant_name}\s*\n\s*Adopt(.*?Trial.*?Assess.*?Hold.*?)(?=\n\s*\n|$)"
        return re.search(quadrant_pattern, cleaned_text, re.DOTALL)

    def _extract_ring(self, ring_title, next_ring_title, quadrant_data):
        ring_pattern = rf"{ring_title}\s*\n(.*?)(?=\n{next_ring_title})"
        return re.search(ring_pattern, quadrant_data, re.DOTALL)

    def _process_base_metadata(self, metadata):
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
