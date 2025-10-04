import chunk
from collections import defaultdict
from math import log
import re
from datetime import datetime
from langchain_core.documents import Document
from regex import B
from src.data_ingestion.tech_radar_graph_builder import TechRadarGraphBuilder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger



class GraphProcessorWithMetadata:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.graph_builder = TechRadarGraphBuilder()

    # def split_using_lib(self, docs):
    #     pattern = r'\d{1,3}\. [^"\n]+\n(?:Adopt|Trial|Hold|Assess)'
    #     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, is_separator_regex=True, separators=[pattern])
    #     return splitter.split_text(docs)

    def split_using_lib(self, docs):
    # Split on the pattern that identifies technology starts
        separator_pattern = r'(?=\d{1,3}\.\s[^"\n]+\n(?:Adopt|Trial|Hold|Assess))'
        # separator_pattern = r'\d{1,3}\. [^"\n]+\n(?:Adopt|Trial|Hold|Assess)'
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            is_separator_regex=True, 
            separators=[separator_pattern],
            chunk_overlap=0
        )
        
        chunks = splitter.split_text(docs)
        # Filter out chunks that are just lists (contain multiple numbered items)
        filtered_chunks = []
        for chunk in chunks:
            # chunk = chunk.strip()
            if chunk and re.match(r'\d{1,3}\.\s', chunk):
                numbered_items = len(re.findall(r'\n\d{1,3}\.', chunk))
                # If there's only one numbered item (the main one), it's likely valid
                if numbered_items <= 1:  # Allow 0 or 1 additional numbered items
                    filtered_chunks.append(chunk)
        self._write_to_file(chunks=filtered_chunks)
        return filtered_chunks
    
    # def split_using_lib(self, docs):
    # # Use a separator pattern that splits BEFORE each numbered item
    #     separator_pattern = r'(?=\d{1,3}\.\s[^"\n]+\n(?:Adopt|Trial|Hold|Assess))'
        
    #     splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,  # Increase chunk size to avoid cutting off content
    #         chunk_overlap=0,   # No overlap needed for this use case
    #         is_separator_regex=True, 
    #         separators=[separator_pattern]
    #     )
        
    #     chunks = splitter.split_text(docs)
    #     # Manual splitting - no size limits
    #     # pattern = r'(?=\d{1,3}\.\s[^"\n]+\n(?:Adopt|Trial|Hold|Assess))'
    #     # chunks = re.split(pattern, docs)
        
    #     # # Remove empty chunks
    #     # chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    #     self._write_to_file(chunks=chunks, filename="before_chunks.txt")
    #     # Simple filter: keep only chunks that have descriptive content after ring
    #     valid_chunks = []
    #     for chunk in chunks:
    #         chunk = chunk.strip()
    #         if not chunk:
    #             continue
                
    #         lines = chunk.split('\n')
    #         if len(lines) >= 3:  # At least: title, ring, some content
    #             # Check if this looks like a valid technology entry
    #             first_line = lines[0].strip()
    #             if re.match(r'\d{1,3}\.\s', first_line):  # Starts with number
    #                 # Find the ring line
    #                 for i, line in enumerate(lines[1:3], 1):  # Check next 2 lines
    #                     if line.strip() in ['Adopt', 'Trial', 'Hold', 'Assess']:
    #                         # Check if there's content after the ring (not just numbers)
    #                         if i + 1 < len(lines):
    #                             next_line = lines[i + 1].strip()
    #                             # If next line is not empty and not a numbered item
    #                             if next_line and not re.match(r'\d{1,3}\.', next_line):
    #                                 valid_chunks.append(chunk)
    #                         break
    #     self._write_to_file(chunks=chunks, filename="after_chunks.txt")
    #     return valid_chunks


    def _write_to_file(self, chunks, filename="chunks.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"CHUNK {i}:\n")
                f.write("-" * 10 + "\n")
                f.write(chunk)
                f.write("\n" + "=" * 40 + "\n\n")
    
    def _write_doc_to_file(self, content, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)

    def graph_content(self, loaded_docs):
        """Process each document and split into chunks at title boundaries."""
        logger.info(f"Chunking documents...{len(loaded_docs)}")
        for doc_dict in loaded_docs:
            base_metadata = self._process_base_metadata(doc_dict.metadata)
            cleaned_page_content = self._cleanup_page_content(doc_dict.page_content)
            all_blips_metadata = self._process_metadata(cleaned_page_content)
            chunks = self.split_using_lib(cleaned_page_content)
            self.graph_builder.create_radar_node(base_metadata)
            for chunk in chunks:
                chunk_title_response = re.match(r'(\d{1,3}\.) ([^"\n]+)', chunk)
                if chunk_title_response:
                    chunk_title = chunk_title_response.group(2).strip()
                    current_blip_metadata = all_blips_metadata.get(chunk_title, base_metadata)
                    # if(current_blip_metadata.get("ring") == "Adopt"):
                        # logger.info(f"{chunk_title} in {current_blip_metadata["quadrant"]} in {current_blip_metadata["ring"]}") 
                    blip_detail = {
                        "doc": chunk,
                        "blip_title": chunk_title,
                        **current_blip_metadata,
                        **base_metadata
                    }
                    try:
                        self.graph_builder.create_blip_nodes(blip_detail)
                    except Exception as e:
                        print(f"Query failed with error: {e}")
                        print(f"Parameters: {blip_detail}")
        return self.graph_builder

    def _cleanup_page_content(self, raw_page_content):
        mega_pattern = (
            r"(Hold\s+HoldAssess\s+AssessTrial\s+TrialAdopt\s+Adopt\s*\n(?:\s*\d+(?:\s+\d+)*\s*\n?)*)"
            r"|(©\s*Thoughtworks,\s*Inc\.\s*All\s*Rights\s*Reserved\.(?:\s*\n\s*\d+)?)"
            r"|(?:New\s+Moved\s+in/out\s+No\s+change)"
        )
        cleaned = re.sub(mega_pattern, "", raw_page_content, flags=re.MULTILINE)
        return cleaned.strip()

    def _process_metadata(self, cleaned_text):
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
                                chunk_title_response = re.match(r'(\d{1,3}\.) ([^"\n]+)', ring_item)
                                if chunk_title_response:
                                    chunk_title = chunk_title_response.group(2)
                                    cleaned_item = re.sub(r"\s+", " ", chunk_title).strip()
                                    result[cleaned_item] = {
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
        filename, title_parts, creation_date, period = "", "", "", ""
        if source:
            filename = source.split("/")[-1] 
            title_parts = filename.title().split("_")[1:-1] # error handling required
        creation_date = metadata.get("creationdate", "")
        period = datetime.fromisoformat(creation_date).strftime("%B %Y") if creation_date else "" # error handling required
        return  {
            "creationdate": creation_date,
            "filename": filename,
            "title": " ".join(title_parts),
            "volume": title_parts[-1][-2:] if title_parts else "",
            "period": period,
        }
