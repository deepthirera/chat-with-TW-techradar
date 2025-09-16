from collections import defaultdict
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

    def split_using_lib(self, docs):
        pattern = r'\d{1,3}\. [^"\n]+\n(?:Adopt|Trial|Hold|Assess)'
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, is_separator_regex=True, separators=[pattern])
        return splitter.split_text(docs)

    def graph_content(self, loaded_docs):
        """Process each document and split into chunks at title boundaries."""
        graph_content = defaultdict(dict)
        self.graph_builder.create_quadrants()
        # self.graph_builder.create_rings()
        logger.info(f"Chunking documents...{len(loaded_docs)}")
        for doc_dict in loaded_docs:
            base_metadata = self._process_base_metadata(doc_dict.metadata)
            cleaned_page_content = self._cleanup_page_content(doc_dict.page_content)
            all_blips_metadata = self._process_metadata(cleaned_page_content)
            chunks = self.split_using_lib(cleaned_page_content)
            graph_content[base_metadata["title"]]["metadata"] = base_metadata
            blips = []
            other_chunks = []
            self.graph_builder.create_radar_node(base_metadata)
            for chunk in chunks:
                chunk_title_response = re.match(r'\d{1,3}\. [^"\n]+', chunk)
                if chunk_title_response:
                    chunk_title = chunk_title_response.group(0).strip()
                    current_blip_metadata = all_blips_metadata.get(chunk_title, base_metadata)
                    blip_detail = {
                        "doc": chunk,
                        "blip_title": chunk_title,
                        **current_blip_metadata,
                        **base_metadata
                    }
                    self.graph_builder.create_blip_nodes(blip_detail)
                    blips.extend([blip_detail])
                else:
                    other_chunks.extend([Document(page_content=chunk, metadata=base_metadata) ])
            graph_content[base_metadata["title"]]["blips"] = blips
            graph_content[base_metadata["title"]]["other_chunks"] = other_chunks
        return graph_content

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
                                cleaned_item = re.sub(r"\s+", " ", ring_item).strip()
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
