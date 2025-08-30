import unittest
from unittest.mock import patch, Mock
from langchain_core.documents import Document
from src.data_ingestion.doc_processor_with_metadata import DocProcessorWithMetadata

class TestDocProcessorWithMetadata(unittest.TestCase):
    def setUp(self):
        self.processor = DocProcessorWithMetadata()
        self.sample_metadata = {
            'producer': 'Adobe PDF Library 17.0',
            'creator': 'Adobe InDesign 20.1 (Macintosh)',
            'creationdate': '2025-04-08T11:51:49-03:00',
            'author': 'Thoughtworks',
            'moddate': '2025-04-08T11:51:53-03:00',
            'title': 'Thoughtworks Technology Radar',
            'trapped': '/False',
            'source': '/Users/deepthir/projects/code/tech_radar_chat_anthropic/data/tr_technology_radar_vol_32_en.pdf',
            'total_pages': 46
        }
        self.sample_text = "This is a sample text"
        
    @patch('src.data_ingestion.doc_processor_with_metadata.RecursiveCharacterTextSplitter')
    def test_chunk_pdfs(self, mock_splitter):
        mock_splitter_instance = Mock()
        mock_splitter.return_value = mock_splitter_instance
        
        expected_chunks = ["Chunk 1", "Chunk 2"]
        mock_splitter_instance.split_text.return_value = expected_chunks
        expected_metadata = {
          'creationdate': '2025-04-08T11:51:49-03:00',
          'filename': 'tr_technology_radar_vol_32_en.pdf',
          'title': 'Technology Radar Vol 32',
          'volume': '32',
          'period': 'April 2025'
        }
        
        test_doc = Document(
            page_content=self.sample_text,
            metadata=self.sample_metadata
        )
        
        result = self.processor.chunk_pdfs([test_doc])
        
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Document)
        self.assertEqual(result[0].page_content, "Chunk 1")
        self.assertEqual(result[1].page_content, "Chunk 2")
        self.assertEqual(result[0].metadata, expected_metadata)
        self.assertEqual(result[1].metadata, expected_metadata)
        
        # Verify mock was called correctly
        mock_splitter_instance.split_text.assert_called_once_with(self.sample_text)

    def test_process_metadata_missing_source(self):
        """Test metadata processing with missing source"""
        metadata = {'creationdate': '2025-04-08T11:51:49-03:00'}
        expected = {
            'creationdate': '2025-04-08T11:51:49-03:00',
            'filename': '',
            'title': '',
            'volume': '',
            'period': 'April 2025'
        }
        result = self.processor._process_base_metadata(metadata)
        self.assertEqual(result, expected)

    def test_process_metadata_invalid_source(self):
        """Test metadata processing with invalid source path"""
        metadata = {'source': ''}
        expected = {
            'creationdate': '',
            'filename': '',
            'title': '',
            'volume': '',
            'period': 'April 2025'
        }
        result = self.processor._process_base_metadata(metadata)
        self.assertEqual(result, expected)

    def test_data_extraction(self):
        cleaned_text = """
Tools Languages and Frameworks

Techniques
Adopt
1. 1% canary
2. Retrieval-augmented generation (RAG) 
Trial
12. Using GenAI to understand 
legacy codebases 
Assess
13. AI team assistants
14. Dynamic few-shot prompting
Hold
20. Complacency with AI-generated code
21. Enterprise-wide integration 
test environments

Techniques
1. 1% canary
Adopt
For many years, we’ve used the canary release approach to encourage early feedback on new 
software versions, while reducing risk through incremental rollout to selected users. 
"""
        base_metadata = {
          'creationdate': '2025-04-08T11:51:49-03:00',
          'filename': 'tr_technology_radar_vol_32_en.pdf',
          'title': 'Technology Radar Vol 32',
          'volume': '32',
          'period': 'April 2025'
        }
        quadrant_metadata = self.processor._process_metadata(cleaned_text, base_metadata)
        expected_metadata = {
            "1. 1% canary": {
                "quadrant": "Techniques",
                "ring": "Adopt"
            },
            "2. Retrieval-augmented generation (RAG)": {
                "quadrant": "Techniques",
                "ring": "Adopt"
            },
            "12. Using GenAI to understand legacy codebases": {
                "quadrant": "Techniques",
                "ring": "Trial"
            },
            "13. AI team assistants": {
                "quadrant": "Techniques",
                "ring": "Assess"
            },
            "14. Dynamic few-shot prompting": {
                "quadrant": "Techniques",
                "ring": "Assess"
            },
            "20. Complacency with AI-generated code": {
                "quadrant": "Techniques",
                "ring": "Hold"
            },
            "21. Enterprise-wide integration test environments": {
                "quadrant": "Techniques",
                "ring": "Hold"
            }
        }
        
        # Assert that all expected items are present in the result
        for item_name, expected_data in expected_metadata.items():
            self.assertIn(item_name, quadrant_metadata, f"Item '{item_name}' not found in quadrant_metadata")
            actual_data = quadrant_metadata[item_name]
            self.assertEqual(actual_data["quadrant"], expected_data["quadrant"], 
                           f"Quadrant mismatch for '{item_name}'")
            self.assertEqual(actual_data["ring"], expected_data["ring"], 
                           f"Ring mismatch for '{item_name}'")
            
        # Assert that the base metadata is included in each item
        for item_name, item_data in quadrant_metadata.items():
            self.assertEqual(item_data["creationdate"], base_metadata["creationdate"])
            self.assertEqual(item_data["filename"], base_metadata["filename"])
            self.assertEqual(item_data["title"], base_metadata["title"])
            self.assertEqual(item_data["volume"], base_metadata["volume"])
            self.assertEqual(item_data["period"], base_metadata["period"])

if __name__ == '__main__':
    unittest.main()
