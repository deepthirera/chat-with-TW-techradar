import unittest
from unittest.mock import patch, Mock
from langchain_core.documents import Document
from src.data_ingestion.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()
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
        
    @patch('src.data_ingestion.document_processor.RecursiveCharacterTextSplitter')
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
        result = self.processor._process_metadata(metadata)
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
        result = self.processor._process_metadata(metadata)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
