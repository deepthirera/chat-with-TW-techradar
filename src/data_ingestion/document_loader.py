import re
from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader

from config import (
    PDF_FILE_PATTERN,
    TECH_RADAR_FILENAME_PATTERN,
)
from src.utils.logger import logger


class DocumentLoader:
    """Handles loading and validation of PDF documents."""

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load_radar_files(self) -> dict[str, dict[str, list[Document]]]:
        result_docs = []

        try:
            pdf_files = list(self.folder_path.glob(PDF_FILE_PATTERN))
            if not pdf_files:
                logger.warning(f"No PDF files found in directory: {self.folder_path}")
                raise ValueError(f"No PDF files found in directory: {self.folder_path}")

            for filepath in pdf_files:
                try:
                    logger.info(f"Loading {filepath}...")
                    if self._isvalid_pdf(filepath):
                        loader = PyPDFLoader(str(filepath), mode="single")
                        doc = loader.load()
                        result_docs.extend(doc)
                        logger.info(f"Successfully loaded {filepath.name}")
                    else:
                        logger.warning(f"Skipping {filepath.name} as it is not a valid Tech Radar PDF file.")
                except Exception as e:
                    logger.error(f"Error processing file {filepath.name}: {e!s}")
                    continue

        except Exception as e:
            logger.error(f"Facing exception while loading pdf files from {self.folder_path}: {e!s}")
            raise RuntimeError(f"Error accessing folder {self.folder_path}: {e!s}")

        if not result_docs:
            logger.error("Looks like loading pdf files is not successful")
            raise ValueError("Looks like loading pdf files is not successful")

        logger.info(f"Successfully loaded {len(result_docs)} Tech Radar PDF files")
        return result_docs

    def _isvalid_pdf(self, filepath: Path) -> bool:
        if filepath.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF file: {filepath}")
            return False

        if filepath.stat().st_size == 0:
            logger.error(f"Empty file: {filepath}")
            return False

        try:
            with filepath.open("rb") as f:
                pdf = PdfReader(f)

                # # Check if it's a Tech Radar PDF
                # if pdf.metadata["author"] == TECH_RADAR_AUTHOR and pdf.metadata["title"] == TECH_RADAR_TITLE:
                #     logger.error(f"PDF metadata: {filepath}")
                #     return False

                # Check filename pattern
                if not re.search(TECH_RADAR_FILENAME_PATTERN, filepath.name.lower()):
                    logger.error(f"Not a Tech Radar PDF based on filename: {filepath}")
                    return False

                # Check required metadata fields
                # for field in REQUIRED_METADATA_FIELDS:
                #     if field not in pdf.metadata:
                #         logger.error(f"Missing required metadata field '{field}': {filepath}")
                #         return False

                logger.debug(f"PDF metadata: {pdf.metadata}")
                return True

        except Exception as e:
            logger.error(f"Error validating PDF {filepath}: {e!s}")
            return False
