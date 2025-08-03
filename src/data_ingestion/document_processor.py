import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger

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

  def chunk_pdfs(self, docs_by_created_date):
      """Process each document and split into chunks at title boundaries."""

      logger.info("Chunking documents...")
      for created_at, doc_dict in docs_by_created_date.items():
          docs_by_created_date[created_at]["split_docs"] = []
          chunks = self.split_using_lib(doc_dict["original_doc"].page_content)
          docs_by_created_date[created_at]["split_docs"].extend(chunks)
          logger.info(f"Split document {doc_dict['original_doc'].metadata["source"].split('/')[-1]} into {len(docs_by_created_date[created_at]['split_docs'])} chunks")
          # for doc in docs_by_created_date[created_at]["split_docs"]:
          #     print("=============")
          #     print(doc)
      return docs_by_created_date
