import logging
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def setup_logging(name: str) -> logging.Logger:
    """Configures and returns a logger with the specified format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logging(__name__)

def get_env_var(var_name: str) -> str:
    """Retrieves an environment variable or raises an exception if not found."""
    val = os.getenv(var_name)
    if not val:
        logger.error(f"Missing environment variable: {var_name}")
        raise ValueError(f"Missing environment variable: {var_name}")
    return val

def semantic_chunk_text(documents: List[Document]) -> List[Document]:
    """Chunks documents into semantic pieces."""
    logger.info("Starting semantic chunking of documents.")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Document chunking failed: {str(e)}")
        raise
