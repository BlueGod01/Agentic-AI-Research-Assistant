import os
from langchain_community.document_loaders import UnstructuredFileLoader
from backend.utils import setup_logging, semantic_chunk_text
from backend.storage import download_from_s3, store_embeddings

logger = setup_logging(__name__)

def process_document(file_key: str):
    """Downloads file from S3, parses it, chunks it, and stores embeddings in Pinecone."""
    logger.info(f"Starting document pipeline for {file_key}")
    local_path = f"/tmp/{file_key}"
    
    try:
        # Step 1: Download from S3
        download_from_s3(file_key, local_path)
        
        # Step 2: Parse document using Unstructured
        logger.info(f"Parsing document {local_path} using Unstructured.")
        loader = UnstructuredFileLoader(local_path)
        docs = loader.load()
        logger.info(f"Document parsed into {len(docs)} LangChain Documents.")
        
        # Inject metadata
        for doc in docs:
            doc.metadata['source_file'] = file_key
        
        # Step 3: Semantic chunking
        chunks = semantic_chunk_text(docs)
        
        # Step 4: Generate and store embeddings
        store_embeddings(chunks)
        logger.info(f"Pipeline completed successfully for {file_key}")
        
    except Exception as e:
        logger.error(f"Document processing pipeline failed for {file_key}: {str(e)}")
        raise
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Cleaned up local file {local_path}")
