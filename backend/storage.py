import os
import boto3
from pinecone import Pinecone
from botocore.exceptions import ClientError
from backend.utils import setup_logging, get_env_var
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain_core.documents import Document

logger = setup_logging(__name__)

def get_s3_client():
    """Initializes the S3 boto3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=get_env_var("AWS_ACCESS_KEY"),
        aws_secret_access_key=get_env_var("AWS_SECRET_KEY"),
        region_name=get_env_var("AWS_REGION")
    )

def upload_to_s3(file_path: str, file_key: str) -> bool:
    """Uploads a file to S3."""
    logger.info(f"Uploading {file_path} to S3 bucket {get_env_var('S3_BUCKET_NAME')} as {file_key}")
    try:
        s3 = get_s3_client()
        s3.upload_file(file_path, get_env_var("S3_BUCKET_NAME"), file_key)
        logger.info("S3 upload successful.")
        return True
    except ClientError as e:
        logger.error(f"S3 upload failed: {str(e)}")
        raise

def download_from_s3(file_key: str, download_path: str) -> bool:
    """Downloads a file from S3."""
    logger.info(f"Downloading {file_key} from S3")
    try:
        s3 = get_s3_client()
        s3.download_file(get_env_var("S3_BUCKET_NAME"), file_key, download_path)
        logger.info("S3 download successful.")
        return True
    except ClientError as e:
        logger.error(f"S3 download failed: {str(e)}")
        return False

def generate_presigned_url(file_key: str) -> str:
    """Generates a presigned URL for uploading to S3."""
    logger.info(f"Generating presigned url for {file_key}")
    try:
        s3 = get_s3_client()
        url = s3.generate_presigned_url(
            ClientMethod='put_object',
            Params={'Bucket': get_env_var("S3_BUCKET_NAME"), 'Key': file_key},
            ExpiresIn=3600
        )
        return url
    except ClientError as e:
        logger.error(f"Presigned URL generation failed: {str(e)}")
        raise

def get_pinecone_client() -> Pinecone:
    """Initializes and returns the Pinecone client."""
    pc = Pinecone(api_key=get_env_var("PINECONE_API_KEY"))
    return pc

def store_embeddings(documents: List[Document]):
    """Stores a list of Langchain Documents into Pinecone vector store."""
    logger.info(f"Storing {len(documents)} chunks into Pinecone.")
    try:
        pc = get_pinecone_client()
        index_name = get_env_var("PINECONE_INDEX_NAME")
        embeddings = OpenAIEmbeddings(api_key=get_env_var("LLM_API_KEY"))
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )
        logger.info("Successfully stored embeddings in Pinecone.")
    except Exception as e:
        logger.error(f"Embedding storage failed: {str(e)}")
        raise

def query_embeddings(query: str, top_k: int = 5) -> List[Document]:
    """Queries Pinecone for relevant documents."""
    logger.info(f"Querying vector store for: {query}")
    try:
        pc = get_pinecone_client()
        index_name = get_env_var("PINECONE_INDEX_NAME")
        embeddings = OpenAIEmbeddings(api_key=get_env_var("LLM_API_KEY"))
        vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
        return vector_store.similarity_search(query, k=top_k)
    except Exception as e:
        logger.error(f"Pinecone querying failed: {str(e)}")
        raise
