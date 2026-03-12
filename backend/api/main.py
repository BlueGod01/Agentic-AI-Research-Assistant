import os
import json
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
from backend.utils import setup_logging
from backend.storage import upload_to_s3
from backend.pipeline import process_document
from backend.agents import build_agent_graph
from langchain_core.messages import HumanMessage

logger = setup_logging(__name__)

app = FastAPI(title="Agentic RAG Research Assistant API")

class QueryRequest(BaseModel):
    query: str
    generate_pdf: bool = False

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads document to S3 and triggers the ingestion pipeline."""
    logger.info(f"Received upload request for file: {file.filename}")
    try:
        content = await file.read()
        local_tmp = f"/tmp/{file.filename}"
        
        with open(local_tmp, "wb") as f:
            f.write(content)
            
        # Simulating S3 upload flow directly from backend to bypass presigned URL for simplicity
        upload_to_s3(local_tmp, file.filename)
        
        # Trigger processing pipeline synchronously (should be async via Celery in large scale production)
        process_document(file.filename)
        
        logger.info("Upload and processing complete.")
        return {"status": "success", "message": f"File {file.filename} uploaded and processed."}
    except Exception as e:
        logger.error(f"Error in /upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during upload")

async def stream_agent_responses(query: str, generate_pdf: bool) -> AsyncGenerator[str, None]:
    """Streams the agent graph execution."""
    try:
        graph = build_agent_graph()
        initial_state = {"messages": [HumanMessage(content=query)], "generate_pdf": generate_pdf}
        
        # Async stream over graph workflow events
        for s in graph.stream(initial_state, stream_mode="updates"):
            yield f"data: {json.dumps(str(s))}\n\n"
            await asyncio.sleep(0.01) # Non-blocking yield
    except Exception as e:
        logger.error(f"Error during graph streaming: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/query")
async def handle_query(req: QueryRequest):
    """Handles user query and streams LangGraph agent workflow responses."""
    logger.info(f"Received query request: {req.query}")
    try:
        return StreamingResponse(
            stream_agent_responses(req.query, req.generate_pdf),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in /query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
