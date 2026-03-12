import os
from langchain.tools import tool
from backend.storage import query_embeddings
from backend.utils import setup_logging
from langchain_community.tools import DuckDuckGoSearchRun
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import uuid

logger = setup_logging(__name__)

@tool
def vector_search_tool(query: str) -> str:
    """Queries the Pinecone vector database for relevant context based on user documents."""
    logger.info("Executing vector_search_tool")
    try:
        docs = query_embeddings(query)
        if not docs:
            return "No relevant information found in uploaded documents."
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        logger.error(f"vector_search_tool failed: {str(e)}")
        return f"Error executing vector search: {str(e)}"

@tool
def web_search_tool(query: str) -> str:
    """Searches the web for external information using DuckDuckGo."""
    logger.info(f"Executing web_search_tool for: {query}")
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        logger.error(f"web_search_tool failed: {str(e)}")
        return f"Error executing web search: {str(e)}"

@tool
def pdf_generator_tool(content: str) -> str:
    """Generates a PDF from text content and returns the filename."""
    logger.info("Executing pdf_generator_tool")
    try:
        filename = f"/tmp/notes_{uuid.uuid4().hex[:8]}.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        textobject = c.beginText(40, 750)
        
        # Simple text splitting to handle newlines
        lines = content.split('\n')
        for line in lines:
            textobject.textLine(line[:100]) # Quick wrap
        
        c.drawText(textobject)
        c.save()
        logger.info(f"PDF successfully generated at {filename}")
        return filename
    except Exception as e:
        logger.error(f"pdf_generator_tool failed: {str(e)}")
        return f"Error generating PDF: {str(e)}"
