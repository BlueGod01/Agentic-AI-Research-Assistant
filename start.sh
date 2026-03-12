#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the foreground
# This ensures that frontend/app.py is the main running process and its logs are visible
echo "Starting Streamlit frontend (app.py)..."
streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
