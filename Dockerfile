FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure start script is executable
RUN chmod +x start.sh

# Expose ports
EXPOSE 8000
EXPOSE 8501

# The user wants frontend/app.py to be the file running
# We use start.sh to launch the backend in background and frontend in foreground
CMD ["./start.sh"]
