FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ben \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# Exclude some MacOS specific packages if any sneaked in
RUN grep -v "appnope" requirements.txt > req_filtered.txt && \
    pip install --no-cache-dir -r req_filtered.txt

# Copy source
COPY . .

# Ensure cache directory exists and is writable
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

EXPOSE 8501

CMD ["streamlit", "run", "banglaocr/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
