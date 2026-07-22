FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and necessary system libraries (e.g. for OpenCV)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
# Install PyTorch with CUDA 12.1 support first, then other requirements
RUN python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Set environment variable for llama-cpp-python to compile with CUDA support
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1
RUN python3.11 -m pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
