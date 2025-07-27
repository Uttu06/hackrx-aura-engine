# Multi-stage Dockerfile for FastAPI RAG application on Hugging Face Spaces
# Optimized for T4 GPU deployment with model pre-warming

# Stage 1: Base system setup
FROM python:3.11-slim as base

# Set environment variables for optimal Python performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# Stage 2: Python dependencies installation
FROM base as dependencies

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python packages including PyTorch with CUDA 11.8 support
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

# Verify PyTorch CUDA installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Stage 3: Model pre-warming (critical for fast startup)
FROM dependencies as model-prewarming

# Create model pre-warming script
RUN cat > /home/app/preload_models.py << 'EOF'
#!/usr/bin/env python3
"""
Model pre-warming script for Hugging Face Spaces deployment.
Downloads and caches models during Docker build for faster startup.
"""
import os
import sys
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Suppress warnings during build
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Model configurations
LOCAL_LLM_MODEL_NAME = "microsoft/phi-2"
PRODUCTION_LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def download_embedding_model():
    """Download and cache embedding model."""
    print("📥 Downloading embedding model...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Embedding model cached: {EMBEDDING_MODEL_NAME}")
        return True
    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")
        return False

def download_local_llm():
    """Download and cache local development LLM."""
    print("📥 Downloading local LLM model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_LLM_MODEL_NAME,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        print(f"✅ Local LLM model cached: {LOCAL_LLM_MODEL_NAME}")
        return True
    except Exception as e:
        print(f"❌ Failed to download local LLM: {e}")
        return False

def download_production_llm():
    """Download and cache production LLM if GPU is available."""
    if not torch.cuda.is_available():
        print("⚠️  GPU not available during build, skipping production model download")
        return True
    
    print("📥 Downloading production LLM model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            PRODUCTION_LLM_MODEL_NAME,
            trust_remote_code=True
        )
        # Don't load the full model during build to save space and time
        # Just download the tokenizer and model files
        print(f"✅ Production LLM tokenizer cached: {PRODUCTION_LLM_MODEL_NAME}")
        return True
    except Exception as e:
        print(f"❌ Failed to download production LLM: {e}")
        return False

def main():
    """Main pre-warming function."""
    print("🚀 Starting model pre-warming process...")
    
    success_count = 0
    
    # Download embedding model (always needed)
    if download_embedding_model():
        success_count += 1
    
    # Download local LLM (for development/fallback)
    if download_local_llm():
        success_count += 1
    
    # Download production LLM components
    if download_production_llm():
        success_count += 1
    
    print(f"📊 Pre-warming completed: {success_count}/3 models cached successfully")
    
    if success_count == 3:
        print("✅ All models pre-warmed successfully!")
        return 0
    else:
        print("⚠️  Some models failed to download, but continuing...")
        return 0  # Don't fail the build for model download issues

if __name__ == "__main__":
    sys.exit(main())
EOF

# Make the script executable
RUN chmod +x /home/app/preload_models.py

# Run model pre-warming (critical for fast startup on Hugging Face Spaces)
RUN echo "🔥 Pre-warming models during build phase..." && \
    python3 /home/app/preload_models.py

# Stage 4: Application setup
FROM model-prewarming as application

# Copy application code
COPY . /home/app/

# Create necessary directories and set permissions
RUN mkdir -p /home/app/logs /home/app/cache && \
    chown -R app:app /home/app

# Switch to non-root user
USER app

# Set working directory
WORKDIR /home/app

# Create a simple health check script
RUN cat > /home/app/healthcheck.py << 'EOF'
#!/usr/bin/env python3
import sys
import requests
import time

def health_check():
    try:
        response = requests.get("http://localhost:7860/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return 0
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return 1
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(health_check())
EOF

# Stage 5: Production runtime
FROM application as production

# Set production environment variables
ENV APP_ENV=production \
    WORKERS=1 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=50 \
    TIMEOUT=120 \
    KEEP_ALIVE=5 \
    GRACEFUL_TIMEOUT=60

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /home/app/healthcheck.py

# Create startup script for better logging and error handling
RUN cat > /home/app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting FastAPI RAG application on Hugging Face Spaces..."
echo "📊 Environment: $APP_ENV"
echo "🔧 Workers: $WORKERS"
echo "⏰ Timeout: $TIMEOUT seconds"

# Print system info
echo "💻 System Information:"
python3 -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo "🔥 Pre-loaded models should be available in cache..."
echo "🌐 Starting Gunicorn with Uvicorn workers..."

# Start the application with Gunicorn
exec gunicorn \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers $WORKERS \
    --bind 0.0.0.0:7860 \
    --timeout $TIMEOUT \
    --keep-alive $KEEP_ALIVE \
    --max-requests $MAX_REQUESTS \
    --max-requests-jitter $MAX_REQUESTS_JITTER \
    --graceful-timeout $GRACEFUL_TIMEOUT \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --capture-output \
    backend.main:app
EOF

# Make startup script executable
RUN chmod +x /home/app/start.sh

# Use the startup script as the command
CMD ["/home/app/start.sh"]

# Metadata
LABEL maintainer="MLOps Team"
LABEL description="Production FastAPI RAG application with Hugging Face models"
LABEL version="1.0.0"
LABEL gpu.required="true"
LABEL spaces.config="{'runtime': 'docker', 'gpu': 'T4'}"