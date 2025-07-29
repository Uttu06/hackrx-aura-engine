# ==============================================================================
# Definitive Multi-stage Dockerfile for FastAPI RAG application
# Version: 3.0 (Robust Multi-Stage Wheel Build)
# ==============================================================================

# Stage 1: Build Environment - Install build tools and compile wheels
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=0

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Create a directory to store our compiled wheels
RUN mkdir -p /wheelhouse

# First, install the most complex, hard-to-build dependencies (PyTorch)
# and build them into wheels.
RUN pip wheel --no-cache-dir --wheel-dir=/wheelhouse \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118
    
# Now, install the other dependencies using the pre-built PyTorch wheels
# This makes the dependency resolution much faster.
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheelhouse \
    --find-links=/wheelhouse \
    -r requirements.txt

# --- The final build stage starts here ---

# Stage 2: Final Production Image
FROM python:3.11-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive
    
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the pre-compiled wheels from our builder stage
COPY --from=builder /wheelhouse /wheelhouse

# Install everything from the fast, local wheelhouse
RUN pip install --no-cache-dir --no-index --find-links=/wheelhouse /wheelhouse/*.whl

RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app
USER app

# This ENV var now MUST be set to tell Transformers where to save models
# It points to a directory outside the main code.
ENV HUGGING_FACE_HUB_CACHE=/home/app/.cache/huggingface

# Copy application code AFTER all installations are done
COPY --chown=app:app ./backend /home/app/backend

# Pre-warm the models by running the script
# We copy the script and then run it.
COPY --chown=app:app preload_models.py .
RUN python3 preload_models.py

ENV APP_ENV=production

EXPOSE 8080

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "--timeout", "300", "backend.main:app"]