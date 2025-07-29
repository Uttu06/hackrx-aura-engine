# ==============================================================================
# Definitive Multi-stage Dockerfile for FastAPI RAG application
# Version: 4.0 (Lean Build, Late Model Initialization)
# ==============================================================================

# Stage 1: Build Environment - Just to compile wheels efficiently
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /wheelhouse

COPY requirements.txt .

# Pre-compile PyTorch and Torchvision for a specific CUDA version first
RUN pip wheel --no-cache-dir . --wheel-dir=/wheelhouse \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
# Then compile the rest of the dependencies
RUN pip wheel --no-cache-dir . --wheel-dir=/wheelhouse -r requirements.txt

# Stage 2: Final Production Image
FROM python:3.11-slim AS production

RUN apt-get update && apt-get install -y git poppler-utils && rm -rf /var/lib/apt/lists/*

# Copy the pre-compiled wheels from our builder stage
COPY --from=builder /wheelhouse /wheelhouse

# Install everything from the fast, local wheelhouse. This is much faster.
RUN pip install --no-cache-dir --no-index --find-links=/wheelhouse /wheelhouse/*.whl

# We still install NLTK data during the build
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app
USER app

ENV APP_ENV=production
ENV HUGGING_FACE_HUB_CACHE=/home/app/.cache/huggingface

# Copy our application code last
COPY --chown=app:app ./backend /home/app/backend

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300", "-b", "0.0.0.0:8080", "backend.main:app"]