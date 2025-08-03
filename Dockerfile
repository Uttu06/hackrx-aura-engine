# ==============================================================================
# Definitive Dockerfile - Version 5.0 (CPU Build, GPU Runtime)
# ==============================================================================

# Use a standard base image
FROM python:3.11-slim

# Set environment variables for best practices
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install required system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CRITICAL: Install the standard CPU version of torch first.
RUN pip install --no-cache-dir torch torchvision torchaudio

# Now install the rest of the application dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by 'unstructured'
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True);"

# Copy our application source code
COPY ./backend /app/backend

# Create a non-root user and switch to it for security
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# The models will be downloaded here when the container starts for the first time.
ENV HUGGING_FACE_HUB_CACHE=/home/appuser/.cache/huggingface

# Set the production environment variable
ENV APP_ENV=production

# Expose the port the application will run on
EXPOSE 8080

# The final command to start the application server
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "900", "-b", "0.0.0.0:8080", "backend.main:app"]