FROM python:3.9-slim

# Metadata labels
LABEL org.opencontainers.image.base.name="python:3.9-slim"
LABEL org.opencontainers.image.architecture="amd64"
LABEL org.opencontainers.image.operating-system="linux"

# Install minimal system dependencies and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements for caching
COPY requirements.txt .

# Upgrade pip and install CPU-only PyTorch, torchvision, and dependencies with retries
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
         --retries 3 --timeout 60 \
         torch==2.2.2+cpu torchvision==0.17.2+cpu \
    && pip install --no-cache-dir --retries 3 --timeout 60 -r requirements.txt

# Copy and run the pre-caching script with offline mode disabled
COPY precache_models.py .
RUN TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python precache_models.py

# Set environment variables for offline mode in the final image
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=""

# Copy application code and create I/O directories
COPY main.py .
RUN mkdir -p /input/PDFs /output

# Set entrypoint
CMD ["python", "main.py"]