# Dockerfile RunPod Serverless - Spinoza Inference (BERT + 3B + 7B)
# Base pytorch/pytorch : CUDA 12.1 pré-installé, build confirmé OK dans les logs
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HUGGINGFACE_HUB_CACHE=/app/cache
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch/CUDA déjà dans l'image de base — installe uniquement le reste
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p /app/cache && chmod -R 777 /app/cache

CMD ["python", "-u", "main.py"]
