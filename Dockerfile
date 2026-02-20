# Use a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HUGGINGFACE_HUB_CACHE=/app/cache

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source code
COPY app/ .

# Bake models into the image
# Note: You need to provide HF_TOKEN if models are private
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python download_models.py

# Expose the port the app runs on
EXPOSE 7860

# Command to run the new entrypoint
CMD ["python", "-u", "main.py"]
