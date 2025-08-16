# Use vLLM's official Dockerfile approach with CUDA 12.8
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

# Install Python 3.12 (matching vLLM's default)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install PyTorch and dependencies with CUDA 12.8 support first
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Install Triton with CUDA 12.8 support
RUN pip install triton --extra-index-url https://download.pytorch.org/whl/cu128

# Install vLLM with CUDA 12.8 support
# Note: vLLM includes FP8 quantization support for models like Kimi-K2
RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

# Install bitsandbytes
RUN pip install bitsandbytes --extra-index-url https://download.pytorch.org/whl/cu128

# Install additional dependencies required by Kimi-K2 tokenizer
RUN pip install --no-cache-dir blobfile tiktoken

# Install Python dependencies for download script
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy download script and entrypoint
COPY download.py /app/download.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/download.py /app/entrypoint.sh

# Set environment variables for HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Workaround for triton issues (from vLLM's Dockerfile)
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Set vLLM usage source
ENV VLLM_USAGE_SOURCE production-docker-image

# Set workdir
WORKDIR /vllm-workspace

# Set entrypoint to our wrapper script
ENTRYPOINT ["/app/entrypoint.sh"]