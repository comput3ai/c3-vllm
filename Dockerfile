# Use official vLLM OpenAI image as base
# This image already includes vLLM, PyTorch, CUDA 12.8 support, and all dependencies
FROM vllm/vllm-openai:latest

# The base image already includes:
# - Python 3.12
# - PyTorch with CUDA 12.8 support
# - vLLM with FP8 quantization support
# - Triton
# - FlashInfer
# - OpenAI API server entrypoint
# - accelerate, hf_transfer, modelscope, bitsandbytes, timm, boto3

# Install additional dependencies for our customizations
# Most of these are already in the base image, but we ensure they're present
RUN pip install --no-cache-dir \
    huggingface_hub \
    python-dotenv \
    requests \
    blobfile \
    tiktoken

# Copy our custom download script and entrypoint wrapper
COPY download.py /app/download.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/download.py /app/entrypoint.sh

# Set environment variables for HuggingFace cache (if not already set)
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# The base image already sets VLLM_USAGE_SOURCE=production-docker-image
# and uses /vllm-workspace as WORKDIR

# Override the entrypoint to use our wrapper script that handles model downloads
ENTRYPOINT ["/app/entrypoint.sh"]