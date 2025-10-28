ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

# Install additional dependencies for our customizations
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

# Create examples directory for runtime chat template downloads
RUN mkdir -p /app/examples

# Set environment variables for HuggingFace cache (if not already set)
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Override the entrypoint to use our wrapper script that handles model downloads
ENTRYPOINT ["/app/entrypoint.sh"]
