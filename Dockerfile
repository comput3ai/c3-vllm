FROM vllm/vllm-openai:latest

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

# Set entrypoint to our wrapper script
ENTRYPOINT ["/app/entrypoint.sh"]