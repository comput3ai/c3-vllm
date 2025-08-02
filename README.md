# 🚀 c3-vllm

A Docker container that seamlessly runs [vLLM](https://github.com/vllm-project/vllm) with automatic HuggingFace model downloads. Deploy any LLM with blazing-fast inference using a single command!

## 🌟 Features

- **High-Performance Inference**: Built on vLLM for production-grade serving
- **Automatic Model Downloads**: Downloads models from HuggingFace at runtime
- **OpenAI Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Advanced Features**: Tool calling, FP8 quantization, tensor parallelism
- **Flexible Configuration**: Environment variables for easy customization
- **Production Ready**: Health checks, metrics, and Traefik integration

## 🙏 Credits

This project builds upon the excellent work of:
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference engine
- The HuggingFace community for hosting and sharing models

## 📦 Quick Start

### Pull the image

```bash
docker pull ghcr.io/comput3ai/c3-vllm:latest
```

### Run with your model

```bash
docker run --gpus all \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e SERVED_MODEL_NAME=llama-3.1-8b \
  -e TENSOR_PARALLEL_SIZE=1 \
  -p 8080:8000 \
  ghcr.io/comput3ai/c3-vllm:latest
```

## 🔧 Configuration

### Core Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model repository | Required |
| `SERVED_MODEL_NAME` | Model name in API responses | - |
| `TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism | `1` |
| `MAX_MODEL_LEN` | Maximum context length (e.g., 128000 for Kimi K2) | Auto-detected |
| `DTYPE` | Model data type (auto, float16, bfloat16, float32) | `auto` |
| `TRUST_REMOTE_CODE` | Allow remote code execution (required for some models) | `false` |

### Performance Tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `GPU_MEMORY_UTILIZATION` | Fraction of GPU memory to use (0.0-1.0) | `0.9` |
| `MAX_NUM_SEQS` | Maximum sequences to process concurrently | `256` |
| `MAX_NUM_BATCHED_TOKENS` | Maximum tokens per iteration | Auto |
| `KV_CACHE_DTYPE` | KV cache data type (auto, fp8, fp8_e4m3, fp8_e5m2) | `auto` |
| `ENABLE_CHUNKED_PREFILL` | Enable chunked prefill for long prompts | `false` |
| `ENFORCE_EAGER` | Disable CUDA graphs (may help with large models) | `false` |
| `DISABLE_SLIDING_WINDOW` | Disable sliding window attention | `false` |
| `VLLM_USE_V1` | Use vLLM V1 engine (set to 0 for V0 engine) | `1` |

### Tool Calling Support

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_AUTO_TOOL_CHOICE` | Enable automatic tool choice | `false` |
| `TOOL_CALL_PARSER` | Tool call parser to use (e.g., kimi_k2, hermes) | - |

### API Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Internal server port | `8000` |
| `API_KEY` | API key for authentication | - |
| `HF_TOKEN` | HuggingFace token for gated models | - |
| `DOWNLOAD_DIR` | Directory to download/cache models | `/models` |
| `DISABLE_LOG_STATS` | Disable logging statistics | `false` |

## 📚 Examples

### Running Different Models

#### Small Model (7B-13B) on Single GPU

```bash
docker run --gpus all \
  -e MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct \
  -e SERVED_MODEL_NAME=llama-3.2-3b \
  -p 8080:8000 \
  ghcr.io/comput3ai/c3-vllm:latest
```

#### Large Model with Tensor Parallelism

```bash
docker run --gpus all \
  -e MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct \
  -e SERVED_MODEL_NAME=llama-3.1-70b \
  -e TENSOR_PARALLEL_SIZE=4 \
  -e MAX_MODEL_LEN=8192 \
  -p 8080:8000 \
  ghcr.io/comput3ai/c3-vllm:latest
```

#### Model with FP8 KV Cache

```bash
docker run --gpus all \
  -e MODEL_NAME=moonshotai/Kimi-K2-Instruct \
  -e SERVED_MODEL_NAME=kimi-k2 \
  -e TENSOR_PARALLEL_SIZE=8 \
  -e KV_CACHE_DTYPE=fp8 \
  -e MAX_MODEL_LEN=65536 \
  -p 8080:8000 \
  ghcr.io/comput3ai/c3-vllm:latest
```

### API Authentication

To secure your vLLM API with authentication:

1. Set the `API_KEY` environment variable in your `.env` file:
```env
API_KEY=your-secret-api-key-here
```

2. All API requests must include the API key in the Authorization header:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{...}'
```

**Note**: Without the `API_KEY` environment variable set, the API will accept all requests without authentication.

### Using the API

#### With curl

```bash
# Chat completion (no auth)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "temperature": 0.7
  }'

# Chat completion (with auth)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ]
  }'

# Streaming response
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding"}
    ],
    "stream": true
  }'

# Check model info
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-api-key-here"

# Health check (usually doesn't require auth)
curl http://localhost:8000/health
```

#### With Python

```python
from openai import OpenAI

# Without authentication
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Required by client but ignored by server
)

# With authentication
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-api-key-here"  # Must match API_KEY env var
)

response = client.chat.completions.create(
    model="llama-3.1-8b",  # Uses the SERVED_MODEL_NAME
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

## 🐳 Deployment Options

### Option 1: Docker Run (Simple)

```bash
docker run -t --gpus all \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e SERVED_MODEL_NAME=llama-3.1-8b \
  -p 8080:8000 \
  ghcr.io/comput3ai/c3-vllm:latest
```

### Option 2: Docker Compose (Recommended)

1. Copy the environment template:
```bash
cp env.sample .env
```

2. Edit `.env` with your model configuration:
```env
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
SERVED_MODEL_NAME=llama-3.1-8b
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=16384

# Optional but recommended: Enable API authentication
# API_KEY=your-secret-api-key-here

# Optional: For private/gated models
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. Start the service:
```bash
docker-compose up -d
```

### Option 3: Production Deployment

For production deployments with SSL/TLS and optimized storage:

```bash
docker-compose -f docker-compose.prod.yaml up -d
```

This uses:
- NVMe storage for models (`/opt/dlami/nvme/vllm-models`)
- Traefik for SSL termination
- Health checks and monitoring

## 🔨 Building Locally

```bash
git clone https://github.com/comput3ai/c3-vllm
cd c3-vllm
docker build -t c3-vllm .
```

## 🛠️ How It Works

1. The container starts and checks for the model in the download directory
2. If not present, it downloads the model from HuggingFace
3. vLLM server starts with optimized settings for your hardware
4. OpenAI-compatible API endpoints become available

## 📊 Performance Tips

### General Recommendations

- **Tensor Parallelism**: Use `TENSOR_PARALLEL_SIZE` equal to your GPU count for large models
- **KV Cache**: Enable `KV_CACHE_DTYPE=fp8` for 2x memory efficiency with minimal quality loss
- **Context Length**: Set `MAX_MODEL_LEN` based on your GPU memory
- **Batch Size**: Tune `MAX_NUM_BATCHED_TOKENS` for your workload

### Large MoE Models (e.g., Kimi K2)

For models with 1T+ parameters and 128k context support:

```env
# Recommended settings for 8xB200 or 8xH100
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=65536          # Start conservatively, increase as needed
MAX_NUM_SEQS=256             # High concurrency
MAX_NUM_BATCHED_TOKENS=32768 # Large batch for throughput
GPU_MEMORY_UTILIZATION=0.90  # Can push higher on datacenter GPUs
KV_CACHE_DTYPE=fp8           # Essential for large contexts
ENABLE_CHUNKED_PREFILL=true  # Helps with long prompts
TRUST_REMOTE_CODE=true       # Required for custom architectures
# VLLM_USE_V1=0              # Uncomment for B200 GPUs if FP8 KV cache fails
```

**Note**: If you encounter "TritonMLA V1 with FP8 KV cache not yet supported" errors on Blackwell (B200) GPUs, set `VLLM_USE_V1=0` to use the V0 engine which has better compatibility.

### Memory Estimation

For FP8 models with FP8 KV cache:
- Model weights: ~1.2TB for 1T parameter model
- KV cache per token: ~0.5MB per sequence
- Total VRAM needed: Model weights + (context_length × batch_size × 0.5MB)

## 📄 License

BSD 3-Clause License

Copyright (c) 2025, Comput3.ai

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.