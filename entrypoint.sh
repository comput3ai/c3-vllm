#!/bin/bash
set -e

# Optional startup delay (useful for multi-node coordination)
if [ -n "${STARTUP_DELAY}" ]; then
    echo "Sleeping for ${STARTUP_DELAY} seconds before startup"
    sleep "${STARTUP_DELAY}"
fi

# Handle empty HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    unset HF_TOKEN
fi

# Check if model needs to be downloaded
# For vLLM, we can check if the model exists in the download directory
if [ -n "${DOWNLOAD_DIR}" ] && [ -n "${MODEL_NAME}" ]; then
    # Use the served model name or extract from MODEL_NAME
    MODEL_DIR_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME##*/}}"
    MODEL_PATH="${DOWNLOAD_DIR}/${MODEL_DIR_NAME}"
    
    # Check if model exists and is complete
    if [ ! -d "${MODEL_PATH}" ] || [ -z "$(ls -A ${MODEL_PATH} 2>/dev/null)" ]; then
        echo "Model not found at ${MODEL_PATH}, downloading..."
        python3 /app/download.py \
            --repo "${MODEL_NAME}" \
            --api-name "${MODEL_DIR_NAME}" \
            --output-dir "${DOWNLOAD_DIR}"
    else
        echo "Model found at ${MODEL_PATH}, using local path"
    fi
    
    # Use the local path for vLLM instead of the HuggingFace repo name
    MODEL_ARG="${MODEL_PATH}"
else
    # If no download dir specified, use the HuggingFace repo name directly
    MODEL_ARG="${MODEL_NAME}"
fi

# Build the vLLM server command arguments
VLLM_ARGS=(
    "serve"  # vLLM subcommand
    "${MODEL_ARG}"
    "--host" "0.0.0.0"
    "--port" "8000"
)

# Add optional parameters only if they are set and non-empty
if [ -n "${SERVED_MODEL_NAME}" ]; then
    VLLM_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
fi

if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_ARGS+=("--max-model-len" "${MAX_MODEL_LEN}")
fi

if [ -n "${MAX_NUM_SEQS}" ]; then
    VLLM_ARGS+=("--max-num-seqs" "${MAX_NUM_SEQS}")
fi

if [ -n "${MAX_NUM_BATCHED_TOKENS}" ]; then
    VLLM_ARGS+=("--max-num-batched-tokens" "${MAX_NUM_BATCHED_TOKENS}")
fi

if [ -n "${GPU_MEMORY_UTILIZATION}" ]; then
    VLLM_ARGS+=("--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}")
fi

if [ -n "${DTYPE}" ]; then
    VLLM_ARGS+=("--dtype" "${DTYPE}")
fi

if [ -n "${KV_CACHE_DTYPE}" ]; then
    VLLM_ARGS+=("--kv-cache-dtype" "${KV_CACHE_DTYPE}")
fi

if [ -n "${QUANTIZATION}" ]; then
    VLLM_ARGS+=("--quantization" "${QUANTIZATION}")
fi

# Boolean flags - only add if explicitly set to true
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
    VLLM_ARGS+=("--trust-remote-code")
fi

if [ "${ENABLE_CHUNKED_PREFILL}" = "true" ]; then
    VLLM_ARGS+=("--enable-chunked-prefill")
fi

if [ "${ENFORCE_EAGER}" = "true" ]; then
    VLLM_ARGS+=("--enforce-eager")
fi

if [ "${DISABLE_SLIDING_WINDOW}" = "true" ]; then
    VLLM_ARGS+=("--disable-sliding-window")
fi

if [ "${DISABLE_LOG_STATS}" = "true" ]; then
    VLLM_ARGS+=("--disable-log-stats")
fi

# Tool calling support
if [ "${ENABLE_AUTO_TOOL_CHOICE}" = "true" ]; then
    VLLM_ARGS+=("--enable-auto-tool-choice")
fi

if [ -n "${TOOL_CALL_PARSER}" ]; then
    VLLM_ARGS+=("--tool-call-parser" "${TOOL_CALL_PARSER}")
fi

# Handle chat template - download if URL, otherwise use as path
if [ -n "${CHAT_TEMPLATE}" ]; then
    if [[ "${CHAT_TEMPLATE}" =~ ^https?:// ]]; then
        echo "Downloading chat template from URL: ${CHAT_TEMPLATE}"
        wget -O /app/examples/chat_template.jinja "${CHAT_TEMPLATE}"
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded chat template to /app/examples/chat_template.jinja"
            VLLM_ARGS+=("--chat-template" "/app/examples/chat_template.jinja")
        else
            echo "Failed to download chat template from URL, proceeding without custom template"
        fi
    else
        VLLM_ARGS+=("--chat-template" "${CHAT_TEMPLATE}")
    fi
fi

# API key if set
if [ -n "${API_KEY}" ]; then
    VLLM_ARGS+=("--api-key" "${API_KEY}")
fi

# Reasoning parser for thinking models
if [ -n "${REASONING_PARSER}" ]; then
    VLLM_ARGS+=("--reasoning-parser" "${REASONING_PARSER}")
fi

# Determine parallelization mode: Data Parallel vs Tensor Parallel
if [ -n "${DATA_PARALLEL_ADDRESS}" ]; then
    echo "=========================================="
    echo "DATA PARALLEL MODE"
    echo "=========================================="
    echo "Data parallel address: ${DATA_PARALLEL_ADDRESS}"

    # Data parallel configuration
    VLLM_ARGS+=("--data-parallel-address" "${DATA_PARALLEL_ADDRESS}")

    if [ -n "${DATA_PARALLEL_SIZE}" ]; then
        VLLM_ARGS+=("--data-parallel-size" "${DATA_PARALLEL_SIZE}")
    fi

    if [ -n "${DATA_PARALLEL_SIZE_LOCAL}" ]; then
        VLLM_ARGS+=("--data-parallel-size-local" "${DATA_PARALLEL_SIZE_LOCAL}")
    fi

    if [ "${ENABLE_EXPERT_PARALLEL}" = "true" ]; then
        VLLM_ARGS+=("--enable-expert-parallel")
    fi

    if [ -n "${DATA_PARALLEL_RPC_PORT}" ]; then
        VLLM_ARGS+=("--data-parallel-rpc-port" "${DATA_PARALLEL_RPC_PORT}")
    fi

    if [ "${HEADLESS}" = "true" ]; then
        VLLM_ARGS+=("--headless")
        echo "Running in headless mode (worker node)"
    fi

    if [ -n "${DATA_PARALLEL_START_RANK}" ]; then
        VLLM_ARGS+=("--data-parallel-start-rank" "${DATA_PARALLEL_START_RANK}")
        echo "Data parallel start rank: ${DATA_PARALLEL_START_RANK}"
    fi

elif [ -n "${NODE_MASTER_IP}" ] && [ -n "${NODE_IP}" ]; then
    echo "=========================================="
    echo "TENSOR PARALLEL MODE (Ray-based)"
    echo "=========================================="
    echo "This node IP: ${NODE_IP}"
    echo "Master node IP: ${NODE_MASTER_IP}"

    # Tensor parallel configuration
    if [ -n "${TENSOR_PARALLEL_SIZE}" ]; then
        VLLM_ARGS+=("--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE}")
    fi

    if [ -n "${DISTRIBUTED_EXECUTOR_BACKEND}" ]; then
        VLLM_ARGS+=("--distributed-executor-backend" "${DISTRIBUTED_EXECUTOR_BACKEND}")
    fi

    if [ -n "${PIPELINE_PARALLEL_SIZE}" ]; then
        VLLM_ARGS+=("--pipeline-parallel-size" "${PIPELINE_PARALLEL_SIZE}")
    fi

    if [ "${NODE_IP}" != "${NODE_MASTER_IP}" ]; then
        # Worker node configuration
        echo "Configuring as worker node"

        # Wait for master node to be ready
        echo "Waiting for master node at ${NODE_MASTER_IP} to be ready..."
        MASTER_READY=false
        RETRY_COUNT=0
        MAX_RETRIES=60  # 5 minutes with 5s intervals

        while [ "$MASTER_READY" = "false" ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if nc -z "${NODE_MASTER_IP}" 8000 2>/dev/null; then
                echo "Master node is ready!"
                MASTER_READY=true
            else
                echo "Master not ready yet, waiting... (attempt $((RETRY_COUNT + 1))/${MAX_RETRIES})"
                sleep 5
                RETRY_COUNT=$((RETRY_COUNT + 1))
            fi
        done

        if [ "$MASTER_READY" = "false" ]; then
            echo "Error: Master node did not become ready within timeout"
            exit 1
        fi
    else
        echo "Configuring as master node"
    fi

else
    echo "=========================================="
    echo "SINGLE NODE MODE"
    echo "=========================================="

    # Single node can still use tensor parallelism
    if [ -n "${TENSOR_PARALLEL_SIZE}" ]; then
        VLLM_ARGS+=("--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE}")
    fi

    if [ -n "${DISTRIBUTED_EXECUTOR_BACKEND}" ]; then
        VLLM_ARGS+=("--distributed-executor-backend" "${DISTRIBUTED_EXECUTOR_BACKEND}")
    fi

    if [ -n "${PIPELINE_PARALLEL_SIZE}" ]; then
        VLLM_ARGS+=("--pipeline-parallel-size" "${PIPELINE_PARALLEL_SIZE}")
    fi
fi

# Don't pass --download-dir when using a local model path

# Add any additional arguments passed to the container
VLLM_ARGS+=("$@")

# Execute vLLM server or custom command
if [ -n "${C3_VLLM_COMMAND}" ]; then
    echo "Custom command override: ${C3_VLLM_COMMAND}"
    exec bash -c "${C3_VLLM_COMMAND}"
else
    echo "Starting vLLM server with arguments: ${VLLM_ARGS[@]}"
    exec vllm "${VLLM_ARGS[@]}"
fi