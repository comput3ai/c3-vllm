#!/bin/bash
set -euo pipefail

# Parse command line arguments
BUILD_FROM_GIT=false
if [ "${1:-}" = "--git" ]; then
    BUILD_FROM_GIT=true
    shift
fi

# Detect CPU thread count for optimal build parallelism
CPU_THREADS=$(nproc)
NVCC_THREADS=$((CPU_THREADS / 4))
# Ensure nvcc_threads is at least 2
if [ ${NVCC_THREADS} -lt 2 ]; then
    NVCC_THREADS=2
fi

echo "🔨 Building all c3-vllm image variants..."
echo "💻 Detected ${CPU_THREADS} CPU threads - will use max_jobs=${CPU_THREADS}, nvcc_threads=${NVCC_THREADS}"
if [ "$BUILD_FROM_GIT" = true ]; then
    echo "🐙 Git mode: Building vLLM from source"
else
    echo "📦 Registry mode: Using pre-built vllm/vllm-openai:nightly"
fi
echo ""

# Step 1: Get or build vllm/vllm-openai:nightly
if [ "$BUILD_FROM_GIT" = true ]; then
    # Clone vllm if not present
    if [ ! -d "vllm/.git" ]; then
        echo "📥 vllm not found, cloning from GitHub..."
        rm -rf vllm
        git clone https://github.com/vllm-project/vllm.git
        echo "✅ vllm cloned"
        echo ""
    fi

    # Build upstream vLLM from git
    echo "📦 Step 1/4: Building upstream vLLM from source → vllm/vllm-openai:nightly"
    echo "   Using max_jobs=${CPU_THREADS}, nvcc_threads=${NVCC_THREADS}"
    echo "   (this will take a while...)"
    cd vllm
    DOCKER_BUILDKIT=1 docker build \
        --target vllm-openai \
        -t vllm/vllm-openai:nightly \
        -f docker/Dockerfile \
        --build-arg max_jobs=${CPU_THREADS} \
        --build-arg nvcc_threads=${NVCC_THREADS} \
        --build-arg RUN_WHEEL_CHECK=false \
        .
    cd ..
    echo "✅ vllm/vllm-openai:nightly complete"
    echo ""
else
    echo "📥 Step 1/4: Pulling vllm/vllm-openai:nightly from registry"
    docker pull vllm/vllm-openai:nightly
    echo "✅ vllm/vllm-openai:nightly pulled"
    echo ""
fi

# Step 2: Build all c3-vllm variants
echo "🎨 Step 2/4: Building c3-vllm variants..."
echo ""

# c3-vllm:latest (based on upstream latest)
echo "   Building c3-vllm:latest (vllm/vllm-openai:latest + C3)"
docker pull vllm/vllm-openai:latest
docker build \
    -t c3-vllm:latest \
    --build-arg BASE_IMAGE=vllm/vllm-openai:latest \
    -f Dockerfile \
    .
echo "   ✅ c3-vllm:latest complete"
echo ""

# c3-vllm:v0 (based on v0.10.2 engine)
echo "   Building c3-vllm:v0 (vllm/vllm-openai:v0.10.2 + C3)"
docker pull vllm/vllm-openai:v0.10.2
docker build \
    -t c3-vllm:v0 \
    --build-arg BASE_IMAGE=vllm/vllm-openai:v0.10.2 \
    -f Dockerfile \
    .
echo "   ✅ c3-vllm:v0 complete"
echo ""

# c3-vllm:nightly (based on nightly)
echo "   Building c3-vllm:nightly (vllm/vllm-openai:nightly + C3)"
docker build \
    -t c3-vllm:nightly \
    --build-arg BASE_IMAGE=vllm/vllm-openai:nightly \
    -f Dockerfile \
    .
echo "   ✅ c3-vllm:nightly complete"
echo ""

# Step 3: Build c3-vllm:minimax with triton-kernels patch
echo "🔧 Step 3/4: Building c3-vllm:minimax (c3-vllm:nightly + triton-kernels)"
docker build \
    -t c3-vllm:minimax \
    --build-arg BASE_IMAGE=c3-vllm:nightly \
    -f Dockerfile.minimax \
    .
echo "✅ c3-vllm:minimax complete"
echo ""

echo "✨ All builds complete!"
echo ""
echo "🚀 Images ready:"
echo "   - c3-vllm:latest (upstream latest + C3 customizations)"
echo "   - c3-vllm:v0 (v0.10.2 engine + C3 customizations)"
echo "   - c3-vllm:nightly (nightly + C3 customizations)"
echo "   - c3-vllm:minimax (nightly + C3 + triton-kernels patch)"
echo ""
echo "To verify MiniMax M2 support:"
echo "  docker run --rm --entrypoint ls c3-vllm:minimax /usr/local/lib/python3.12/dist-packages/vllm/reasoning/ | grep minimax"
