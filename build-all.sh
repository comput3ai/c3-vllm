#!/bin/bash
set -euo pipefail

echo "🔨 Building all c3-vllm image variants..."
echo ""

# Step 0: Clone vllm if not present
if [ ! -d "vllm/.git" ]; then
    echo "📥 vllm not found, cloning from GitHub..."
    rm -rf vllm
    git clone https://github.com/vllm-project/vllm.git
    echo "✅ vllm cloned"
    echo ""
fi

# Step 1: Build upstream vLLM from git
echo "📦 Step 1/4: Building upstream vLLM from source → vllm:git"
echo "   (this will take a while...)"
cd vllm
docker build \
    -t vllm:git \
    -f docker/Dockerfile \
    .
cd ..
echo "✅ vllm:git complete"
echo ""

# Step 2: Apply MiniMax triton-kernels patch
echo "🔧 Step 2/4: Applying MiniMax patch → vllm:minimax"
docker build \
    -t vllm:minimax \
    --build-arg BASE_IMAGE=vllm:git \
    -f Dockerfile.minimax \
    .
echo "✅ vllm:minimax complete"
echo ""

# Step 3: Build all c3-vllm variants
echo "🎨 Step 3/4: Building c3-vllm variants..."
echo ""

# c3-vllm:latest (based on upstream latest)
echo "   Building c3-vllm:latest (vllm/vllm-openai:latest + C3)"
docker build \
    -t c3-vllm:latest \
    --build-arg BASE_IMAGE=vllm/vllm-openai:latest \
    -f Dockerfile \
    .
echo "   ✅ c3-vllm:latest complete"
echo ""

# c3-vllm:v0 (based on v0.10.2 engine)
echo "   Building c3-vllm:v0 (vllm/vllm-openai:v0.10.2 + C3)"
docker build \
    -t c3-vllm:v0 \
    --build-arg BASE_IMAGE=vllm/vllm-openai:v0.10.2 \
    -f Dockerfile \
    .
echo "   ✅ c3-vllm:v0 complete"
echo ""

# c3-vllm:minimax (based on our custom build with M2 support)
echo "   Building c3-vllm:minimax (vllm:minimax + C3)"
docker build \
    -t c3-vllm:minimax \
    --build-arg BASE_IMAGE=vllm:minimax \
    -f Dockerfile \
    .
echo "   ✅ c3-vllm:minimax complete"
echo ""

echo "✨ All builds complete!"
echo ""
echo "🚀 Images ready:"
echo "   - vllm:git (clean upstream from git)"
echo "   - vllm:minimax (upstream + triton-kernels patch)"
echo "   - c3-vllm:latest (upstream latest + C3 customizations)"
echo "   - c3-vllm:v0 (v0.10.2 engine + C3 customizations)"
echo "   - c3-vllm:minimax (vllm:minimax + C3 customizations)"
echo ""
echo "To verify MiniMax M2 support:"
echo "  docker run --rm --entrypoint ls c3-vllm:minimax /usr/local/lib/python3.12/dist-packages/vllm/reasoning/ | grep minimax"
