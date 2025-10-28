#!/bin/bash
set -euo pipefail

# Three-stage build for vLLM with MiniMax M2 support
# Stage 1: vllm:git (clean upstream)
# Stage 2: vllm:minimax-base (with triton-kernels patch)
# Stage 3: c3-vllm:minimax (with our customizations)

echo "🔨 Building vLLM with MiniMax M2 support..."
echo ""

# Stage 1: Build upstream vLLM from submodule
echo "📦 Stage 1/3: Building upstream vLLM from source → vllm:git"
echo "   (this will take a while...)"
docker build \
    -t vllm:git \
    -f vllm/docker/Dockerfile \
    vllm/

echo ""
echo "✅ Stage 1 complete: vllm:git"
echo ""

# Stage 2: Apply MiniMax triton-kernels patch
echo "🔧 Stage 2/3: Applying triton-kernels patch → vllm:minimax-base"
docker build \
    -t vllm:minimax-base \
    --build-arg BASE_IMAGE=vllm:git \
    -f Dockerfile.minimax-patch \
    .

echo ""
echo "✅ Stage 2 complete: vllm:minimax-base"
echo ""

# Stage 3: Add C3 customizations
echo "🎨 Stage 3/3: Adding C3 customizations → c3-vllm:minimax"
docker build \
    -t c3-vllm:minimax \
    --build-arg BASE_IMAGE=vllm:minimax-base \
    -f Dockerfile.minimax \
    .

echo ""
echo "✅ Build complete!"
echo ""
echo "🚀 Images ready:"
echo "   - vllm:git (clean upstream)"
echo "   - vllm:minimax-base (with triton-kernels patch)"
echo "   - c3-vllm:minimax (with C3 customizations)"
echo ""
echo "To verify MiniMax M2 support:"
echo "  docker run --rm --entrypoint ls c3-vllm:minimax /usr/local/lib/python3.12/dist-packages/vllm/reasoning/ | grep minimax"
