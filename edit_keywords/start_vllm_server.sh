#!/bin/bash
# Start vLLM server for Qwen3-VL-30B
#
# Usage:
#   ./start_vllm_server.sh                    # Auto-detect GPUs
#   ./start_vllm_server.sh 4                  # Use 4 GPUs
#   ./start_vllm_server.sh 4 8080             # Use 4 GPUs on port 8080
#
# Reference: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html

set -e

# Configuration
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
TP_SIZE="${1:-$(nvidia-smi -L | wc -l)}"  # Default: all available GPUs
PORT="${2:-8000}"
GPU_MEMORY_UTILIZATION="${3:-0.9}"

echo "=============================================="
echo "Starting vLLM Server for Qwen3-VL-30B"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Tensor Parallel Size: ${TP_SIZE} GPUs"
echo "Port: ${PORT}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "=============================================="
echo ""

# Start server
vllm serve "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --mm-encoder-tp-mode data \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "${PORT}"

