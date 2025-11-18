#!/bin/bash
# Docker startup script for vLLM server with GLM-4.5-Air-AWQ-4bit
# Uses NVIDIA's pre-built vLLM container for DGX Spark / GB10 GPU

set -e

usage() {
    cat <<EOF
Usage: ./start_vllm_docker.sh [options]

Options:
  --port <port>           Host/container port to expose (default: 8000 or VLLM_PORT)
  --host <host>           Host interface to bind (default: 0.0.0.0 or VLLM_HOST)
  --model-path <path>     Model directory to mount (default: ./models/GLM-4.5-Air-AWQ-4bit)
  --tensor-parallel <n>   Tensor parallel size (default: 1)
  --max-model-len <n>     Maximum model context length (default: 8192)
  --gpu-mem <fraction>    GPU memory utilization fraction (default: 0.9)
  --container-name <name> Docker container name (default: vllm-server)
  --image <image>         Container image (default: nvcr.io/nvidia/vllm:25.09-py3)
  -h, --help              Show this help message

Environment overrides:
  VLLM_PORT, VLLM_HOST, VLLM_MODEL_PATH, VLLM_CONTAINER_NAME, VLLM_IMAGE,
  VLLM_TP_SIZE, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTIL

Tip: Keep VLLM_PORT in sync with Terrarium by exporting
     VLLM_URL=http://localhost:<port> before starting server.py.
EOF
}

# Configuration (with env overrides)
MODEL_PATH="${VLLM_MODEL_PATH:-./models/GLM-4.5-Air-AWQ-4bit}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-server}"
IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.09-py3}"

# vLLM parameters for GLM-4.5-Air-AWQ-4bit
DTYPE="float16"  # Required for AWQ quantization
TENSOR_PARALLEL_SIZE="${VLLM_TP_SIZE:-1}"  # Adjust based on your GPU count (1, 2, 4, or 8)
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.9}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Port must be a number (received '$PORT')"
    exit 1
fi

echo "========================================"
echo "Starting vLLM Server (Docker)"
echo "========================================"
echo "Image: $IMAGE"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "========================================"
echo

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please download the model first using:"
    echo "  python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"cpatonn/GLM-4.5-Air-AWQ-4bit\", local_dir=\"$MODEL_PATH\")'"
    exit 1
fi

# Check if model download is complete
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Model appears incomplete (config.json missing)"
    echo "Please complete the model download"
    exit 1
fi

# Stop and remove existing container if running
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo "Existing container removed"
    echo
fi

echo "Starting vLLM server in Docker..."
echo "(This may take a few minutes to load the model)"
echo "Check logs with: docker logs -f $CONTAINER_NAME"
echo

# Start vLLM server in Docker
# Note: GLM-4.5-Air supports tool calling and reasoning modes
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "$PORT:$PORT" \
    -v "$(pwd)/models:/models" \
    "$IMAGE" \
    vllm serve "/models/GLM-4.5-Air-AWQ-4bit" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"

echo
echo "✓ Container started successfully!"
echo
echo "Useful commands:"
echo "  - Check logs:   docker logs -f $CONTAINER_NAME"
echo "  - Stop server:  docker stop $CONTAINER_NAME"
echo "  - Remove:       docker rm $CONTAINER_NAME"
echo "  - Health check: curl http://localhost:$PORT/health"
echo "========================================"

# Notes:
# - --gpus all: Enable GPU access
# - --ipc=host: Recommended by NVIDIA for vLLM (shared memory)
# - --ulimit memlock=-1: Unlimited locked memory
# - --ulimit stack=67108864: Increased stack size (64MB)
# - --tool-call-parser glm45: Enables GLM-4.5 tool calling
# - --reasoning-parser glm45: Enables GLM-4.5 reasoning mode
# - --enable-auto-tool-choice: Automatically selects tools
# - --max-model-len: Context length (adjust based on GPU memory)
# - --gpu-memory-utilization: Fraction of GPU memory to use (0.9 = 90%)
#
# Adjust tensor-parallel-size based on your GPU setup:
#   - 1 GPU: --tensor-parallel-size 1
#   - 2 GPUs: --tensor-parallel-size 2
#   - 4 GPUs: --tensor-parallel-size 4
