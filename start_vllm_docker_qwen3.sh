#!/bin/bash
# Docker startup script for vLLM server with Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
# Uses NVIDIA's pre-built vLLM container for DGX Spark / GB10 GPU

set -e

usage() {
    cat <<EOF
Usage: ./start_vllm_docker_qwen3.sh [options]

Options:
  --port <port>           Host/container port to expose (default: 8000 or VLLM_PORT)
  --host <host>           Host interface to bind (default: 0.0.0.0 or VLLM_HOST)
  --model-path <path>     Model directory to mount (default: ./models/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit)
  --tensor-parallel <n>   Tensor parallel size (default: 1)
  --max-model-len <n>     Maximum model context length (default: 32768 or VLLM_MAX_MODEL_LEN)
  --gpu-mem <fraction>    GPU memory utilization fraction (default: auto from num-agents or VLLM_GPU_MEMORY_UTIL)
  --num-agents <n>        Target concurrent agents/contexts to size KV pool (default: 1 or VLLM_NUM_AGENTS)
  --dtype <dtype>         Model dtype (default: bfloat16 or VLLM_DTYPE)
  --tool-parser <parser>  Tool call parser name (default: qwen3_coder)
  --reasoning-parser <p>  Reasoning parser name (optional; omit if unsupported)
  --trust-remote-code     Enable transformers trust_remote_code (default: off)
  --enforce-eager         Disable torch.compile and run eager mode
  --container-name <name> Docker container name (default: vllm-server-qwen3)
  --image <image>         Container image (default: nvcr.io/nvidia/vllm:25.12-py3)
  -h, --help              Show this help message

Environment overrides:
  VLLM_PORT, VLLM_HOST, VLLM_MODEL_PATH, VLLM_CONTAINER_NAME, VLLM_IMAGE,
  VLLM_TP_SIZE, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTIL, VLLM_NUM_AGENTS, VLLM_DTYPE

Tip: Keep VLLM_PORT in sync with Terrarium by exporting
     VLLM_URL=http://localhost:<port> before starting server.py.
EOF
}

# Configuration (with env overrides)
MODEL_PATH="${VLLM_MODEL_PATH:-./models/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-server-qwen3}"
IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.12-py3}"

# vLLM parameters for Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
DTYPE="${VLLM_DTYPE:-bfloat16}"  # Qwen3-Next defaults to bf16; set to "auto" or "float16" if needed
TENSOR_PARALLEL_SIZE="${VLLM_TP_SIZE:-1}"  # Adjust based on your GPU count
# Default context/util geared for long single requests; adjust if you need more concurrency.
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-}"
NUM_AGENTS="${VLLM_NUM_AGENTS:-1}"
TOOL_PARSER="${VLLM_TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-false}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"

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
        --num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --tool-parser)
            TOOL_PARSER="$2"
            shift 2
            ;;
        --reasoning-parser)
            REASONING_PARSER="$2"
            shift 2
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE="true"
            shift 1
            ;;
        --enforce-eager)
            ENFORCE_EAGER="true"
            shift 1
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

# Resolve model path after overrides (fallback to input path if realpath fails)
if HOST_MODEL_DIR="$(realpath "$MODEL_PATH" 2>/dev/null)"; then
    :
else
    HOST_MODEL_DIR="$MODEL_PATH"
fi
MODEL_PARENT_DIR="$(dirname "$HOST_MODEL_DIR")"
MODEL_BASENAME="$(basename "$HOST_MODEL_DIR")"
CONTAINER_MODEL_PATH="/model_mount/$MODEL_BASENAME"

# Auto-compute GPU memory utilization from num_agents + max_model_len if not provided
if [ -z "$GPU_MEMORY_UTIL" ]; then
    # Calibration: at util=0.7, we observed ~398,752 tokens KV; derive tokens per unit util.
    CALIB_TOKENS_PER_UTIL=569646  # 398752 / 0.7 ~= 569646
    HEADROOM_FACTOR=1.2
    GPU_MEMORY_UTIL=$(python3 - <<PY
import math
max_len = float("${MAX_MODEL_LEN}")
agents = float("${NUM_AGENTS}")
headroom = float("${HEADROOM_FACTOR}")
calib = float("${CALIB_TOKENS_PER_UTIL}")
desired_tokens = max_len * agents * headroom
util = desired_tokens / calib
util = max(0.5, min(0.9, util))
print(f"{util:.3f}")
PY
)
    if ! [[ "$GPU_MEMORY_UTIL" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        GPU_MEMORY_UTIL="0.8"
    fi
    echo "Auto-selected GPU memory utilization: $GPU_MEMORY_UTIL (agents=${NUM_AGENTS}, max_len=${MAX_MODEL_LEN})"
fi

# Resolve model path after overrides (fallback to input path if realpath fails)
if HOST_MODEL_DIR="$(realpath "$MODEL_PATH" 2>/dev/null)"; then
    :
else
    HOST_MODEL_DIR="$MODEL_PATH"
fi
MODEL_PARENT_DIR="$(dirname "$HOST_MODEL_DIR")"
MODEL_BASENAME="$(basename "$HOST_MODEL_DIR")"
CONTAINER_MODEL_PATH="/model_mount/$MODEL_BASENAME"

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Port must be a number (received '$PORT')"
    exit 1
fi

echo "========================================"
echo "Starting vLLM Server (Docker) for Qwen3-Next"
echo "========================================"
echo "Image: $IMAGE"
echo "Model: $HOST_MODEL_DIR"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTIL:-auto}"
echo "Num Agents (target): $NUM_AGENTS"
echo "DType: $DTYPE"
echo "========================================"
echo

# Check if model exists
if [ ! -d "$HOST_MODEL_DIR" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please download the model first using:"
    echo "  huggingface-cli download cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit --local-dir \"$MODEL_PATH\" --resume-download"
    exit 1
fi

# Check if model download is complete
if [ ! -f "$HOST_MODEL_DIR/config.json" ]; then
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
# Note: Qwen supports tool calling via 'qwen' parser; no built-in reasoning parser flag.
#       Long context >262k requires rope scaling; enable via --rope-scaling as needed.
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "$PORT:$PORT" \
    -v "$MODEL_PARENT_DIR:/model_mount" \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    "$IMAGE" \
    vllm serve "$CONTAINER_MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    ${TOOL_PARSER:+--tool-call-parser "$TOOL_PARSER"} \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    $( [ "$TRUST_REMOTE_CODE" = "true" ] && echo "--trust-remote-code" ) \
    $( [ "$ENFORCE_EAGER" = "true" ] && echo "--enforce-eager" ) \
    --enable-auto-tool-choice
    # Uncomment to experiment with speculative decoding (multi-token prediction):
    # --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
    # For long-context rope scaling (example to 1M tokens):
    # --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}'

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
# - --tool-call-parser qwen: Enables Qwen-compatible tool calling
# - --enable-auto-tool-choice: Automatically selects tools
# - No --reasoning-parser: Model uses non-thinking mode only
# - --max-model-len: Default conservative 2048 to avoid OOM; raise gradually
# - --gpu-memory-utilization: Fraction of GPU memory to use (0.8 = 80%)
# - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1: Required for long contexts
# - Adjust tensor-parallel-size based on GPU setup; >1 shards across GPUs
