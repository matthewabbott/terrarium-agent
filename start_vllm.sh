#!/bin/bash
# Startup script for vLLM server with GLM-4.5-Air-AWQ-4bit

set -e

# Configuration
MODEL_PATH="./models/GLM-4.5-Air-AWQ-4bit"
PORT=8000
HOST="0.0.0.0"

# vLLM parameters for GLM-4.5-Air-AWQ-4bit
DTYPE="float16"  # Required for AWQ quantization
TENSOR_PARALLEL_SIZE=2  # Adjust based on your GPU count (1, 2, 4, or 8)
PIPELINE_PARALLEL_SIZE=1  # Keep at 1 unless you have multiple nodes

# Optional: Set visible GPUs (uncomment and adjust if needed)
# export CUDA_VISIBLE_DEVICES=0,1

echo "========================================"
echo "Starting vLLM Server"
echo "========================================"
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

# Activate venv if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if vllm is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not installed"
    echo "Install with: pip install vllm"
    exit 1
fi

echo "Starting vLLM server..."
echo "(This may take a few minutes to load the model)"
echo

# Start vLLM server
# Note: GLM-4.5-Air supports tool calling and reasoning modes
vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9

# Notes:
# - --tool-call-parser glm45: Enables GLM-4.5 tool calling
# - --reasoning-parser glm45: Enables GLM-4.5 reasoning mode
# - --enable-auto-tool-choice: Automatically selects tools
# - --max-model-len: Context length (adjust based on your GPU memory)
# - --gpu-memory-utilization: Fraction of GPU memory to use (0.9 = 90%)
#
# Adjust tensor-parallel-size based on your GPU setup:
#   - 1 GPU: --tensor-parallel-size 1
#   - 2 GPUs: --tensor-parallel-size 2
#   - 4 GPUs: --tensor-parallel-size 4
#
# For debugging, add: --log-level debug
