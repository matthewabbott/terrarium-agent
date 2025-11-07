# Docker Setup Guide

This guide explains how terrarium-agent uses Docker for vLLM inference on DGX Spark / GB10 GPU systems.

## Why Docker?

The GB10 GPU (Blackwell architecture, compute capability sm_12.1) requires bleeding-edge software:
- PyTorch 2.7+ with CUDA 12.8 support
- vLLM compiled for that specific PyTorch version
- CUDA 12.8+ libraries with Blackwell support

NVIDIA's official vLLM container bundles all of these, ensuring compatibility.

## Architecture

```
┌────────────────────────────────────────┐
│  Host System (DGX Spark / GB10)        │
│  ├─ CUDA 12.8                          │
│  ├─ NVIDIA Driver 580.x                │
│  └─ Docker + nvidia-container-toolkit  │
└────────────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────┐
│  Docker Container                      │
│  ├─ vLLM 0.10.1.1                      │
│  ├─ PyTorch (with GB10 support)        │
│  ├─ CUDA 12.8 libraries                │
│  └─ Model: GLM-4.5-Air-AWQ-4bit        │
│     Port: 8000 (exposed to host)       │
└────────────────────────────────────────┘
                 ↑
                 │ HTTP API calls
                 │
┌────────────────────────────────────────┐
│  Python venv (Host)                    │
│  ├─ chat.py / main.py                  │
│  ├─ openai (client library)            │
│  └─ aiohttp, pyyaml, pydantic          │
└────────────────────────────────────────┘
```

**Key benefit:** Client code runs on host with lightweight dependencies. Heavy ML stack runs in container.

## Prerequisites

1. **NVIDIA GPU:** GB10 (Blackwell) or compatible
2. **NVIDIA Driver:** 580.x+ installed
3. **Docker:** Installed and running
4. **nvidia-container-toolkit:** GPU access in containers
5. **Model Downloaded:** GLM-4.5-Air-AWQ-4bit in `models/` directory

### Verify Prerequisites

```bash
# Check GPU
nvidia-smi

# Check Docker
docker --version

# Check nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Check model
./check_model.sh
```

All should succeed before continuing.

## Quick Start

### 1. Pull NVIDIA vLLM Container

```bash
docker pull nvcr.io/nvidia/vllm:25.09-py3
```

This is a large image (~15-20GB). It includes vLLM, PyTorch, and all CUDA libraries.

### 2. Start vLLM Server

```bash
./start_vllm_docker.sh
```

This script:
- Stops any existing vLLM container
- Starts new container named `vllm-server`
- Mounts `models/` directory to `/models` in container
- Exposes port 8000 to host
- Configures proper GPU access and shared memory

**First startup takes ~2 minutes** to load the 60GB model into GPU memory.

### 3. Verify Server is Running

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check logs
docker logs -f vllm-server

# Look for: "Application startup complete"
```

### 4. Use the Agent

```bash
# Activate venv (lightweight client dependencies only)
source venv/bin/activate

# Interactive chat
python chat.py

# Full agent runtime
python main.py
```

## Container Management

### Start/Stop

```bash
# Start (creates new container)
./start_vllm_docker.sh

# Stop (preserves container)
docker stop vllm-server

# Restart stopped container
docker start vllm-server

# Remove container
docker rm vllm-server
```

### Monitoring

```bash
# View logs (live)
docker logs -f vllm-server

# View last 50 lines
docker logs --tail 50 vllm-server

# Check container status
docker ps | grep vllm-server

# Check GPU usage
nvidia-smi
```

### Debugging

```bash
# Shell into running container
docker exec -it vllm-server /bin/bash

# Check vLLM version
docker exec vllm-server vllm --version

# View container config
docker inspect vllm-server
```

## Configuration

### start_vllm_docker.sh Parameters

Edit the script to customize:

```bash
# Port (default: 8000)
PORT=8000

# Model path (default: ./models/GLM-4.5-Air-AWQ-4bit)
MODEL_PATH="./models/GLM-4.5-Air-AWQ-4bit"

# Tensor parallelism (default: 1 for single GPU)
TENSOR_PARALLEL_SIZE=1

# Max context length (default: 8192)
MAX_MODEL_LEN=8192

# GPU memory utilization (default: 0.9 = 90%)
GPU_MEMORY_UTIL=0.9
```

### Docker Run Flags

The script uses these important flags:

```bash
--gpus all                  # Enable all GPUs
--ipc=host                  # Shared memory (NVIDIA recommendation)
--ulimit memlock=-1         # Unlimited locked memory
--ulimit stack=67108864     # 64MB stack size
-p 8000:8000                # Port mapping
-v $(pwd)/models:/models    # Mount model directory
```

## Troubleshooting

### Container Won't Start

**Check GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, nvidia-container-toolkit isn't configured properly.

**Check port availability:**
```bash
lsof -i :8000
```

If port 8000 is in use, stop the conflicting process or change the port in `start_vllm_docker.sh`.

### Model Not Found

**Error:** "Model not found at /models/GLM-4.5-Air-AWQ-4bit"

**Solution:** Ensure model is in correct location:
```bash
ls -lh models/GLM-4.5-Air-AWQ-4bit/
# Should show 13 safetensors files + config.json
```

### Out of Memory

**Error:** "CUDA out of memory" or model loading fails

**Solutions:**
1. Reduce `--gpu-memory-utilization` (try 0.8 instead of 0.9)
2. Reduce `--max-model-len` (try 4096 instead of 8192)
3. Check nothing else is using GPU: `nvidia-smi`

### Slow First Query

**Expected:** First query after startup takes ~5-10s (graph compilation)

**Subsequent queries:** ~1-2s (normal inference speed)

This is torch.compile warming up CUDA graphs - happens once per container start.

## Performance

### Model Loading

- **Time:** ~90-120 seconds
- **GPU Memory:** ~60GB (for 4-bit quantized model)
- **Disk Space:** ~60GB (model files)

### Inference

- **First query:** ~5-10s (includes torch.compile warmup)
- **Subsequent queries:** ~1-2s typical
- **Tokens/second:** ~20-30 (depends on sequence length)

### Container Overhead

- **Memory:** ~1-2GB additional for container runtime
- **CPU:** Minimal (vLLM is GPU-bound)
- **Startup time:** ~10s (excluding model loading)

## Comparison: venv vs Docker

### Old Setup (venv - deprecated)
```
✗ Install PyTorch with pip (may not support GB10)
✗ Install vLLM (may have ABI incompatibilities)
✗ Manual CUDA version management
✗ ~5-6GB in venv directory
```

### New Setup (Docker - recommended)
```
✓ Pre-built container with matched versions
✓ Official NVIDIA support for GB10
✓ Isolated from host Python environment
✓ Venv only ~1.5GB (client libraries)
✓ Easy to update (just pull new image)
```

## Updating

### Update Container

```bash
# Pull latest image
docker pull nvcr.io/nvidia/vllm:25.09-py3

# Stop old container
docker stop vllm-server
docker rm vllm-server

# Start with new image
./start_vllm_docker.sh
```

### Update Model

```bash
# Download new model to models/ directory
# (or update existing model files)

# Restart container (it will load updated model)
docker restart vllm-server
```

## Integration with terrarium-irc

When integrating with terrarium-irc:

1. **vLLM server runs in Docker** (always on, port 8000)
2. **terrarium-irc connects via HTTP** (http://localhost:8000)
3. **No vLLM/PyTorch in IRC venv** (uses OpenAI client library only)

See `INTEGRATION_DESIGN.md` for details.

## Security Notes

- Container runs as root by default (standard for NVIDIA containers)
- Model directory is mounted read-only by default (can add `:ro` flag)
- No external network access required (local inference)
- Health endpoint is unauthenticated (localhost only by default)

To expose to network (not recommended without authentication):
```bash
# Change HOST="0.0.0.0" to HOST="127.0.0.1" in script for localhost-only
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA vLLM Container Docs](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/)
- [DGX Spark vLLM Setup](https://build.nvidia.com/spark/vllm/instructions)
- [nvidia-container-toolkit Setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Support

For issues:
1. Check logs: `docker logs vllm-server`
2. Verify GPU: `nvidia-smi`
3. Test container: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
4. Check model files: `./check_model.sh`
