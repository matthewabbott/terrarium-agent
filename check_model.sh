#!/bin/bash
# Check if GLM-4.5-Air-AWQ-4bit model is fully downloaded

MODEL_PATH="./models/GLM-4.5-Air-AWQ-4bit"

echo "========================================"
echo "Checking Model Download Status"
echo "========================================"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model directory not found: $MODEL_PATH"
    exit 1
fi

echo "✓ Model directory exists"

# Check for essential files
REQUIRED_FILES=(
    "config.json"
    "generation_config.json"
    "chat_template.jinja"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✓ $file"
    else
        echo "❌ Missing: $file"
    fi
done

# Count safetensors files (should be 13)
SAFETENSOR_COUNT=$(find "$MODEL_PATH" -maxdepth 1 -name "*.safetensors" -type f | wc -l)
echo
echo "Safetensors files: $SAFETENSOR_COUNT / 13"

if [ "$SAFETENSOR_COUNT" -eq 13 ]; then
    echo "✓ All model weight files present"
else
    echo "❌ Missing model weight files"
    echo
    echo "Expected 13 .safetensors files (model-00001-of-00013.safetensors through model-00013-of-00013.safetensors)"
fi

# Check for incomplete downloads in cache
INCOMPLETE_COUNT=$(find "$MODEL_PATH/.cache" -name "*.incomplete" 2>/dev/null | wc -l)

if [ "$INCOMPLETE_COUNT" -gt 0 ]; then
    echo
    echo "⚠️  Warning: $INCOMPLETE_COUNT incomplete download(s) found in cache"
    echo "   Download may still be in progress"
fi

# Show total size
TOTAL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
echo
echo "Total size: $TOTAL_SIZE"
echo "(Expected: ~25-30GB for complete download)"

# Final verdict
echo
echo "========================================"
if [ "$SAFETENSOR_COUNT" -eq 13 ] && [ "$INCOMPLETE_COUNT" -eq 0 ]; then
    echo "Status: ✅ Model appears complete and ready!"
    echo
    echo "You can start vLLM with:"
    echo "  ./start_vllm_docker.sh"
elif [ "$INCOMPLETE_COUNT" -gt 0 ]; then
    echo "Status: ⏳ Download in progress..."
    echo
    echo "Check back later or monitor with:"
    echo "  watch -n 5 ./check_model.sh"
else
    echo "Status: ❌ Model incomplete"
    echo
    echo "Resume download with:"
    echo "  python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"cpatonn/GLM-4.5-Air-AWQ-4bit\", local_dir=\"$MODEL_PATH\")'"
fi
echo "========================================"
