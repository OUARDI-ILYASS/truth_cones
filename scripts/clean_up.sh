#!/usr/bin/env bash
# cleanup_model.sh — delete a specific HF model from cache and free VRAM
# Usage: bash cleanup_model.sh <hf_model_id>
# Example: bash cleanup_model.sh Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL_ID="${1:-}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"

if [[ -z "$MODEL_ID" ]]; then
    echo "Usage: $0 <hf_model_id>"
    echo "Example: $0 Qwen/Qwen2.5-7B-Instruct"
    exit 1
fi

# HF cache uses "models--ORG--NAME" directory naming
MODEL_DIR_NAME="models--$(echo "$MODEL_ID" | tr '/' '--')"
MODEL_CACHE_PATH="$HF_CACHE_DIR/$MODEL_DIR_NAME"

echo "=== Cache cleanup: $MODEL_ID ==="

rm -rf ~/.cache/huggingface/hub/*
# 1. Delete model weights from HF cache
if [[ -d "$MODEL_CACHE_PATH" ]]; then
    SIZE=$(du -sh "$MODEL_CACHE_PATH" 2>/dev/null | cut -f1)
    echo "[1/3] Deleting $MODEL_CACHE_PATH ($SIZE)..."
    rm -rf ~/.cache/huggingface/hub/*
    echo "      Done."
else
    echo "[1/3] Cache dir not found, skipping: $MODEL_CACHE_PATH"
fi

# 2. Python: CUDA empty_cache + gc
echo "[2/3] Freeing VRAM..."
python3 - <<'EOF'
import gc
import torch

if torch.cuda.is_available():
    before = torch.cuda.memory_reserved() / 1e9
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()         # second pass clears fragmented blocks
    after = torch.cuda.memory_reserved() / 1e9
    print(f"      VRAM reserved: {before:.2f} GB → {after:.2f} GB")
else:
    print("      No CUDA device found, skipping.")
EOF

# 3. Print free disk + VRAM summary
echo "[3/3] Post-cleanup state:"
df -h /root 2>/dev/null || df -h "$HOME"

python3 - <<'EOF'
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"      VRAM free: {free/1e9:.2f} GB / {total/1e9:.2f} GB")
EOF

echo "=== Done ==="