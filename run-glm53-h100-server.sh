#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${DS4_GLM_MODEL:-/models/ds4-models/GLM-5.3-Flash-Q4_K.gguf}"
cache_target="${DS4_GLM_CACHE_TARGET:-62GB}"
guard_reserve_gb="${DS4_GLM_MEMORY_GUARD_RESERVE_GB:-1}"
ctx="${DS4_GLM_SERVER_CTX:-32768}"
port="${DS4_GLM_SERVER_PORT:-8000}"
[[ -r "$model" ]] || { printf 'Model not readable: %s
' "$model" >&2; exit 1; }
cd "$repo_dir"
exec env \
    DS4_CUDA_NO_DIRECT_IO=1 \
    DS4_CUDA_KEEP_MODEL_PAGES=1 \
    DS4_CUDA_PIN_MODEL_MIRROR=1 \
    DS4_GLM_MEMORY_GUARD_RESERVE_GB="$guard_reserve_gb" \
    ./ds4-server \
        -m "$model" \
        --cuda \
        --ssd-streaming \
        --ssd-streaming-cache-experts "$cache_target" \
        --ctx "$ctx" \
        --host 127.0.0.1 \
        --port "$port"
