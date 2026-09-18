#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${DS4_GLM_MODEL:-/models/ds4-models/GLM-5.3-Flash-Q4_K.gguf}"
cache_target="${DS4_GLM_CACHE_TARGET:-62GB}"
guard_reserve_gb="${DS4_GLM_MEMORY_GUARD_RESERVE_GB:-1}"
pin_mirror="${DS4_CUDA_PIN_MODEL_MIRROR:-1}"

if [[ ! -r "$model" ]]; then
    printf 'Model not readable: %s
' "$model" >&2
    exit 1
fi

if [[ "$pin_mirror" == 0 ]]; then
    # Buffered fallback: faster startup, but decode is limited by host page-cache bandwidth.
    printf 'Warming GLM model in host page cache: %s
' "$model" >&2
    dd if="$model" of=/dev/null bs=64M status=none
else
    printf 'Pinned mirror mode: startup takes about 4-5 minutes; keep this process running.
' >&2
fi

cd "$repo_dir"
exec env \
    DS4_CUDA_NO_DIRECT_IO=1 \
    DS4_CUDA_KEEP_MODEL_PAGES=1 \
    DS4_CUDA_PIN_MODEL_MIRROR="$pin_mirror" \
    DS4_GLM_MEMORY_GUARD_RESERVE_GB="$guard_reserve_gb" \
    ./ds4 \
        -m "$model" \
        --cuda \
        --ssd-streaming \
        --ssd-streaming-cache-experts "$cache_target" \
        "$@"
