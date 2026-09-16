#!/usr/bin/env bash
# Template for a Qwen3-VL API, executed in the separate gui-pra-vllm environment.
set -euo pipefail
: "${VLM_MODEL_PATH:?Set VLM_MODEL_PATH to a Qwen3-VL checkpoint or HF ID}"
: "${VLM_GPUS:?Set VLM_GPUS to GPUs reserved for this API}"
VLM_MODEL_NAME="${VLM_MODEL_NAME:-qwen3-vl-8b-instruct}"
VLM_PORT="${VLM_PORT:-6008}"
VLM_TP="${VLM_TP:-1}"
VLM_CONTEXT="${VLM_CONTEXT:-16384}"
VLM_MAX_SEQS="${VLM_MAX_SEQS:-4}"
VLM_MEMORY="${VLM_MEMORY:-0.85}"
command=(vllm serve "${VLM_MODEL_PATH}"
  --served-model-name "${VLM_MODEL_NAME}"
  --host "${VLM_HOST:-127.0.0.1}" --port "${VLM_PORT}"
  --tensor-parallel-size "${VLM_TP}" --dtype bfloat16
  --max-model-len "${VLM_CONTEXT}" --max-num-seqs "${VLM_MAX_SEQS}"
  --gpu-memory-utilization "${VLM_MEMORY}"
  --limit-mm-per-prompt '{"image": 1}' --seed 42)
if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${VLM_GPUS}"
  printf '%q ' "${command[@]}"
  printf '\n'
else
  export CUDA_VISIBLE_DEVICES="${VLM_GPUS}"
  exec "${command[@]}" "$@"
fi
