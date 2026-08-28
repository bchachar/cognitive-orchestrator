#!/usr/bin/env bash
# Single-A40 fallback: one VLM serves BOTH perception and reasoning.
# Point CO_LLM_BASE_URL and CO_VLM_BASE_URL at this same endpoint (port 8001),
# and set CO_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8001 --dtype bfloat16 --gpu-memory-utilization 0.92 \
  --max-model-len 8192 --limit-mm-per-prompt image=2
