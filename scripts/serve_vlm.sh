#!/usr/bin/env bash
# Perception VLM as a second OpenAI-compatible server. Pin to GPU 1.
# Needs a recent vLLM (Qwen2.5-VL support) + `pip install qwen-vl-utils`.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 2}' \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct
