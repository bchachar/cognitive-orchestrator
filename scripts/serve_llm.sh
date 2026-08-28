#!/usr/bin/env bash
# Reasoning LLM (planner / supervisor / verifier) as an OpenAI-compatible server.
# Pin to GPU 0. ~28GB in bf16 on one A40 (48GB); drop to Qwen2.5-7B if tight.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name Qwen/Qwen2.5-14B-Instruct
