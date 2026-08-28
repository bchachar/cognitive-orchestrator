#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot bring-up + run for the cognitive-orchestrator against a live vLLM.
# ASSUMES you are already INSIDE a GPU allocation on a compute node
# (srun --gres=gpu:a40:1 ... --pty bash) and in the project directory.
#
# Usage:
#   bash run_demo.sh serve      # start the vLLM server in the background
#   bash run_demo.sh wait       # wait until the server answers, then ping it
#   bash run_demo.sh run "your command here"   # run the cognitive loop
#   bash run_demo.sh all "your command here"   # serve + wait + run in one go
#   bash run_demo.sh stop       # kill the background server
# ---------------------------------------------------------------------------
set -uo pipefail

PROJECT="$HOME/projects/cognitive-orchestrator"
VENV="$PROJECT/.venv"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PORT=8001
LOG=/tmp/vllm.log

# Always use the venv's interpreter explicitly (this venv has no pip; it's uv-built).
PY="$VENV/bin/python"

setup_env() {
  cd "$PROJECT"
  export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES=0
  export HF_HOME="$HOME/hf_cache"
  # avoid the FlashInfer JIT-compile crash (node has CUDA runtime but no compiler)
  export VLLM_ATTENTION_BACKEND=FLASH_ATTN
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # point the agents at the local server
  export CO_MOCK_MODELS=false
  export CO_VLM_BASE_URL="http://localhost:$PORT/v1"
  export CO_VLM_MODEL="$MODEL"
  export CO_LLM_BASE_URL="http://localhost:$PORT/v1"
  export CO_LLM_MODEL="$MODEL"
}

serve() {
  setup_env
  echo ">> starting vLLM server on port $PORT (log: $LOG)"
  "$VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 1}' \
    > "$LOG" 2>&1 &
  echo ">> server PID $! (backgrounded). Use 'bash run_demo.sh wait' next."
}

wait_up() {
  echo ">> waiting for server to answer on :$PORT ..."
  for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
      echo ">> server is UP:"
      curl -s "http://localhost:$PORT/v1/models" | head -c 300; echo
      return 0
    fi
    sleep 5
  done
  echo "!! server did not come up in time; check $LOG"
  return 1
}

run_loop() {
  setup_env
  local cmd="${1:-Sort the cubes into matching bins and set fragile items aside.}"
  echo ">> running cognitive loop: $cmd"
  "$PY" -m cognitive_orchestrator.main "$cmd"
}

stop() {
  echo ">> stopping any vllm serve process"
  pkill -f "vllm serve" && echo "stopped." || echo "none running."
}

case "${1:-}" in
  serve) serve ;;
  wait)  wait_up ;;
  run)   run_loop "${2:-}" ;;
  all)   serve; wait_up && run_loop "${2:-}" ;;
  stop)  stop ;;
  *) echo "usage: bash run_demo.sh {serve|wait|run \"cmd\"|all \"cmd\"|stop}";;
esac