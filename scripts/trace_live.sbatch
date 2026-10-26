#!/usr/bin/env bash
#SBATCH --job-name=cog-trace
#SBATCH --partition=gpus
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/trace-%j.log
# ---------------------------------------------------------------------------
# One live run with CO_TRACE=true, then render the trace to HTML.
# Submit from the project root:  sbatch scripts/trace_live.sbatch
# Output: traces/run-<timestamp>.json and .html
# ---------------------------------------------------------------------------
set -uo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT/logs" "$PROJECT/traces"
VENV="$PROJECT/.venv"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PORT=8001
SRVLOG="$PROJECT/logs/vllm-server-$SLURM_JOB_ID.log"

cd "$PROJECT"

export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/hf_cache"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export CO_MOCK_MODELS=false
export CO_VLM_BASE_URL="http://localhost:$PORT/v1"
export CO_VLM_MODEL="$MODEL"
export CO_LLM_BASE_URL="http://localhost:$PORT/v1"
export CO_LLM_MODEL="$MODEL"
export CO_TRACE=true
export CO_TRACE_DIR="$PROJECT/traces"

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --enforce-eager --limit-mm-per-prompt '{"image": 1}' \
  > "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'echo ">> stopping server"; kill $SERVER_PID 2>/dev/null' EXIT

UP=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    UP=1; echo ">> server UP after $((i*5))s"; break
  fi
  sleep 5
done
if [ "$UP" -ne 1 ]; then
  echo "!! server never came up. Tail of server log:"; tail -40 "$SRVLOG"; exit 1
fi

CMD="${1:-Sort the cubes into matching bins and set fragile items aside.}"
"$VENV/bin/python" -m cognitive_orchestrator.main "$CMD"
RC=$?

LATEST="$(ls -t "$PROJECT"/traces/run-*.json 2>/dev/null | head -1)"
if [ -n "$LATEST" ]; then
  "$VENV/bin/python" tools/view_trace.py "$LATEST"
else
  echo "!! no trace file produced"
fi
echo "=== done rc=$RC $(date) ==="
