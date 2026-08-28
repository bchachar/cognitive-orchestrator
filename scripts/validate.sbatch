#!/usr/bin/env bash
#SBATCH --job-name=cog-validate
#SBATCH --partition=gpus
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate-%j.log      # all output goes here; %j = job id
# ---------------------------------------------------------------------------
# Batch validation: bring up vLLM, run the cognitive loop 5x, tear down.
# Submit with:   sbatch validate.sbatch
# Watch with:    tail -f validate-<jobid>.log
# ---------------------------------------------------------------------------
set -uo pipefail

# Resolve the project root from the location of this script (scripts/ subdir).
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT/logs"
VENV="$PROJECT/.venv"                 # the WORKING venv (not .venv.old)
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PORT=8001
SRVLOG="$PROJECT/logs/vllm-server-$SLURM_JOB_ID.log"

cd "$PROJECT"

# --- environment (same knobs proven to work interactively) ---
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

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- start server in background ---
echo ">> starting vLLM (server log: $SRVLOG)"
"$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --enforce-eager --limit-mm-per-prompt '{"image": 1}' \
  > "$SRVLOG" 2>&1 &
SERVER_PID=$!

# --- ensure server is killed no matter how the job ends ---
trap 'echo ">> stopping server"; kill $SERVER_PID 2>/dev/null' EXIT

# --- wait up to 10 min for readiness ---
echo ">> waiting for server..."
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

# --- run the loop 5 times ---
CMD="Sort the cubes into matching bins and set fragile items aside."
for i in 1 2 3 4 5; do
  echo ""; echo "########## RUN $i ##########"
  "$VENV/bin/python" -m cognitive_orchestrator.main "$CMD"
done

echo ""; echo "=== also running mock unit tests for regression ==="
"$VENV/bin/python" -m pytest -q || echo "(pytest reported failures above)"

echo "=== done $(date) ==="
