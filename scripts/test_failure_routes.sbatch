#!/usr/bin/env bash
#SBATCH --job-name=cog-routes
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/routes-%j.log
# ---------------------------------------------------------------------------
# Full test of the failure-triage routes:
#   1. mock-mode unit suite (CPU logic, deterministic)
#   2. bring up vLLM (single VLM serving perception + reasoning)
#   3. live happy-path runs of the CLI
#   4. live route tests (tests/test_failure_routes_live.py)
# Submit:  sbatch scripts/test_failure_routes.sbatch
# Watch:   tail -f logs/routes-<jobid>.log
# Exit code is non-zero if any phase failed.
# ---------------------------------------------------------------------------
set -uo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT/logs"
VENV="$PROJECT/.venv"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PORT=8001
SRVLOG="$PROJECT/logs/vllm-server-$SLURM_JOB_ID.log"
FAIL=0

cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/hf_cache"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo; echo "########## PHASE 1: mock unit suite ##########"
CO_MOCK_MODELS=true "$VENV/bin/python" -m pytest -v tests || { FAIL=1; echo "!! PHASE 1 FAILED"; }

echo; echo "########## PHASE 2: start vLLM ##########"
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
  kill -0 $SERVER_PID 2>/dev/null || break
  sleep 5
done
if [ "$UP" -ne 1 ]; then
  echo "!! server never came up. Tail of server log:"; tail -40 "$SRVLOG"; exit 1
fi

export CO_MOCK_MODELS=false
export CO_VLM_BASE_URL="http://localhost:$PORT/v1"
export CO_VLM_MODEL="$MODEL"
export CO_LLM_BASE_URL="http://localhost:$PORT/v1"
export CO_LLM_MODEL="$MODEL"

echo; echo "########## PHASE 3: live happy-path CLI x3 ##########"
CMD="Sort the cubes into matching bins and set fragile items aside."
for i in 1 2 3; do
  echo "--- run $i ---"
  "$VENV/bin/python" -m cognitive_orchestrator.main "$CMD" | tee "$PROJECT/logs/run-$SLURM_JOB_ID-$i.out"
  grep -q "'success': True" "$PROJECT/logs/run-$SLURM_JOB_ID-$i.out" || { FAIL=1; echo "!! run $i did not succeed"; }
done

echo; echo "########## PHASE 4: live failure-route tests ##########"
"$VENV/bin/python" -m pytest -v tests/test_failure_routes_live.py || { FAIL=1; echo "!! PHASE 4 FAILED"; }

echo; echo "=== done $(date)  overall: $([ $FAIL -eq 0 ] && echo PASS || echo FAIL) ==="
exit $FAIL
