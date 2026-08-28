#!/usr/bin/env bash
#SBATCH --job-name=cog-orch
#SBATCH --partition=longgpu            # <-- set to your cluster's GPU partition
#SBATCH --gres=gpu:a40:1           # 2x A40; use gpu:a40:1 for the single-GPU path
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cog-orch-%j.log

mkdir -p logs

set -euo pipefail
module load cuda || true           # adjust to your module system
source .venv/bin/activate

# Launch both model servers in the background, wait for them to come up.
bash scripts/serve_llm.sh &  LLM_PID=$!
bash scripts/serve_vlm.sh &  VLM_PID=$!
echo "waiting for vLLM servers..."
for url in http://localhost:8000/v1/models http://localhost:8001/v1/models; do
  until curl -sf "$url" >/dev/null; do sleep 5; done
done
echo "servers up."

export CO_MOCK_MODELS=false
python -m cognitive_orchestrator.main \
  "Sort the cubes into their matching colored bins and set fragile items aside."

kill $LLM_PID $VLM_PID
