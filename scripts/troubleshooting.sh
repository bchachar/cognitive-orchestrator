#!/usr/bin/env bash
# ============================================================================
# Reproducer: vLLM offline inference works on the A40 node, but `vllm serve`
# fails at import with:  ImportError: libtorch_cuda.so: cannot open shared
# object file. Looks like a vLLM<->torch build mismatch (the file is absent
# from the venv entirely), not an LD_LIBRARY_PATH issue.
#
# HOW IT WAS BUILT (per your recommendation):
#   uv venv
#   source .venv/bin/activate
#   uv pip install vllm --torch-backend auto
#   uv pip install -e ".[dev]"          # small project deps (langgraph, fastapi, ...)
#
# Run this from the project root on an A40 node inside the venv:
#   bash repro_vllm_serve.sh
# It prints the environment facts, confirms offline inference, then shows the
# serve failure. Please share the torch + vllm versions from YOUR working
# setup so I can pin the matching pair.
# ============================================================================
set -uo pipefail

echo "===== 1. Node / GPU ====="
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo

echo "===== 2. Versions (the key question) ====="
python - <<'PY'
import importlib.metadata as m
def v(pkg):
    try:    return m.version(pkg)
    except Exception as e: return f"NOT INSTALLED ({e})"
import torch
print("python       :", __import__("sys").version.split()[0])
print("torch        :", torch.__version__)
print("torch.cuda   :", torch.version.cuda)
print("cuda_avail   :", torch.cuda.is_available())
print("vllm         :", v("vllm"))
print("xformers     :", v("xformers"))
PY
echo

echo "===== 3. Is libtorch_cuda.so present anywhere in the venv? ====="
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "site-packages: $SITE"
echo "-- libtorch_cuda.so matches --"
find "$SITE" -name "libtorch_cuda*.so*" 2>/dev/null || true
echo "(empty above = the file vLLM's extension wants is missing)"
echo "-- what torch DOES ship in torch/lib --"
ls -1 "$SITE/torch/lib/" 2>/dev/null | grep -i cuda || echo "  (no *cuda* libs in torch/lib)"
echo

echo "===== 4. Offline inference (this WORKS) ====="
python - <<'PY'
try:
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", dtype="bfloat16",
              gpu_memory_utilization=0.9, max_model_len=4096)
    out = llm.generate(["Say hello in three words."], SamplingParams(max_tokens=16))
    print("OFFLINE OK ->", out[0].outputs[0].text.strip())
except Exception as e:
    print("OFFLINE FAILED ->", type(e).__name__, e)
PY
echo

echo "===== 5. Server mode (this FAILS at import) ====="
echo "\$ vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --dtype bfloat16 --gpu-memory-utilization 0.9 --max-model-len 8192"
timeout 60 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8001 --dtype bfloat16 \
  --gpu-memory-utilization 0.9 --max-model-len 8192 2>&1 | head -40
echo
echo "===== end ====="