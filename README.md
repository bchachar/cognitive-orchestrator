# Cognitive Orchestrator

A multi-agent agentic loop for a tabletop robot: natural-language command + camera
image → VLM perception → task planning → skill execution → verification → bounded
replan.

---

## What it does

Cognitive Orchestrator takes a natural-language instruction (e.g. *"sort the cubes
into matching bins and set fragile items aside"*) together with a camera image of a
tabletop scene.  A vision-language model (VLM) converts the image into a structured
scene description; a planner turns goal + scene into an ordered skill sequence; those
skills are dispatched to an edge service representing the robot's actuator layer; and
a verifier inspects the result — triggering a bounded replan loop if the outcome does
not match the goal.  The whole pipeline runs as a [LangGraph](https://github.com/langchain-ai/langgraph)
state machine, making the control flow explicit and inspectable.

The same codebase runs in two modes:
- **Local + mocked** — deterministic fake model responses, no GPU, instant CI.
- **Real** — against `vLLM`-served Qwen models on a GPU node.

---

## Architecture

```
     user text + camera image
              │
              ▼
     ┌─────────────────┐  clarify
     │   Supervisor    │──────────────▶  ask user, stop
     │ (conversational)│
     └────────┬────────┘  execute
              ▼
     ┌─────────────────┐  image ──▶ Scene (objects, colours, fragility, bins)
     │  Perception VLM │
     └────────┬────────┘
              ▼
     ┌─────────────────┐  goal + scene + skill manifest ──▶ Plan (ordered steps)
     │    Planner      │◀─────────────────┐
     └────────┬────────┘  replan hint     │
              ▼                            │
     ┌─────────────────┐                  │
     │  Skill executor │  HTTP / local    │  world state mutated
     │  (edge service) │──────────────────┘
     └────────┬────────┘
              ▼
     ┌─────────────────┐  fail + budget left → replan hint ──▶ Planner
     │    Verifier     │
     └────────┬────────┘
              │  success / budget spent
              ▼
             END
```

**Cloud / server side:** supervisor, VLM perception, planner, verifier (LLM-heavy).  
**Edge side:** the FastAPI skill-service that owns world state and executes
low-latency actuation — the component that would ship on the robot.

---

## Key ideas

| Idea | Where it lives |
|---|---|
| **Multi-agent orchestration (LangGraph)** | `orchestration/graph.py` — a typed LangGraph state machine with 5 nodes and conditional edges for the replan branch |
| **VLM perception** | `perception/vlm.py` — an image is encoded as base-64 and sent to a Qwen2.5-VL endpoint; the JSON response is validated into a `Scene` Pydantic model |
| **Edge / cloud split** | Reasoning and perception run server-side; `skills/service.py` (FastAPI) is the edge component that owns the world state and would deploy on the robot. `skills/client.py` provides a `LocalSkillClient \| HTTPSkillClient` interface so the graph is agnostic to deployment topology |
| **Verification-based feedback loop** | `agents/verifier.py` re-examines world state against the original goal and emits a structured `VerifierResult`; a failing result re-enters the planner with a natural-language hint, bounded by `CO_MAX_REPLANS` |
| **Typed contracts everywhere** | Every agent-to-agent handoff crosses a Pydantic boundary (`schemas.py`), catching hallucinated field names at the boundary rather than deep in downstream code |

---

## Quickstart (local, mocked, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # installs the package + pytest

pytest -q                        # all tests should pass

python -m cognitive_orchestrator.main
# or with a custom command:
python -m cognitive_orchestrator.main "Put the red cube in the red bin."
```

Expected transcript:

```
mock_models=True
command='Sort the cubes into their matching colored bins and set any fragile objects aside in the safe zone.'

[supervisor] Understood. Planning now.
[perceive] 3 objects seen
[plan] 6 steps (replans=0)
[execute] ran 6 skills
[verify] success=True :: All cubes binned; fragile item set aside.
```

Run the edge service standalone and call it over HTTP:

```bash
uvicorn cognitive_orchestrator.skills.service:app --port 8080
curl localhost:8080/skills
curl -X POST localhost:8080/execute \
     -H 'Content-Type: application/json' \
     -d '{"skill":"pick","args":{"object_id":"red_cube_1"}}'
```

---

## Run against a real model (vLLM on GPU)

The system targets Qwen2.5-14B-Instruct (reasoning) and Qwen2.5-VL-7B-Instruct
(perception).  A single A40 (48 GB) can serve both.

### Two GPUs (recommended)

```bash
pip install -e .
pip install "vllm>=0.6" qwen-vl-utils

bash scripts/serve_llm.sh   # GPU 0 → Qwen2.5-14B-Instruct on :8000
bash scripts/serve_vlm.sh   # GPU 1 → Qwen2.5-VL-7B-Instruct on :8001

cp .env.example .env
# edit .env: set CO_MOCK_MODELS=false and uncomment the URL lines

python -m cognitive_orchestrator.main \
  "Sort the cubes into matching bins and set fragile items aside."
```

### One GPU (fallback)

```bash
bash scripts/serve_single_gpu.sh   # one VLM serves both roles on :8001

export CO_MOCK_MODELS=false
export CO_LLM_BASE_URL=http://localhost:8001/v1
export CO_VLM_BASE_URL=http://localhost:8001/v1
export CO_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

python -m cognitive_orchestrator.main
```

### On SLURM

`scripts/slurm_gpu.sbatch` requests the GPUs, starts both servers, waits for
`/v1/models` to respond, runs the loop, and tears the servers down.  Edit
`--partition` and `--gres` to match your cluster, then:

```bash
sbatch scripts/slurm_gpu.sbatch
```

For interactive development, grab a node (`srun --gres=gpu:a40:2 --pty bash`),
start the servers in separate `tmux` panes, and run the CLI in a third.

> **Note:** models download from Hugging Face on first run.  Set `HF_HOME` to a
> scratch directory with enough space.  vLLM flags and Qwen2.5-VL support vary by
> version — if a flag is rejected, consult `vllm serve --help` on the installed
> build.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest -v
```

Tests are fully mocked (no model servers needed):

| File | What it covers |
|---|---|
| `tests/test_schemas.py` | Pydantic model validation and round-trips |
| `tests/test_registry.py` | Skill registry, manifest generation, error handling |
| `tests/test_graph_mock.py` | End-to-end graph run with the mock brain |

---

## Repo layout

```
src/cognitive_orchestrator/
  schemas.py              Pydantic contracts crossing every agent boundary
  config.py               pydantic-settings; the only place URLs/models live
  llm.py                  OpenAI-compatible client (vLLM) + deterministic mock brain
  perception/
    vlm.py                VLM perception agent: image → Scene
    simulator.py          Pillow-based synthetic scene renderer
  agents/
    supervisor.py         conversational front door / router
    planner.py            high-level task planner
    verifier.py           closes the feedback loop; emits replan hints
  skills/
    registry.py           skill library (pick / place / inspect) + manifest
    service.py            FastAPI edge service exposing skills over HTTP
    client.py             LocalSkillClient | HTTPSkillClient (same interface)
  orchestration/
    graph.py              LangGraph state machine wiring the whole loop
  main.py                 end-to-end CLI entry point
tests/                    schema / registry / full mocked end-to-end tests
scripts/                  vLLM serve scripts and SLURM sbatch files
Dockerfile.skills         containerised edge component (no GPU, no model weights)
```

---

## Configuration reference

All settings are in `config.py` and can be overridden via environment variables
(prefix `CO_`) or a `.env` file.  See `.env.example` for the full list.

| Variable | Default | Meaning |
|---|---|---|
| `CO_MOCK_MODELS` | `true` | Use deterministic mock responses (no GPU needed) |
| `CO_LLM_BASE_URL` | `http://localhost:8000/v1` | Reasoning LLM endpoint |
| `CO_LLM_MODEL` | `Qwen/Qwen2.5-14B-Instruct` | Model name sent to the API |
| `CO_VLM_BASE_URL` | `http://localhost:8001/v1` | Perception VLM endpoint |
| `CO_VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | VLM model name |
| `CO_SKILL_SERVICE_URL` | `http://localhost:8080` | Edge skill-service URL |
| `CO_MAX_REPLANS` | `2` | Maximum replan iterations before giving up |

---

## Design notes / limitations

- **Synthetic renderer, not a real camera.** `perception/simulator.py` uses Pillow
  to generate a simple top-down scene image.  The VLM therefore reads a rendered
  picture, not real camera footage.  Hooks for a PyBullet renderer are documented
  in the file but not yet implemented.  This is a demonstrator, not a production
  perception pipeline.

- **Skills are in-memory stubs.** The skill-service tracks object positions in a
  Python dict.  Real actuation (motor commands, gripper drivers) would replace the
  stub implementations in `skills/registry.py`.

- **No benchmarks.** No quantitative results are reported here because none have
  been measured.  Reliability of the real-model path depends on prompt quality
  and model version and has not been systematically evaluated.

- **LLM API keys.** vLLM's OpenAI-compatible endpoint does not validate API keys.
  The default value `"EMPTY"` is intentional — the client library requires a
  non-empty string but the server ignores it.  Do not put a real key in config;
  use environment variables or `.env` if your deployment requires one.

---

## Requirements

Python ≥ 3.10.  Core dependencies are in `pyproject.toml`.  The GPU path
additionally needs `vllm` (recent build with Qwen2.5-VL support) and
`qwen-vl-utils`.  If LangGraph or vLLM APIs have changed since this was written,
the underlying structure (typed state, nodes, conditional edges, OpenAI-compatible
endpoints) is stable — adapt specific call signatures to the installed versions.
