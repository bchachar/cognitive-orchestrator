"""Score the orchestrator against a live OmniSim arm on OmniLink's five axes.

Runs the deterministic (mocked-brain) orchestrator against the real OmniSim
bridge and reports, separately:

  1. skill_selection        - planner chose the right skills, right order (pick, place)
  2. valid_argument_construction - every skill arg was well-formed & legal
  3. refusal_handling       - supervisor proceeds on a valid task, clarifies on a vague one
  4. final_object_pose      - real world xyz of the block vs. target + error (metres)
  5. completion_identity    - the sim confirms it acted on the intended object (DEF)

Reasoning stays deterministic/mocked; only skill EXECUTION is real (OmniSim).

PREREQUISITES
-------------
  * OmniSim running with the 2f140 world; bridge reachable at BRIDGE_URL.
  * Run from the repo root, in the venv where the package is installed.

USAGE
-----
  export CO_MOCK_MODELS=true CO_USE_OMNISIM=true CO_TASK=omnisim_single
  python score_omnisim.py                 # 5 task runs + 1 refusal run
  python score_omnisim.py --runs 3        # custom repeat count

The script prints a human-readable summary and a machine-readable JSON block
you can paste back. It does NOT modify any repo code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import httpx

# --- config -----------------------------------------------------------------
BRIDGE_URL = os.getenv("BRIDGE_URL", "http://127.0.0.1:8765").rstrip("/")
OBJECT_DEF = os.getenv("OBJECT_DEF", "GRASP_BLOCK")
OBJECT_ID = os.getenv("OBJECT_NAME", "block")
TARGET_LOCATION = os.getenv("TARGET_LOCATION", "left_bin")

# Must match the adapter's LOCATIONS table for the target used by the task.
# Overridable via env if your coords differ.
TARGET_XYZ = json.loads(os.getenv("TARGET_XYZ", "[0.24, -0.42, 0.20]"))

# The valid command the planner should ACT on, and a vague one it should NOT.
VALID_CMD = "Pick the block and place it in the left bin."
VAGUE_CMD = "do something"

POSE_TOLERANCE_M = 0.10   # pose considered "on target" within this radius


def _post(path: str, body: dict) -> dict:
    r = httpx.post(f"{BRIDGE_URL}/{path}", json=body, timeout=60.0)
    r.raise_for_status()
    return r.json()


def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _run_loop(command: str) -> dict:
    """Invoke the orchestrator once and return its final state dict.
    Imported lazily so the script fails cleanly if run outside the venv."""
    from cognitive_orchestrator.main import run  # noqa: WPS433
    return run(command)


def score_task_run() -> dict:
    """One valid-task run; scores axes 1, 2, 4, 5 (and half of 3)."""
    final = _run_loop(VALID_CMD)

    transcript = final.get("transcript", [])
    plan = final.get("plan") or {}
    steps = plan.get("steps", [])
    verdict = final.get("verdict") or {}
    results = final.get("results", [])

    # 1. skill selection: exactly pick then place, in order
    skills_in_order = [s.get("skill") for s in steps]
    skill_selection_ok = skills_in_order[:2] == ["pick", "place"]

    # 2. valid argument construction: every executed skill returned ok
    #    (planner code-validation would have raised on a bad location, so a
    #     completed run with all-ok results means args were well-formed & legal)
    valid_args_ok = bool(results) and all(r.get("ok") for r in results)

    # 5. completion identity: ask the sim what it last released, confirm it's ours
    #    (read straight from the bridge to avoid trusting only our own state())
    last = _post("get_robot_state", {}).get("last_command", {}) or {}
    released = last.get("released")
    completion_identity_ok = (released == OBJECT_DEF)

    # 4. final object pose: real world xyz from the bridge vs. target
    pose = _post("node_pose", {"def": OBJECT_DEF})
    obj_xyz = pose.get("pos")
    pose_error_m = _dist(obj_xyz, TARGET_XYZ) if obj_xyz else None
    pose_on_target = (pose_error_m is not None and pose_error_m <= POSE_TOLERANCE_M)

    return {
        "supervisor_action_on_valid": (final.get("action") or "?"),
        "verify_success": bool(verdict.get("success")),
        "skill_selection_ok": skill_selection_ok,
        "skills_in_order": skills_in_order,
        "valid_argument_construction_ok": valid_args_ok,
        "completion_identity_ok": completion_identity_ok,
        "released_def": released,
        "final_object_pose_world_xyz": obj_xyz,
        "target_world_xyz": TARGET_XYZ,
        "pose_error_m": pose_error_m,
        "pose_on_target": pose_on_target,
        "transcript": transcript,
    }


def score_refusal_run() -> dict:
    """Vague command; the supervisor should CLARIFY, not execute (axis 3)."""
    final = _run_loop(VAGUE_CMD)
    action = final.get("action") or "?"
    return {
        "command": VAGUE_CMD,
        "supervisor_action": action,
        "refused_as_expected": (action == "clarify"),
        "transcript": final.get("transcript", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="valid-task repeat count")
    args = ap.parse_args()

    if os.getenv("CO_USE_OMNISIM") != "true":
        print("WARNING: CO_USE_OMNISIM is not 'true' — you may be scoring the "
              "in-process client, not OmniSim.", file=sys.stderr)

    task_runs = []
    for i in range(args.runs):
        print(f"--- task run {i + 1}/{args.runs} ---", file=sys.stderr)
        task_runs.append(score_task_run())

    print("--- refusal run ---", file=sys.stderr)
    refusal = score_refusal_run()

    # aggregate
    def _all(key):
        return all(r.get(key) for r in task_runs)

    summary = {
        "bridge_url": BRIDGE_URL,
        "object_def": OBJECT_DEF,
        "target_location": TARGET_LOCATION,
        "target_world_xyz": TARGET_XYZ,
        "runs": len(task_runs),
        "axis_1_skill_selection": {
            "all_runs_ok": _all("skill_selection_ok"),
            "observed_order": task_runs[0]["skills_in_order"] if task_runs else None,
        },
        "axis_2_valid_argument_construction": {
            "all_runs_ok": _all("valid_argument_construction_ok"),
        },
        "axis_3_refusal_handling": {
            "valid_task_action": task_runs[0]["supervisor_action_on_valid"] if task_runs else None,
            "vague_task_action": refusal["supervisor_action"],
            "refused_vague_as_expected": refusal["refused_as_expected"],
            "ok": (task_runs and task_runs[0]["supervisor_action_on_valid"] == "execute"
                   and refusal["refused_as_expected"]),
        },
        "axis_4_final_object_pose": {
            "pose_error_m_per_run": [r["pose_error_m"] for r in task_runs],
            "mean_pose_error_m": (
                sum(r["pose_error_m"] for r in task_runs) / len(task_runs)
                if task_runs and all(r["pose_error_m"] is not None for r in task_runs) else None
            ),
            "all_on_target": _all("pose_on_target"),
            "tolerance_m": POSE_TOLERANCE_M,
        },
        "axis_5_completion_identity": {
            "all_runs_ok": _all("completion_identity_ok"),
            "released_def": task_runs[0]["released_def"] if task_runs else None,
            "expected_def": OBJECT_DEF,
        },
        "verify_success_all_runs": _all("verify_success"),
    }

    print("\n================ FIVE-AXIS SCORE ================")
    print(json.dumps(summary, indent=2))
    print("================================================")


if __name__ == "__main__":
    main()