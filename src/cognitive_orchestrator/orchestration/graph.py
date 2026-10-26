"""The cognitive loop, as a LangGraph state machine.

    supervisor ──clarify──▶ END
        │execute
        ▼
    perceive (VLM) ──▶ plan ──▶ execute (skills) ──▶ verify
        ▲                ▲          ▲                      │
        │                │          └─ retry failed step ──┤
        └────────────────┴────── triage (typed Failure) ◀──┘
                                    │success      │EXHAUSTED
                                    ▼             ▼
                                   END         escalate ──▶ END

Every stage hands off to `triage`, which classifies the failure and routes:
PERCEPTION -> perceive, PLANNING -> plan (constraint in hint), EXECUTION -> retry
the failed step (max `max_step_retries`, then replan), VERIFICATION -> plan,
EXHAUSTED -> escalate (a question for a human). Every route except a step retry
spends one unit of `max_replans`, so the loop always terminates.

This is the "end-to-end cognitive loop: multimodal input -> contextual reasoning
-> action planning -> skill orchestration -> feedback" from the job description,
expressed as an explicit, inspectable graph rather than a prompt-chain black box.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..agents import planner, preflight, supervisor, triage, verifier
from ..config import settings
from ..observability import traced
from ..perception import vlm
from ..schemas import Failure, FailureKind, Plan, Scene, SkillName
from ..skills.client import SkillClient


class LoopState(TypedDict, total=False):
    user_message: str
    image_b64: str
    seed: dict                 # ground-truth to (re)initialise the edge world
    scene: dict
    plan: dict
    results: list[dict]
    final_state: dict
    verdict: dict
    replans: int               # global budget spent on perceive/plan restarts
    step_retries: dict         # failed step index -> retries used (this plan)
    retry_step: Optional[int]  # set by triage: re-run from this step only
    plan_error: str
    hint: str                  # constraint fed to the next plan pass
    failure: Optional[dict]
    stage: str
    route: str
    escalation: str
    action: str
    transcript: list[str]
    trace: Any                 # observability.Trace when CO_TRACE=true


def build_graph(skill: SkillClient):
    """Factory: closes over a concrete SkillClient (local or HTTP)."""

    def _log(state: LoopState, line: str) -> list[str]:
        return [*state.get("transcript", []), line]

    def supervisor_node(state: LoopState) -> dict:
        decision = supervisor.route(state["user_message"])
        return {"action": decision.get("action", "execute"),
                "transcript": _log(state, f"[supervisor] {decision.get('message', '')}")}

    def perceive_node(state: LoopState) -> dict:
        seed = state.get("seed") or {}
        known_ids = list((seed.get("locations") or {}).keys()) or None
        scene: Scene = vlm.perceive(state["image_b64"], known_ids=known_ids)
        # Seed fragility is ground truth; VLM may be uncertain about material.
        fragile_map: dict[str, bool] = seed.get("fragile") or {}
        for obj in scene.objects:
            if obj.id in fragile_map:
                obj.fragile = fragile_map[obj.id]
        return {"stage": "perceive", "scene": scene.model_dump(),
                "transcript": _log(state, f"[perceive] {len(scene.objects)} objects seen")}

    def plan_node(state: LoopState) -> dict:
        scene = Scene.model_validate(state["scene"])
        goal = state["user_message"]
        hint = state.get("hint", "")
        if hint:
            goal = f"{goal}\n(Previous attempt failed. Fix: {hint})"
        update: dict = {"stage": "plan", "hint": "", "plan_error": "", "step_retries": {},
                        "retry_step": None, "results": [], "verdict": None}
        try:
            p: Plan = planner.plan(goal, scene, skill.manifest())
            manifest = skill.manifest()
            bad = [s.index for s in p.steps if s.skill.value not in manifest]
            if bad:
                raise ValueError(f"Plan steps {bad} use skills missing from the manifest "
                                 f"{sorted(manifest)}")
            seed = state["seed"]
            violations = preflight.validate_plan(
                p, {"locations": seed["locations"], "fragile": seed.get("fragile", {}),
                    "held": None})
            if violations:
                raise ValueError("; ".join(violations))
        except ValueError as e:
            return {**update, "plan": None,
                    "plan_error": f"Plan rejected: {e}",
                    "transcript": _log(state, f"[plan] rejected: {e}")}
        return {**update, "plan": p.model_dump(),
                "transcript": _log(state, f"[plan] {len(p.steps)} steps (replans={state.get('replans', 0)})")}

    def execute_node(state: LoopState) -> dict:
        plan = Plan.model_validate(state["plan"])
        retry = state.get("retry_step")
        if retry is None:
            # Full pass: re-seed the edge world so replanning is idempotent.
            seed = state["seed"]
            skill.seed(seed["locations"], seed["fragile"])
            start, results = 0, []
        else:
            # Step retry: keep the world and the plan, redo the failed step, then
            # continue with the steps after it (they never ran).
            start, results = retry, list(state["results"][:retry])
        for step in plan.steps[start:]:
            res = skill.execute(SkillName(step.skill), step.args)
            results.append(res.model_dump())
            if not res.ok:
                break  # later steps depend on this one; triage decides what next
        final = skill.state()
        return {"stage": "execute", "retry_step": None, "results": results, "final_state": final,
                "transcript": _log(state, f"[execute] ran {len(results)} skills")}

    def verify_node(state: LoopState) -> dict:
        v = verifier.verify(state["user_message"], state["final_state"], state.get("scene"))
        return {"stage": "verify", "verdict": v.model_dump(),
                "transcript": _log(state, f"[verify] success={v.success} :: {v.reason}")}

    def triage_node(state: LoopState) -> dict:
        """Classify any failure and pick the recovery route (stored in state['route'])."""
        stage = state.get("stage", "")
        if triage.underlying_failure(state) is None:
            nxt = {"perceive": "plan", "plan": "execute", "execute": "verify"}.get(stage, "end")
            return {"route": nxt, "failure": None}

        f = triage.classify_failure(state)
        log = _log(state, f"[triage] {f.kind.value}: {f.reason}")
        base: dict = {"failure": f.model_dump()}

        if f.kind is FailureKind.EXHAUSTED:
            return {**base, "route": "escalate", "transcript": log}

        if f.kind is FailureKind.EXECUTION and triage.step_retries_left(state, f):
            retries = dict(state.get("step_retries") or {})
            retries[f.step_index] = retries.get(f.step_index, 0) + 1
            return {**base, "route": "execute", "retry_step": f.step_index,
                    "step_retries": retries,
                    "transcript": [*log, f"[triage] retrying step {f.step_index} "
                                         f"({retries[f.step_index]}/{settings.max_step_retries})"]}

        # Every remaining route spends one unit of the global replan budget.
        base["replans"] = state.get("replans", 0) + 1
        if f.kind is FailureKind.PERCEPTION:
            return {**base, "route": "perceive", "scene": None, "plan": None,
                    "results": [], "verdict": None, "plan_error": "", "transcript": log}
        if f.kind is FailureKind.PLANNING:
            hint = f"The previous plan was invalid. Constraint: {f.reason}"
        elif f.kind is FailureKind.EXECUTION:
            hint = f"A step kept failing after retries. {f.reason}"
        else:  # VERIFICATION
            hint = (state.get("verdict") or {}).get("replan_hint") or f.reason
        return {**base, "route": "plan", "hint": hint, "plan_error": "", "verdict": None,
                "results": [], "retry_step": None, "transcript": log}

    def escalate_node(state: LoopState) -> dict:
        f = Failure.model_validate(state["failure"])
        question = (f"I could not complete \"{state['user_message']}\" after "
                    f"{state.get('replans', 0)} replans. {f.reason} "
                    f"What should I do: adjust the goal or the scene, or try a different approach?")
        return {"escalation": question,
                "transcript": _log(state, f"[escalate] {question}")}

    def after_supervisor(state: LoopState) -> str:
        return "clarify" if state.get("action") == "clarify" else "execute"

    g = StateGraph(LoopState)
    g.add_node("supervisor", traced("supervisor", supervisor_node))
    g.add_node("perceive", traced("perceive", perceive_node))
    g.add_node("plan", traced("plan", plan_node))
    g.add_node("execute", traced("execute", execute_node))
    g.add_node("verify", traced("verify", verify_node))
    g.add_node("triage", traced("triage", triage_node))
    g.add_node("escalate", traced("escalate", escalate_node))

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", after_supervisor,
                            {"clarify": END, "execute": "perceive"})
    for stage_node in ("perceive", "plan", "execute", "verify"):
        g.add_edge(stage_node, "triage")
    g.add_conditional_edges("triage", lambda s: s["route"],
                            {"perceive": "perceive", "plan": "plan", "execute": "execute",
                             "verify": "verify", "escalate": "escalate", "end": END})
    g.add_edge("escalate", END)
    return g.compile()
