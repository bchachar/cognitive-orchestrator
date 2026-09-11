"""The cognitive loop, as a LangGraph state machine.

    supervisor ──clarify──▶ END
        │execute
        ▼
    perceive (VLM) ──▶ plan ──▶ execute (skills) ──▶ verify
                         ▲                              │
                         └────────── replan ◀──────────┘  (bounded)
                                                        │success / budget spent
                                                        ▼
                                                       END

This is the "end-to-end cognitive loop: multimodal input -> contextual reasoning
-> action planning -> skill orchestration -> feedback" from the job description,
expressed as an explicit, inspectable graph rather than a prompt-chain black box.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..agents import planner, supervisor, verifier
from ..config import settings
from ..perception import vlm
from ..schemas import Plan, Scene, SkillName
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
    replans: int
    action: str
    transcript: list[str]


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
        return {"scene": scene.model_dump(),
                "transcript": _log(state, f"[perceive] {len(scene.objects)} objects seen")}

    def plan_node(state: LoopState) -> dict:
        scene = Scene.model_validate(state["scene"])
        goal = state["user_message"]
        # If we already have a verdict, this plan pass is a replan -> spend a credit.
        is_replan = state.get("verdict") is not None
        replans = state.get("replans", 0) + (1 if is_replan else 0)
        hint = (state.get("verdict") or {}).get("replan_hint", "")
        if hint:
            goal = f"{goal}\n(Previous attempt failed. Fix: {hint})"
        p: Plan = planner.plan(goal, scene, skill.manifest())
        return {"plan": p.model_dump(), "replans": replans,
                "transcript": _log(state, f"[plan] {len(p.steps)} steps (replans={replans})")}

    def execute_node(state: LoopState) -> dict:
        # Re-seed the edge world each pass so replanning is idempotent.
        seed = state["seed"]
        skill.seed(seed["locations"], seed["fragile"])
        results: list[dict] = []
        plan = Plan.model_validate(state["plan"])
        for step in plan.steps:
            res = skill.execute(SkillName(step.skill), step.args)
            results.append(res.model_dump())
        final = skill.state()
        return {"results": results, "final_state": final,
                "transcript": _log(state, f"[execute] ran {len(results)} skills")}

    def verify_node(state: LoopState) -> dict:
        v = verifier.verify(state["user_message"], state["final_state"], state.get("scene"))
        return {"verdict": v.model_dump(),
                "transcript": _log(state, f"[verify] success={v.success} :: {v.reason}")}

    def after_supervisor(state: LoopState) -> str:
        return "clarify" if state.get("action") == "clarify" else "execute"

    def after_verify(state: LoopState) -> str:
        v = state.get("verdict") or {}
        if v.get("success") or state.get("replans", 0) >= settings.max_replans:
            return "end"
        return "replan"

    g = StateGraph(LoopState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("perceive", perceive_node)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", after_supervisor,
                            {"clarify": END, "execute": "perceive"})
    g.add_edge("perceive", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges("verify", after_verify, {"end": END, "replan": "plan"})
    return g.compile()
