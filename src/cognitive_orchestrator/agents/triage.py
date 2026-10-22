"""Failure triage: turn loop state into a typed Failure so the graph can pick a
recovery route instead of always replanning the whole task.

Rules, in priority order:
  PERCEPTION   zero objects perceived, or the plan references an unseen object
  PLANNING     plan validation raised (state["plan_error"]) / a step is unachievable
  EXECUTION    some SkillResult.ok is false (step_index = index into plan.steps)
  VERIFICATION every skill succeeded but the verifier rejected the world state
  EXHAUSTED    the replan budget is spent and no cheaper recovery remains
"""
from __future__ import annotations

from typing import Optional

from ..config import settings
from ..schemas import Failure, FailureKind


def underlying_failure(state: dict) -> Optional[Failure]:
    """The raw failure visible in state, ignoring budgets. None if nothing failed."""
    scene = state.get("scene")
    if scene is not None:
        objects = scene.get("objects", [])
        if not objects:
            return Failure(kind=FailureKind.PERCEPTION, reason="Perception returned zero objects.")
        seen = {o["id"] for o in objects}
        for pos, step in enumerate((state.get("plan") or {}).get("steps", [])):
            oid = step.get("args", {}).get("object_id")
            if oid and oid not in seen:
                return Failure(
                    kind=FailureKind.PERCEPTION, step_index=pos,
                    reason=f"Plan step {pos} references object {oid!r} that perception did not see.")

    if state.get("plan_error"):
        return Failure(kind=FailureKind.PLANNING, reason=state["plan_error"])

    for pos, res in enumerate(state.get("results") or []):
        if not res.get("ok"):
            return Failure(
                kind=FailureKind.EXECUTION, step_index=pos,
                reason=f"Step {pos} ({res.get('skill')} {res.get('args')}) failed: {res.get('message')}")

    verdict = state.get("verdict")
    if verdict is not None and not verdict.get("success"):
        return Failure(kind=FailureKind.VERIFICATION, reason=verdict.get("reason", ""))
    return None


def step_retries_left(state: dict, failure: Failure) -> bool:
    if failure.kind is not FailureKind.EXECUTION or failure.step_index is None:
        return False
    used = (state.get("step_retries") or {}).get(failure.step_index, 0)
    return used < settings.max_step_retries


def classify_failure(state: dict) -> Failure:
    """Classify the current failure. Raises ValueError if the state shows none."""
    failure = underlying_failure(state)
    if failure is None:
        raise ValueError("classify_failure called on a state with no failure")
    replans = state.get("replans", 0)
    if replans >= settings.max_replans and not step_retries_left(state, failure):
        return Failure(
            kind=FailureKind.EXHAUSTED, step_index=failure.step_index,
            reason=(f"Replan budget spent ({replans}/{settings.max_replans}); "
                    f"last failure [{failure.kind.value}]: {failure.reason}"))
    return failure
