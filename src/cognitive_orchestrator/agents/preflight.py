"""Symbolic preflight: reject plans whose preconditions can't hold, before anything runs.

Walks the plan against a simulated *copy* of the world state, applying each
step's declared effects. Nothing real is executed.
"""
from __future__ import annotations

import copy

from ..schemas import Plan
from ..skills import registry


def _resolve(value, args: dict):
    if isinstance(value, str) and value.startswith("$"):
        return args.get(value[1:])
    return value


def _check(key: str, ops: dict, state: dict, args: dict) -> list[str]:
    actual, out = state.get(key), []
    for op, operand in ops.items():
        want = _resolve(operand, args)
        if op == "is":
            ok, text = actual is want, f"{key} is {want!r}"
        elif op == "eq":
            ok, text = actual == want, f"{key} == {want!r}"
        elif op == "contains":
            ok, text = want in (actual or {}), f"{want!r} in {key}"
        else:
            raise ValueError(f"unknown precondition op {op!r}")
        if not ok:
            out.append(f"{text} (actual {key}={actual!r})")
    return out


def _apply(effects: dict, state: dict, args: dict) -> None:
    for key, val in effects.items():
        if isinstance(val, dict):
            state.setdefault(key, {}).update(
                {_resolve(k, args): _resolve(v, args) for k, v in val.items()})
        else:
            state[key] = _resolve(val, args)


def validate_plan(plan: Plan, world_state: dict) -> list[str]:
    """Return violated preconditions as 'step N (skill): ...' strings; [] if valid."""
    state = copy.deepcopy(world_state)
    state.setdefault("held", None)
    state.setdefault("locations", {})
    violations: list[str] = []
    for i, step in enumerate(plan.steps):
        sp = registry.spec(step.skill)
        if sp is None:
            continue
        failed = [m for key, ops in sp.preconditions.items()
                  for m in _check(key, ops, state, step.args)]
        if failed:
            violations += [f"step {i} ({step.skill.value}): requires {m}" for m in failed]
            continue  # effects of an invalid step are not applied
        _apply(sp.effects, state, step.args)
    return violations
