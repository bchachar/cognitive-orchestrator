"""Planner agent: high-level task planning.

Given the goal (natural language) + the perceived Scene + the skill manifest,
decompose intent into an ordered list of skill calls. This is the "translate
human intent into actionable, step-by-step logic" requirement. The plan is
schema-validated, so a step referencing a nonexistent skill fails loudly.
"""
from __future__ import annotations

from ..llm import reasoning_llm
from ..schemas import Plan, Scene

# Locations that a 'place' step is allowed to target.
_VALID_PLACE_DESTINATIONS = frozenset({"left_bin", "right_bin", "safe_zone", "table"})

_SYSTEM = """ROLE: planner
You are the task planner for a tabletop robot. You are given a GOAL, the current
SCENE (objects with colors/fragility and available bins), and the SKILLS you may
use. Output ONLY a JSON object:
{"goal": "<restate>", "steps": [
  {"index": 0, "skill": "<pick|place|inspect>",
   "args": {"object_id": "...", "location": "..."}, "rationale": "..."}]}
Rules:
- inspect fragile-looking items first; never leave the gripper full between
  unrelated objects; fragile items go to 'safe_zone', colored cubes go to the bin
  whose color matches.
- Preconditions are checked before execution: 'pick' needs an empty gripper and a
  known object; 'place' needs that same object to be held (pick it first).
- For EVERY 'place' step the "location" field MUST be one of the fixed location
  names: left_bin, right_bin, safe_zone, table. It must NEVER be set to an
  object id (e.g. "red_cube_1" is NOT a valid location). Using an object id as a
  location will cause a hard runtime error.
JSON only, no prose."""


def plan(goal: str, scene: Scene, manifest: dict[str, str]) -> Plan:
    user = (
        f"GOAL:\n{goal}\n\n"
        f"SCENE:\n{scene.model_dump_json(indent=2)}\n\n"
        f"SKILLS:\n{manifest}"
    )
    data = reasoning_llm.chat_json(_SYSTEM, user)
    validated_plan = Plan.model_validate(data)
    _validate_place_destinations(validated_plan)
    return validated_plan


def _validate_place_destinations(plan: Plan) -> None:
    """Raise ValueError for any place step whose destination is not a known location."""
    for step in plan.steps:
        if step.skill.value == "place":
            location = step.args.get("location", "")
            if location not in _VALID_PLACE_DESTINATIONS:
                raise ValueError(
                    f"Plan step {step.index}: 'place' has invalid destination "
                    f"'{location}'. Must be one of: "
                    f"{sorted(_VALID_PLACE_DESTINATIONS)}."
                )
