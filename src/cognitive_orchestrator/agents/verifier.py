"""Verifier agent: closes the cognitive loop.

After execution, deterministically check whether the structured tabletop goal
was achieved. Rules:
  - Fragile objects must be in 'safe_zone'.
  - Non-fragile objects must be in the bin whose accepted color matches the
    object's color (derived from the scene).
  - Any object still on the 'table' is unplaced and counts as a failure.

Deterministic verification eliminates the LLM hallucination risk that plagued
the previous prompt-based approach for this structured task.
"""
from __future__ import annotations

from ..schemas import Verdict


def verify(goal: str, final_state: dict, scene: dict | None = None) -> Verdict:
    locations: dict[str, str] = final_state.get("locations", {})
    fragile_map: dict[str, bool] = final_state.get("fragile", {})

    # id -> color, from scene objects
    id_to_color: dict[str, str] = {}
    if scene:
        for obj in scene.get("objects", []):
            id_to_color[obj["id"]] = obj.get("color", "")

    # color -> bin_name (skip 'safe_zone' entry whose value is 'fragile')
    bins: dict[str, str] = (scene or {}).get("bins", {})
    color_to_bin: dict[str, str] = {
        color: bin_name for bin_name, color in bins.items() if color != "fragile"
    }

    failures: list[str] = []
    for oid, loc in locations.items():
        is_fragile = fragile_map.get(oid, False)
        if is_fragile:
            if loc != "safe_zone":
                failures.append(
                    f"{oid} is fragile but is in '{loc}', expected 'safe_zone'"
                )
        else:
            color = id_to_color.get(oid, "")
            expected_bin = color_to_bin.get(color, "")
            if expected_bin and loc != expected_bin:
                failures.append(
                    f"{oid} (color={color!r}) is in '{loc}', expected '{expected_bin}'"
                )
            elif loc == "table":
                failures.append(f"{oid} is still on the table (unplaced)")

    if failures:
        reason = "; ".join(failures)
        return Verdict(
            success=False,
            reason=reason,
            replan_hint="Fix the following placement errors: " + reason,
        )
    return Verdict(success=True, reason="All objects are correctly placed.", replan_hint="")
