"""Perception agent (Vision-Language Model).

Consumes a camera image (multimodal input) and emits a validated Scene. This is
the "translate multimodal inputs into contextual reasoning" edge of the loop.
The VLM output is forced through the Scene schema, so a hallucinated or malformed
description is caught here rather than corrupting the planner downstream.
"""
from __future__ import annotations

from ..llm import vision_llm
from ..schemas import Scene

# Fixed, known bin locations — never inferred from the image.
_FIXED_BINS: dict[str, str] = {
    "left_bin": "red",
    "right_bin": "blue",
    "safe_zone": "fragile",
}

_SYSTEM = """ROLE: perception
You are a robot's visual perception system. Look at the tabletop image and return
ONLY a JSON object with this shape:
{
  "objects": [
    {"id": "<stable_id>", "label": "<category>", "color": "<color>",
     "fragile": <true|false>, "position": {"x": <float>, "y": <float>, "z": <float>}}
  ]
}

IMPORTANT — BIN RULES:
- Bins (left_bin, right_bin, safe_zone) are fixed infrastructure, NOT objects on the
  table. Do NOT include them in the "objects" list.
- Do NOT return a "bins" key — the bin layout is managed by the system, not the image.
- Only list movable physical objects (cubes, mugs, glass items, etc.) in "objects".
Mark glass / ceramic / thin items as fragile. Use short snake_case ids like
"red_cube_1". Do not add commentary. JSON only."""


def perceive(image_b64: str, known_ids: list[str] | None = None) -> Scene:
    system = _SYSTEM
    if known_ids:
        id_list = ", ".join(known_ids)
        system = (
            system
            + f"\n\nCANONICAL OBJECT IDs: The objects in this scene have these exact IDs: "
            f"{id_list}. You MUST use these exact IDs in your response — do NOT invent "
            f"new identifiers. Assign each visible object to one of these IDs."
        )
    data = vision_llm.chat_json(system, "Describe the current tabletop scene.", image_b64)
    # Bins are always the fixed set — never taken from VLM output.
    data["bins"] = _FIXED_BINS
    return Scene.model_validate(data)
