"""Thin client over an OpenAI-compatible endpoint (vLLM).

Why OpenAI-compatible: vLLM exposes exactly the /v1/chat/completions contract,
so the same client talks to a 7B on your laptop, a 32B on the A40, or a frontier
API -- swapping models is a config change, not a code change.

MOCK mode: when settings.mock_models is True the client returns deterministic,
schema-valid JSON keyed off a ROLE tag in the system prompt. That lets the whole
cognitive loop (and the test suite) run with zero GPU.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

from .config import settings


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str, mock: bool):
        self.model = model
        self.mock = mock
        self._client = None if mock else OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, system: str, user: str, image_b64: Optional[str] = None) -> str:
        if self.mock:
            return _mock_response(system, user, image_b64)

        content: list | str
        if image_b64:
            content = [
                {"type": "text", "text": user},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        else:
            content = user

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            timeout=settings.request_timeout_s,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, image_b64: Optional[str] = None) -> dict:
        """Call the model and robustly extract the first JSON object.

        Empty / unparseable output is treated as an explicit failure rather than
        being silently coerced -- parsing failure is a first-class failure mode.
        """
        raw = self.chat(system, user, image_b64)
        obj = _extract_json(raw)
        if obj is None:
            raise ValueError(f"Model returned no parseable JSON. Raw head: {raw[:200]!r}")
        return obj


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    # Fast path: the whole thing is already valid JSON (vLLM structured output).
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Slow path: find the first balanced object, ignoring braces inside strings.
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# --------------------------------------------------------------------------- #
# Deterministic mock brain. Keyed on a ROLE: tag placed at the top of each
# agent's system prompt. Good enough to exercise the full graph on CPU.
# --------------------------------------------------------------------------- #
def _mock_response(system: str, user: str, image_b64: Optional[str]) -> str:
    role = ""
    m = re.match(r"\s*ROLE:\s*(\w+)", system)
    if m:
        role = m.group(1)

    if role == "perception":
        return json.dumps({
            "objects": [
                {"id": "red_cube_1", "label": "cube", "color": "red",
                 "fragile": False, "position": {"x": 0.1, "y": 0.0, "z": 0.0}},
                {"id": "blue_cube_1", "label": "cube", "color": "blue",
                 "fragile": False, "position": {"x": 0.2, "y": 0.1, "z": 0.0}},
                {"id": "glass_1", "label": "glass", "color": "clear",
                 "fragile": True, "position": {"x": 0.0, "y": 0.2, "z": 0.0}},
            ],
            "bins": {"left_bin": "red", "right_bin": "blue"},
        })

    if role == "planner":
        # Sort cubes by color into matching bins; set fragile items aside.
        return json.dumps({
            "goal": user[:120],
            "steps": [
                {"index": 0, "skill": "inspect", "args": {"object_id": "glass_1"},
                 "rationale": "Check fragility before touching."},
                {"index": 1, "skill": "place", "args": {"object_id": "glass_1", "location": "safe_zone"},
                 "rationale": "Fragile -> set aside."},
                {"index": 2, "skill": "pick", "args": {"object_id": "red_cube_1"},
                 "rationale": "Grab the red cube."},
                {"index": 3, "skill": "place", "args": {"object_id": "red_cube_1", "location": "left_bin"},
                 "rationale": "Red goes to red bin."},
                {"index": 4, "skill": "pick", "args": {"object_id": "blue_cube_1"},
                 "rationale": "Grab the blue cube."},
                {"index": 5, "skill": "place", "args": {"object_id": "blue_cube_1", "location": "right_bin"},
                 "rationale": "Blue goes to blue bin."},
            ],
        })

    if role == "verifier":
        return json.dumps({"success": True, "reason": "All cubes binned; fragile item set aside.",
                           "replan_hint": ""})

    if role == "supervisor":
        # Router: decide whether the request is actionable or needs clarification.
        if "?" in user and len(user) < 15:
            return json.dumps({"action": "clarify",
                               "message": "Which objects should I focus on?"})
        return json.dumps({"action": "execute", "message": "Understood. Planning now."})

    return json.dumps({"note": "unmocked role", "echo": user[:80]})


# Module-level singletons wired from config.
reasoning_llm = LLMClient(settings.llm_base_url, settings.llm_model,
                          settings.llm_api_key, settings.mock_models)
vision_llm = LLMClient(settings.vlm_base_url, settings.vlm_model,
                       settings.vlm_api_key, settings.mock_models)
