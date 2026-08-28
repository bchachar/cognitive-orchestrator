"""End-to-end entrypoint.

    python -m cognitive_orchestrator.main "sort the cubes into matching bins and set fragile items aside"

With CO_MOCK_MODELS=true (default) it runs on CPU with the mock brain.
Set CO_MOCK_MODELS=false and point CO_LLM_BASE_URL / CO_VLM_BASE_URL at your vLLM
servers on the A40 node to run it for real.
"""
from __future__ import annotations

import sys

from .config import settings
from .orchestration.graph import build_graph
from .perception.simulator import render_synthetic, to_b64
from .skills.client import HTTPSkillClient, LocalSkillClient


def make_skill_client():
    # Use the HTTP edge service if it is configured & reachable; else in-process.
    if settings.skill_service_url and not settings.mock_models:
        try:
            c = HTTPSkillClient(settings.skill_service_url)
            c.manifest()  # ping
            return c
        except Exception:
            pass
    return LocalSkillClient()


def run(command: str) -> dict:
    png, seed = render_synthetic()
    graph = build_graph(make_skill_client())
    init = {"user_message": command, "image_b64": to_b64(png), "seed": seed,
            "results": [], "transcript": [], "replans": 0}
    return graph.invoke(init)


def main() -> None:
    command = (sys.argv[1] if len(sys.argv) > 1
               else "Sort the cubes into their matching colored bins and set any "
                    "fragile objects aside in the safe zone.")
    print(f"mock_models={settings.mock_models}\ncommand={command!r}\n")
    final = run(command)
    print("--- transcript ---")
    for line in final.get("transcript", []):
        print(" ", line)
    print("\n--- final world state ---")
    print(" ", final.get("final_state"))
    print("\n--- verdict ---")
    print(" ", final.get("verdict"))


if __name__ == "__main__":
    main()
