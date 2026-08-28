"""Skill-invocation seam.

The orchestration graph never imports skill implementations directly. It talks to
a SkillClient. Two implementations:

  * LocalSkillClient  -> in-process registry (fast, used by the test suite)
  * HTTPSkillClient    -> calls the FastAPI edge service over the network

Swapping one for the other changes nothing in the agent logic. That is what
"architect robust interface pipelines that allow sub-agents to connect" means in
practice: a stable contract, multiple deployment targets.
"""
from __future__ import annotations

from typing import Protocol

import httpx

from ..schemas import SkillName, SkillResult
from . import registry
from .registry import WorldState


class SkillClient(Protocol):
    def seed(self, locations: dict[str, str], fragile: dict[str, bool]) -> dict: ...
    def state(self) -> dict: ...
    def manifest(self) -> dict[str, str]: ...
    def execute(self, skill: SkillName, args: dict[str, str]) -> SkillResult: ...


class LocalSkillClient:
    def __init__(self) -> None:
        self.world = WorldState()

    def seed(self, locations, fragile):
        self.world.locations = dict(locations)
        self.world.fragile = dict(fragile)
        self.world.held = None
        return self.world.snapshot()

    def state(self):
        return self.world.snapshot()

    def manifest(self):
        return registry.manifest()

    def execute(self, skill, args):
        return registry.execute(self.world, skill, args)


class HTTPSkillClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._c = httpx.Client(timeout=timeout)

    def seed(self, locations, fragile):
        return self._c.post(f"{self.base_url}/seed",
                            json={"locations": locations, "fragile": fragile}).json()

    def state(self):
        return self._c.get(f"{self.base_url}/state").json()

    def manifest(self):
        return self._c.get(f"{self.base_url}/skills").json()

    def execute(self, skill, args):
        r = self._c.post(f"{self.base_url}/execute",
                         json={"skill": skill.value if isinstance(skill, SkillName) else skill,
                               "args": args})
        return SkillResult(**r.json())
