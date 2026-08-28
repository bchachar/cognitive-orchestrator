"""Edge skill-service.

This FastAPI app is the piece that conceptually runs *on the robot*: it owns the
world state and executes low-latency skills. The reasoning agents (planner,
verifier, VLM) run server-side / in the cloud and reach skills only through this
HTTP boundary. That is the edge-to-cloud hybrid split, made explicit as an API.

Run it:  uvicorn cognitive_orchestrator.skills.service:app --port 8080
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..schemas import SkillName, SkillResult
from . import registry
from .registry import WorldState

app = FastAPI(title="NEURA-style Edge Skill Service", version="0.1.0")

# In a real robot this is bound to hardware; here it is a process-local world.
WORLD = WorldState()


class SkillCall(BaseModel):
    skill: SkillName
    args: dict[str, str] = {}


class SeedRequest(BaseModel):
    locations: dict[str, str] = {}
    fragile: dict[str, bool] = {}


@app.post("/seed")
def seed(req: SeedRequest) -> dict:
    """Initialise the world (stand-in for the robot booting into a real scene)."""
    WORLD.locations = dict(req.locations)
    WORLD.fragile = dict(req.fragile)
    WORLD.held = None
    return WORLD.snapshot()


@app.get("/skills")
def skills() -> dict[str, str]:
    return registry.manifest()


@app.get("/state")
def state() -> dict:
    return WORLD.snapshot()


@app.post("/execute", response_model=SkillResult)
def execute(call: SkillCall) -> SkillResult:
    return registry.execute(WORLD, call.skill, call.args)
