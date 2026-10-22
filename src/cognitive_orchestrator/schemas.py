"""Typed contracts exchanged between agents.

Every message that crosses an agent boundary is a validated Pydantic model.
This is deliberate: in an agentic system the failure mode is not a crash, it is
a subtly malformed hand-off that the next agent silently misinterprets. Schema
validation turns those silent failures into loud, catchable ones -- the same
"parsing-failure-as-first-class-failure-mode" idea from LLM evaluation work.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class SceneObject(BaseModel):
    """One object perceived on the table."""
    id: str = Field(..., description="Stable identifier, e.g. 'red_cube_1'")
    label: str = Field(..., description="Human category, e.g. 'cube', 'mug'")
    color: str
    fragile: bool = False
    position: Vec3


class Scene(BaseModel):
    """Structured world state produced by the perception (VLM) agent."""
    objects: list[SceneObject] = Field(default_factory=list)
    bins: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of bin name -> color it accepts, e.g. {'left_bin':'red'}",
    )

    def by_id(self, oid: str) -> Optional[SceneObject]:
        return next((o for o in self.objects if o.id == oid), None)


class SkillName(str, Enum):
    """The robot's skill library == the Neuraverse-style skill registry keys."""
    PICK = "pick"
    PLACE = "place"
    INSPECT = "inspect"


class PlanStep(BaseModel):
    """One high-level step emitted by the planner agent."""
    index: int
    skill: SkillName
    args: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)


class SkillResult(BaseModel):
    """Return value from executing one skill on the (edge) skill service."""
    ok: bool
    skill: SkillName
    args: dict[str, str] = Field(default_factory=dict)
    message: str = ""


class FailureKind(str, Enum):
    """Why the loop failed; each kind has its own recovery route in the graph."""
    PERCEPTION = "perception"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    EXHAUSTED = "exhausted"


class Failure(BaseModel):
    kind: FailureKind
    reason: str
    step_index: Optional[int] = None


class Verdict(BaseModel):
    """Verifier agent output: did we actually satisfy the goal?"""
    success: bool
    reason: str
    replan_hint: str = ""
