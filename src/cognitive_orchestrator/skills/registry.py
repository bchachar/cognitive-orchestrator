"""The robot's skill library == a Neuraverse-style skill registry.

Each skill is a small, independently testable unit with a declared signature.
The registry can emit a *manifest* the planner reads, so adding a new skill does
not require touching the planner -- new capabilities are discovered, not hard-coded.
That decoupling is the whole point of a skill marketplace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..schemas import SkillName, SkillResult


@dataclass
class WorldState:
    """Authoritative state that skills mutate. Lives on the edge (the robot)."""
    # object_id -> location ("table", "left_bin", "safe_zone", "gripper", ...)
    locations: dict[str, str] = field(default_factory=dict)
    fragile: dict[str, bool] = field(default_factory=dict)
    held: str | None = None

    def snapshot(self) -> dict:
        return {"locations": dict(self.locations), "held": self.held,
                "fragile": dict(self.fragile)}


SkillFn = Callable[[WorldState, dict], SkillResult]

_REGISTRY: dict[SkillName, SkillFn] = {}
_MANIFEST: dict[SkillName, str] = {}


@dataclass(frozen=True)
class SkillSpec:
    """Declarative contract over world-state keys; read by agents.preflight.

    preconditions: {state_key: {op: operand}} with ops "is" (identity, e.g. None),
        "eq", and "contains" (key membership for dict-valued state). Operands
        starting with "$" resolve to the step arg of that name.
    effects: {state_key: value} assigns the key; for dict-valued state
        {state_key: {k: v}} merges entries. "$name" values resolve like above.
    """
    preconditions: dict = field(default_factory=dict)
    effects: dict = field(default_factory=dict)


_SPECS: dict[SkillName, SkillSpec] = {}


def skill(name: SkillName, doc: str, preconditions: dict | None = None,
          effects: dict | None = None):
    def deco(fn: SkillFn) -> SkillFn:
        _REGISTRY[name] = fn
        _MANIFEST[name] = doc
        _SPECS[name] = SkillSpec(preconditions or {}, effects or {})
        return fn
    return deco


def spec(name: SkillName) -> SkillSpec | None:
    return _SPECS.get(name)


@skill(SkillName.INSPECT, "inspect(object_id) -> report an object's fragility/attributes")
def _inspect(world: WorldState, args: dict) -> SkillResult:
    oid = args.get("object_id", "")
    frag = world.fragile.get(oid, False)
    return SkillResult(ok=True, skill=SkillName.INSPECT, args=args,
                       message=f"{oid} fragile={frag}")


@skill(SkillName.PICK, "pick(object_id) -> grasp an object; fails if hands are full",
       preconditions={"held": {"is": None}, "locations": {"contains": "$object_id"}},
       effects={"held": "$object_id", "locations": {"$object_id": "gripper"}})
def _pick(world: WorldState, args: dict) -> SkillResult:
    oid = args.get("object_id", "")
    if world.held is not None:
        return SkillResult(ok=False, skill=SkillName.PICK, args=args,
                           message=f"Gripper already holding {world.held}")
    if oid not in world.locations:
        return SkillResult(ok=False, skill=SkillName.PICK, args=args,
                           message=f"Unknown object {oid}")
    world.held = oid
    world.locations[oid] = "gripper"
    return SkillResult(ok=True, skill=SkillName.PICK, args=args, message=f"Holding {oid}")


@skill(SkillName.PLACE, "place(object_id, location) -> put held/known object at a location",
       preconditions={"held": {"eq": "$object_id"}},
       effects={"held": None, "locations": {"$object_id": "$location"}})
def _place(world: WorldState, args: dict) -> SkillResult:
    oid = args.get("object_id", "")
    loc = args.get("location", "")
    if not loc:
        return SkillResult(ok=False, skill=SkillName.PLACE, args=args,
                           message="No location given")
    if oid not in world.locations:
        return SkillResult(ok=False, skill=SkillName.PLACE, args=args,
                           message=f"Unknown object {oid!r}; cannot place a phantom object")
    world.locations[oid] = loc
    if world.held == oid:
        world.held = None
    return SkillResult(ok=True, skill=SkillName.PLACE, args=args,
                       message=f"{oid} -> {loc}")


def execute(world: WorldState, name: SkillName, args: dict) -> SkillResult:
    fn = _REGISTRY.get(name)
    if fn is None:
        return SkillResult(ok=False, skill=name, args=args, message="No such skill")
    return fn(world, args)


def manifest() -> dict[str, str]:
    return {k.value: v for k, v in _MANIFEST.items()}
