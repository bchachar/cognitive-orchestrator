"""Symbolic preflight: invalid plans are rejected before any skill executes."""
from cognitive_orchestrator.agents import planner, triage
from cognitive_orchestrator.agents.preflight import validate_plan
from cognitive_orchestrator.orchestration.graph import build_graph
from cognitive_orchestrator.perception.simulator import render_synthetic, to_b64
from cognitive_orchestrator.schemas import FailureKind, Plan, PlanStep, SkillName
from cognitive_orchestrator.skills.client import LocalSkillClient

WORLD = {"locations": {"a": "table", "b": "table"}, "held": None, "fragile": {}}


def _plan(*steps):
    return Plan(goal="g", steps=[PlanStep(index=i, skill=s, args=a)
                                 for i, (s, a) in enumerate(steps)])


def test_valid_plan_passes():
    plan = _plan((SkillName.INSPECT, {"object_id": "a"}),
                 (SkillName.PICK, {"object_id": "a"}),
                 (SkillName.PLACE, {"object_id": "a", "location": "left_bin"}),
                 (SkillName.PICK, {"object_id": "b"}),
                 (SkillName.PLACE, {"object_id": "b", "location": "safe_zone"}))
    assert validate_plan(plan, WORLD) == []
    assert WORLD["held"] is None and WORLD["locations"]["a"] == "table"  # input untouched


def test_two_consecutive_picks_rejected():
    plan = _plan((SkillName.PICK, {"object_id": "a"}), (SkillName.PICK, {"object_id": "b"}))
    v = validate_plan(plan, WORLD)
    assert len(v) == 1 and v[0].startswith("step 1 (pick)")


def test_place_before_pick_rejected_and_nothing_executes(monkeypatch):
    bad = _plan((SkillName.PLACE, {"object_id": "a", "location": "left_bin"}))
    assert validate_plan(bad, WORLD)[0].startswith("step 0 (place)")

    real, calls = planner.plan, []
    monkeypatch.setattr(planner, "plan",
                        lambda g, s, m: bad if not calls.append(g) and len(calls) == 1
                        else real(g, s, m))

    class Spy(LocalSkillClient):
        executed = []

        def execute(self, skill, args):
            Spy.executed.append(skill)
            return super().execute(skill, args)

    png, seed = render_synthetic()
    out = build_graph(Spy()).invoke({"user_message": "Sort the cubes into matching bins.",
                                     "image_b64": to_b64(png), "seed": seed,
                                     "results": [], "transcript": [], "replans": 0})
    assert any("[triage] planning" in l and "step 0 (place)" in l for l in out["transcript"])
    assert "step 0 (place)" in calls[1]  # violation fed back as replan hint
    first_exec = next(i for i, l in enumerate(out["transcript"]) if l.startswith("[execute]"))
    first_rej = next(i for i, l in enumerate(out["transcript"]) if "[plan] rejected" in l)
    assert first_rej < first_exec
    assert SkillName.PLACE not in Spy.executed[:1]
    assert out["verdict"]["success"] is True
