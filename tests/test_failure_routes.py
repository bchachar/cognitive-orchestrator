"""One test per recovery route, all in mock mode."""
import pytest

from cognitive_orchestrator.agents import planner, triage, verifier
from cognitive_orchestrator.config import settings
from cognitive_orchestrator.orchestration.graph import build_graph
from cognitive_orchestrator.perception import vlm
from cognitive_orchestrator.perception.simulator import render_synthetic, to_b64
from cognitive_orchestrator.schemas import FailureKind, Scene, SkillName, Verdict
from cognitive_orchestrator.skills.client import LocalSkillClient

CMD = "Sort the cubes into matching bins and set fragile items aside."


def _run(client=None):
    png, seed = render_synthetic()
    g = build_graph(client or LocalSkillClient())
    return g.invoke({"user_message": CMD, "image_b64": to_b64(png), "seed": seed,
                     "results": [], "transcript": [], "replans": 0})


def _kinds(out):
    return [l for l in out["transcript"] if l.startswith("[triage]")]


class FlakySkills(LocalSkillClient):
    """Fails the first `failures` calls of one skill on one object."""

    def __init__(self, skill, object_id, failures):
        super().__init__()
        self.target, self.failures = (skill, object_id), failures
        self.calls = []

    def execute(self, skill, args):
        self.calls.append((skill, args.get("object_id")))
        if (skill, args.get("object_id")) == self.target and self.failures > 0:
            self.failures -= 1
            from cognitive_orchestrator.schemas import SkillResult
            return SkillResult(ok=False, skill=skill, args=args, message="transient fault")
        return super().execute(skill, args)


def test_perception_failure_goes_back_to_perceive(monkeypatch):
    real, calls = vlm.perceive, []

    def flaky(image_b64, known_ids=None):
        calls.append(1)
        return Scene() if len(calls) == 1 else real(image_b64, known_ids)

    monkeypatch.setattr(vlm, "perceive", flaky)
    out = _run()
    assert len(calls) == 2
    assert out["verdict"]["success"] is True
    assert "[triage] perception" in _kinds(out)[0]
    assert out["replans"] == 1


def test_plan_referencing_missing_object_is_perception():
    state = {"scene": {"objects": [{"id": "a"}]}, "replans": 0,
             "plan": {"steps": [{"index": 0, "args": {"object_id": "ghost"}}]}}
    f = triage.classify_failure(state)
    assert f.kind is FailureKind.PERCEPTION and f.step_index == 0


def test_planning_failure_replans_with_constraint_in_hint(monkeypatch):
    real, goals = planner.plan, []

    def flaky(goal, scene, manifest):
        goals.append(goal)
        if len(goals) == 1:
            raise ValueError("place has invalid destination 'red_cube_1'")
        return real(goal, scene, manifest)

    monkeypatch.setattr(planner, "plan", flaky)
    out = _run()
    assert len(goals) == 2
    assert "invalid destination 'red_cube_1'" in goals[1]
    assert out["verdict"]["success"] is True
    assert "[triage] planning" in _kinds(out)[0]


def test_execution_failure_retries_step_without_discarding_plan(monkeypatch):
    real, plans = planner.plan, []
    monkeypatch.setattr(planner, "plan", lambda *a: plans.append(1) or real(*a))
    client = FlakySkills(SkillName.PICK, "red_cube_1", failures=1)
    out = _run(client)
    assert len(plans) == 1                      # plan kept, never regenerated
    assert out["replans"] == 0                  # retry does not spend the replan budget
    assert out["step_retries"] == {2: 1}
    assert client.calls.count((SkillName.PICK, "red_cube_1")) == 2
    assert client.calls.count((SkillName.INSPECT, "glass_1")) == 1  # earlier steps not rerun
    assert len(out["results"]) == 6 and all(r["ok"] for r in out["results"])
    assert out["verdict"]["success"] is True


def test_execution_failure_falls_back_to_replan_after_step_retries(monkeypatch):
    real, plans = planner.plan, []
    monkeypatch.setattr(planner, "plan", lambda *a: plans.append(1) or real(*a))
    # Fails first 1 + max_step_retries calls, so retries run out, then replan succeeds.
    client = FlakySkills(SkillName.PICK, "red_cube_1", failures=1 + settings.max_step_retries)
    out = _run(client)
    assert len(plans) == 2
    assert out["replans"] == 1
    assert out["verdict"]["success"] is True


def test_verification_failure_goes_back_to_plan(monkeypatch):
    real, n = verifier.verify, []

    def reject_once(goal, final_state, scene=None):
        n.append(1)
        if len(n) == 1:
            return Verdict(success=False, reason="bad", replan_hint="move X")
        return real(goal, final_state, scene)

    monkeypatch.setattr(verifier, "verify", reject_once)
    out = _run()
    assert len(n) == 2 and out["replans"] == 1
    assert out["verdict"]["success"] is True
    assert "[triage] verification" in _kinds(out)[0]


def test_exhaustion_reaches_escalate(monkeypatch):
    monkeypatch.setattr(verifier, "verify",
                        lambda *a, **k: Verdict(success=False, reason="never ok", replan_hint="x"))
    out = _run()
    assert out["replans"] == settings.max_replans
    assert out["failure"]["kind"] == FailureKind.EXHAUSTED.value
    assert "never ok" in out["escalation"]
    assert out["transcript"][-1].startswith("[escalate]")


def test_classify_rejects_state_without_failure():
    with pytest.raises(ValueError):
        triage.classify_failure({"scene": {"objects": [{"id": "a"}]}, "replans": 0})
