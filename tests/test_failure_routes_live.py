"""Failure routes against REAL models (vLLM). Skipped in mock mode.

Real models are not deterministic, so faults are injected around them and the
assertions are about routing, not about exact plans. Run with CO_MOCK_MODELS=false.
"""
import pytest

from cognitive_orchestrator.agents import planner, verifier
from cognitive_orchestrator.config import settings
from cognitive_orchestrator.orchestration.graph import build_graph
from cognitive_orchestrator.perception import vlm
from cognitive_orchestrator.perception.simulator import render_synthetic, to_b64
from cognitive_orchestrator.schemas import FailureKind, Scene, SkillName, SkillResult, Verdict
from cognitive_orchestrator.skills.client import LocalSkillClient

pytestmark = pytest.mark.skipif(settings.mock_models, reason="needs live models (CO_MOCK_MODELS=false)")

CMD = "Sort the cubes into matching bins and set fragile items aside."


def _run(client=None):
    png, seed = render_synthetic()
    g = build_graph(client or LocalSkillClient())
    return g.invoke({"user_message": CMD, "image_b64": to_b64(png), "seed": seed,
                     "results": [], "transcript": [], "replans": 0})


def _triage(out):
    return [l for l in out["transcript"] if l.startswith("[triage]")]


def _terminated_cleanly(out):
    assert out["replans"] <= settings.max_replans
    assert out["verdict"] is not None and out["verdict"]["success"] or out.get("escalation")


def test_live_baseline():
    out = _run()
    assert out["verdict"]["success"] is True
    assert not _triage(out)


def test_live_perception_route(monkeypatch):
    real, calls = vlm.perceive, []

    def empty_once(image_b64, known_ids=None):
        calls.append(1)
        return Scene() if len(calls) == 1 else real(image_b64, known_ids)

    monkeypatch.setattr(vlm, "perceive", empty_once)
    out = _run()
    assert len(calls) >= 2
    assert "[triage] perception" in _triage(out)[0]
    _terminated_cleanly(out)


def test_live_planning_route(monkeypatch):
    real, goals = planner.plan, []

    def reject_once(goal, scene, manifest):
        goals.append(goal)
        if len(goals) == 1:
            raise ValueError("place has invalid destination 'red_cube_1'")
        return real(goal, scene, manifest)

    monkeypatch.setattr(planner, "plan", reject_once)
    out = _run()
    assert len(goals) >= 2
    assert "invalid destination 'red_cube_1'" in goals[1]
    assert "[triage] planning" in _triage(out)[0]
    _terminated_cleanly(out)


def test_live_execution_retries_step_keeping_plan(monkeypatch):
    real, plans = planner.plan, []
    monkeypatch.setattr(planner, "plan", lambda *a: plans.append(1) or real(*a))

    class Flaky(LocalSkillClient):
        """Fails the first skill call, whatever the real plan starts with."""
        failed_step = None

        def execute(self, skill, args):
            if Flaky.failed_step is None:
                Flaky.failed_step = (skill, args.get("object_id"))
                return SkillResult(ok=False, skill=skill, args=args, message="transient fault")
            return super().execute(skill, args)

    out = _run(Flaky())
    assert Flaky.failed_step is not None, "plan had no steps"
    assert len(plans) == 1, "plan was regenerated instead of retrying the step"
    assert out["step_retries"] == {0: 1}
    assert out["replans"] == 0
    assert "[triage] execution" in _triage(out)[0]
    _terminated_cleanly(out)


def test_live_verification_route(monkeypatch):
    real, n = verifier.verify, []

    def reject_once(goal, final_state, scene=None):
        n.append(1)
        if len(n) == 1:
            return Verdict(success=False, reason="forced rejection", replan_hint="redo placements")
        return real(goal, final_state, scene)

    monkeypatch.setattr(verifier, "verify", reject_once)
    out = _run()
    assert len(n) >= 2 and out["replans"] >= 1
    assert "[triage] verification" in _triage(out)[0]
    _terminated_cleanly(out)


def test_live_exhaustion_reaches_escalate(monkeypatch):
    monkeypatch.setattr(verifier, "verify",
                        lambda *a, **k: Verdict(success=False, reason="never ok", replan_hint="x"))
    out = _run()
    assert out["replans"] == settings.max_replans
    assert out["failure"]["kind"] == FailureKind.EXHAUSTED.value
    assert "never ok" in out["escalation"]
    assert out["transcript"][-1].startswith("[escalate]")
