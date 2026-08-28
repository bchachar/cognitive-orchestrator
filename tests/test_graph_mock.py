from cognitive_orchestrator.orchestration.graph import build_graph
from cognitive_orchestrator.skills.client import LocalSkillClient
from cognitive_orchestrator.perception.simulator import render_synthetic, to_b64

def _run(cmd):
    png, seed = render_synthetic()
    g = build_graph(LocalSkillClient())
    return g.invoke({"user_message": cmd, "image_b64": to_b64(png), "seed": seed,
                     "results": [], "transcript": [], "replans": 0})

def test_end_to_end_executes_and_verifies():
    out = _run("Sort the cubes into matching bins and set fragile items aside.")
    assert out["verdict"]["success"] is True
    locs = out["final_state"]["locations"]
    assert locs["red_cube_1"] == "left_bin"
    assert locs["blue_cube_1"] == "right_bin"
    assert locs["glass_1"] == "safe_zone"
    assert len(out["results"]) == 6

def test_clarify_short_circuits():
    out = _run("?")
    assert out["action"] == "clarify"
    assert "scene" not in out  # loop never entered perception
