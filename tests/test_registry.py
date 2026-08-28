from cognitive_orchestrator.skills.registry import WorldState, execute
from cognitive_orchestrator.schemas import SkillName

def test_pick_place_flow():
    w = WorldState(locations={"red_cube_1": "table"}, fragile={})
    r = execute(w, SkillName.PICK, {"object_id": "red_cube_1"})
    assert r.ok and w.held == "red_cube_1"
    r2 = execute(w, SkillName.PLACE, {"object_id": "red_cube_1", "location": "left_bin"})
    assert r2.ok and w.locations["red_cube_1"] == "left_bin" and w.held is None

def test_cannot_double_pick():
    w = WorldState(locations={"a": "table", "b": "table"}, fragile={})
    execute(w, SkillName.PICK, {"object_id": "a"})
    r = execute(w, SkillName.PICK, {"object_id": "b"})
    assert not r.ok
