from cognitive_orchestrator.schemas import Scene, Plan, PlanStep, SkillName

def test_scene_roundtrip():
    s = Scene.model_validate({
        "objects": [{"id": "a", "label": "cube", "color": "red",
                     "fragile": False, "position": {"x": 0, "y": 0, "z": 0}}],
        "bins": {"left_bin": "red"},
    })
    assert s.by_id("a").color == "red"
    assert s.by_id("missing") is None

def test_plan_step_enum():
    step = PlanStep(index=0, skill="pick", args={"object_id": "a"})
    assert step.skill is SkillName.PICK
