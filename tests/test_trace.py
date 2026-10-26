import json

from cognitive_orchestrator.main import run


def test_traced_mock_run_writes_entry_per_node(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_TRACE", "true")
    monkeypatch.setenv("CO_TRACE_DIR", str(tmp_path))
    out = run("Sort the cubes into matching bins and set fragile items aside.")
    assert "trace" not in out

    (path,) = tmp_path.glob("run-*.json")
    entries = json.loads(path.read_text())["entries"]
    nodes = [e["node"] for e in entries]
    assert nodes == ["supervisor", "perceive", "triage", "plan", "triage",
                     "execute", "triage", "verify", "triage"]
    for e in entries:
        assert e["duration_ms"] >= 0 and e["start"] <= e["end"]
        assert "input" in e and "output" in e
    assert entries[0]["model_calls"][0]["model"]

