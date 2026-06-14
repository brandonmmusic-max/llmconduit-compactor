"""Server/contract tests with a STUBBED extractor (no network).

Verifies the /compact contract end-to-end: under-budget passthrough, over-budget
compaction, system preservation, state-block insertion, real shrinkage, and
incremental folding across calls. Run: pip install fastapi httpx && pytest -q
"""
import compactor.server as srv
from compactor.schema import RecordFact, StateDelta
from fastapi.testclient import TestClient


async def _fake_extract(state, folded, **kw):
    # deterministic: one fact per folded user turn (no LLM)
    ops = [RecordFact(text=f"folded:{m.get('content','')[:24]}")
           for m in folded if m.get("role") == "user"]
    return StateDelta(ops=ops)


def _big(n=40):
    msgs = [{"role": "system", "content": "You are a coding agent."}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"task {i} " + "x" * 1500})
        msgs.append({"role": "assistant", "content": f"did {i} " + "y" * 1500})
    return msgs


def test_under_budget_passthrough():
    srv._SESSIONS.clear()
    c = TestClient(srv.app)
    r = c.post("/compact", json={"session_id": "s1",
                                 "messages": [{"role": "user", "content": "hi"}],
                                 "max_input_tokens": 100_000})
    assert r.json()["compacted"] is False


def test_over_budget_compacts(monkeypatch):
    monkeypatch.setattr(srv, "extract_state_delta", _fake_extract)
    srv._SESSIONS.clear()
    c = TestClient(srv.app)
    msgs = _big(40)
    r = c.post("/compact", json={"session_id": "s2", "messages": msgs,
                                 "max_input_tokens": 1000, "keep_recent_turns": 3})
    body = r.json()
    assert body["compacted"] is True
    out = body["messages"]
    assert out[0]["role"] == "system"                 # system preserved
    assert any(m.get("_compactor") for m in out)      # state block inserted
    assert len(out) < len(msgs)                        # actually shrank
    assert body["ops_applied"] > 0


def test_incremental_folding_grows_state(monkeypatch):
    monkeypatch.setattr(srv, "extract_state_delta", _fake_extract)
    srv._SESSIONS.clear()
    c = TestClient(srv.app)
    r1 = c.post("/compact", json={"session_id": "s3", "messages": _big(30),
                                  "max_input_tokens": 1000, "keep_recent_turns": 3})
    v1 = r1.json()["state_version"]
    r2 = c.post("/compact", json={"session_id": "s3", "messages": _big(60),
                                  "max_input_tokens": 1000, "keep_recent_turns": 3})
    v2 = r2.json()["state_version"]
    assert v2 >= v1 and r2.json()["compacted"] is True   # state advanced
