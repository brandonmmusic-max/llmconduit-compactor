"""Core correctness: merge de-dup + replay determinism + tool-pair-safe splice.
Run: pip install pydantic && python -m pytest -q  (no network needed)."""
from compactor.schema import (
    CompactState, StateDelta, RecordFact, NoteFile, OpenTask, ResolveTask, RecordDecision,
)
from compactor.merge import merge, replay
from compactor.transcript import splice, split_system, _tail_start


def test_merge_dedups_facts():
    s = CompactState.empty("x")
    s = merge(s, StateDelta(ops=[RecordFact(text="Python 3.12"), RecordFact(text="python  3.12")]))
    assert len(s.facts) == 1                     # normalized de-dup
    assert s.version == 1


def test_decision_supersede_and_task_lifecycle():
    s = CompactState.empty("x")
    s = merge(s, StateDelta(ops=[
        RecordDecision(id="d1", decision="use sqlite"),
        OpenTask(id="t1", task="add retry"),
    ]))
    s = merge(s, StateDelta(ops=[
        RecordDecision(id="d2", decision="use postgres", supersedes="d1"),
        ResolveTask(id="t1", outcome="done in db.rs"),
    ]))
    assert [d.decision for d in s.decisions] == ["use postgres"]
    assert s.tasks[0].done and s.tasks[0].outcome == "done in db.rs"


def test_replay_is_deterministic():
    deltas = [
        StateDelta(ops=[NoteFile(path="a.rs", change="add fn")]),
        StateDelta(ops=[NoteFile(path="a.rs", change="fix bug", status="modified")]),
    ]
    s1 = replay("x", deltas)
    s2 = replay("x", deltas)
    assert s1.model_dump() == s2.model_dump()    # same input -> identical view
    assert len(s1.files) == 1 and s1.files[0].change == "fix bug"


def test_splice_keeps_tool_pairs_whole():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        {"role": "assistant", "content": "a2"},
    ]
    state = merge(CompactState.empty("x"), StateDelta(ops=[RecordFact(text="prior")]))
    new_msgs, folded = splice(msgs, state, keep_recent_turns=1)
    # tail must not begin with an orphan tool result
    body = [m for m in new_msgs if m.get("role") != "system"]
    assert body[0].get("_compactor")             # state block first
    roles = [m["role"] for m in body[1:]]
    assert "tool" not in roles or roles[roles.index("tool") - 1] == "assistant"
