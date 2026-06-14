"""Pure, deterministic merge of a StateDelta into a CompactState.

Event-sourced: applying the same ops in the same order always yields the same
view (important — the rendered view becomes the upstream prefix, so determinism
keeps the KV cache hitting). De-dup is by stable key (path / id / normalized
text) so re-extracting an already-known fact is a no-op rather than a dup.
"""
from __future__ import annotations

import copy
from typing import List

from .schema import (
    AddArtifact, AddConstraint, CompactState, DecisionItem, FileNote, NoteFile,
    OpenTask, RecordDecision, RecordFact, RecordToolOutcome, ResolveTask,
    SetObjective, StateDelta, TaskItem,
)


def _norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def _dedup_append(items: List[str], value: str, cap: int) -> None:
    key = _norm(value)
    if any(_norm(x) == key for x in items):
        return
    items.append(value)
    # keep most-recent `cap` (bounded view; oldest drop first)
    if len(items) > cap:
        del items[: len(items) - cap]


def merge(state: CompactState, delta: StateDelta, *, caps: dict | None = None) -> CompactState:
    caps = caps or {}
    # Bounded view (keeps the rendered block cache-stable) but generous enough to
    # digest a large document. High-priority items go to typed ops (decisions,
    # constraints, tasks) which are not capped here, so they survive regardless.
    fact_cap = caps.get("facts", 100)
    tool_cap = caps.get("tool_outcomes", 50)
    art_cap = caps.get("artifacts", 60)

    s = state.model_copy(deep=True)

    for op in delta.ops:
        if isinstance(op, SetObjective):
            s.objective = op.text.strip()

        elif isinstance(op, RecordDecision):
            if op.supersedes:
                s.decisions = [d for d in s.decisions if d.id != op.supersedes]
            existing = next((d for d in s.decisions if d.id == op.id), None)
            item = DecisionItem(id=op.id, decision=op.decision, rationale=op.rationale)
            if existing:
                s.decisions[s.decisions.index(existing)] = item
            else:
                s.decisions.append(item)

        elif isinstance(op, NoteFile):
            existing = next((f for f in s.files if f.path == op.path), None)
            item = FileNote(path=op.path, change=op.change, status=op.status)
            if existing:
                s.files[s.files.index(existing)] = item
            else:
                s.files.append(item)

        elif isinstance(op, OpenTask):
            if not any(t.id == op.id for t in s.tasks):
                s.tasks.append(TaskItem(id=op.id, task=op.task, done=False))

        elif isinstance(op, ResolveTask):
            t = next((t for t in s.tasks if t.id == op.id), None)
            if t:
                t.done = True
                t.outcome = op.outcome
            else:
                s.tasks.append(TaskItem(id=op.id, task=op.id, done=True, outcome=op.outcome))

        elif isinstance(op, AddConstraint):
            _dedup_append(s.constraints, op.text.strip(), cap=50)

        elif isinstance(op, RecordFact):
            _dedup_append(s.facts, op.text.strip(), cap=fact_cap)

        elif isinstance(op, RecordToolOutcome):
            mark = "" if op.success is None else ("[ok] " if op.success else "[fail] ")
            _dedup_append(s.tool_outcomes, mark + op.summary.strip(), cap=tool_cap)

        elif isinstance(op, AddArtifact):
            line = op.ref + (f" — {op.note}" if op.note else "")
            _dedup_append(s.artifacts, line, cap=art_cap)

    s.version = state.version + 1
    return s


def replay(session_id: str, deltas: List[StateDelta], *, caps: dict | None = None) -> CompactState:
    """Rebuild a view from an ordered delta log (event-sourcing replay)."""
    s = CompactState.empty(session_id)
    for d in deltas:
        s = merge(s, d, caps=caps)
    return s
