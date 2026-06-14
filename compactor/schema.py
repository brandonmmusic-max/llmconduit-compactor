"""Coding-agent state schema (event-sourced, llmconduit-compactor).

This is the de-legal-ized analogue of KLC's CaseState/StateDelta. Instead of
facts/citations/posture, the materialized view tracks the state that actually
matters for a coding agent (Codex / Claude Code):

  objective, decisions(+rationale), files touched, open/done tasks, constraints
  & user rules, environment facts, important tool outcomes, produced artifacts.

The ledger is operations -> merge -> materialized view (CompactState). Old
transcript turns are extracted into a StateDelta, merged into the running
CompactState, and the view replaces those turns. The op set is intentionally
small and stable so merges de-dup cleanly and the rendered view is byte-stable
(cache-friendly).
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field

SCHEMA_VERSION = "compact-state/v1"


# --------------------------------------------------------------------------- #
# Handles — stable references so ops can supersede/resolve prior entries
# without re-stating their whole content.
# --------------------------------------------------------------------------- #
class Handle(BaseModel):
    id: str = Field(..., description="stable slug, e.g. 'task:add-retry' or 'file:src/db.rs'")


# --------------------------------------------------------------------------- #
# StateDelta operations (discriminated union on `op`)
# --------------------------------------------------------------------------- #
class SetObjective(BaseModel):
    op: Literal["set_objective"] = "set_objective"
    text: str = Field(..., description="the overall task the agent is working on")


class RecordDecision(BaseModel):
    op: Literal["record_decision"] = "record_decision"
    id: str = Field(..., description="slug, e.g. 'decision:use-sqlite'")
    decision: str
    rationale: Optional[str] = None
    supersedes: Optional[str] = Field(None, description="id of a prior decision this replaces")


class NoteFile(BaseModel):
    op: Literal["note_file"] = "note_file"
    path: str
    change: str = Field(..., description="what changed / current role of this file")
    status: Literal["created", "modified", "deleted", "read"] = "modified"


class OpenTask(BaseModel):
    op: Literal["open_task"] = "open_task"
    id: str
    task: str


class ResolveTask(BaseModel):
    op: Literal["resolve_task"] = "resolve_task"
    id: str
    outcome: Optional[str] = None


class AddConstraint(BaseModel):
    op: Literal["add_constraint"] = "add_constraint"
    text: str = Field(..., description="hard rule / invariant / user instruction to never violate")


class RecordFact(BaseModel):
    op: Literal["record_fact"] = "record_fact"
    text: str = Field(..., description="durable env fact: versions, paths, commands that work, API shapes")


class RecordToolOutcome(BaseModel):
    op: Literal["record_tool_outcome"] = "record_tool_outcome"
    summary: str = Field(..., description="e.g. 'pytest: 142 passed', 'cargo build: error E0277 in db.rs:88'")
    success: Optional[bool] = None


class AddArtifact(BaseModel):
    op: Literal["add_artifact"] = "add_artifact"
    ref: str = Field(..., description="produced thing: file path, PR url, endpoint, image tag")
    note: Optional[str] = None


StateDeltaOp = Annotated[
    Union[
        SetObjective, RecordDecision, NoteFile, OpenTask, ResolveTask,
        AddConstraint, RecordFact, RecordToolOutcome, AddArtifact,
    ],
    Field(discriminator="op"),
]


class StateDelta(BaseModel):
    """A patch extracted from a window of transcript turns."""
    schema_version: Literal["compact-state/v1"] = SCHEMA_VERSION
    ops: List[StateDeltaOp] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Materialized view
# --------------------------------------------------------------------------- #
class FileNote(BaseModel):
    path: str
    change: str
    status: str = "modified"


class TaskItem(BaseModel):
    id: str
    task: str
    done: bool = False
    outcome: Optional[str] = None


class DecisionItem(BaseModel):
    id: str
    decision: str
    rationale: Optional[str] = None


class CompactState(BaseModel):
    schema_version: Literal["compact-state/v1"] = SCHEMA_VERSION
    session_id: str
    version: int = 0
    objective: Optional[str] = None
    decisions: List[DecisionItem] = Field(default_factory=list)
    files: List[FileNote] = Field(default_factory=list)
    tasks: List[TaskItem] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    tool_outcomes: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    # high-water mark: index of the last transcript turn already folded in.
    folded_through: int = -1

    @staticmethod
    def empty(session_id: str) -> "CompactState":
        return CompactState(session_id=session_id)
