"""HTTP companion implementing llmconduit's external-compactor contract.

  POST /compact
    req : {session_id, messages[], max_input_tokens, keep_recent_turns,
           model?, upstream_base_url?, upstream_api_key?}
    resp: {messages[], compacted: bool, state_version: int, ops_applied: int}

Behavior:
- Below budget -> return messages unchanged (compacted=false).
- Over budget  -> incrementally fold the OLD prefix into a per-session
  CompactState (event-sourced), splice [system + state block + verbatim tail],
  and return that. Best-effort: any failure returns the original messages.

State is in-memory per session + a delta log (so replay() works). For
durability, swap `_SESSIONS` for the KLC sqlite event_log — same delta stream.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .extractor import extract_state_delta
from .merge import merge
from .schema import CompactState, StateDelta
from .transcript import render_state_block, split_system, _tail_start
from .triggers import estimate_tokens

app = FastAPI(title="llmconduit-compactor")

# session_id -> (state, [deltas])
_SESSIONS: Dict[str, tuple[CompactState, List[StateDelta]]] = {}

_DEF_BASE = os.environ.get("COMPACTOR_UPSTREAM_BASE_URL", "http://127.0.0.1:8000/v1")
_DEF_MODEL = os.environ.get("COMPACTOR_MODEL", "")  # default: summarize with the same upstream model
_DEF_KEY = os.environ.get("COMPACTOR_API_KEY") or None


class CompactRequest(BaseModel):
    session_id: str
    messages: List[dict]
    max_input_tokens: int = 96_000
    keep_recent_turns: int = 6
    model: Optional[str] = None
    upstream_base_url: Optional[str] = None
    upstream_api_key: Optional[str] = None


class CompactResponse(BaseModel):
    messages: List[dict]
    compacted: bool
    state_version: int = 0
    ops_applied: int = 0


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(_SESSIONS)}


@app.post("/compact", response_model=CompactResponse)
async def compact(req: CompactRequest) -> CompactResponse:
    msgs = req.messages
    if estimate_tokens(msgs) <= req.max_input_tokens:
        return CompactResponse(messages=msgs, compacted=False)

    state, deltas = _SESSIONS.get(req.session_id, (CompactState.empty(req.session_id), []))

    sys_msgs, body = split_system(msgs)
    fold_point = _tail_start(body, req.keep_recent_turns)  # body[:fold_point] is foldable

    ops_applied = 0
    # incrementally fold only turns past the high-water mark
    new_to_fold = body[state.folded_through + 1 : fold_point]
    if new_to_fold:
        base = req.upstream_base_url or _DEF_BASE
        model = req.model or _DEF_MODEL or _infer_model(req)
        delta = await extract_state_delta(
            state, new_to_fold,
            base_url=base, model=model, api_key=req.upstream_api_key or _DEF_KEY,
        )
        if delta is not None:
            state = merge(state, delta)
            deltas.append(delta)
            state.folded_through = fold_point - 1
            ops_applied = len(delta.ops)
            _SESSIONS[req.session_id] = (state, deltas)

    # if we still have nothing folded (extractor failed), forward original
    if state.version == 0 and not new_to_fold:
        return CompactResponse(messages=msgs, compacted=False)

    tail = body[fold_point:]
    state_msg = {"role": "user", "content": render_state_block(state), "_compactor": True}
    new_msgs = sys_msgs + [state_msg] + tail
    return CompactResponse(
        messages=new_msgs, compacted=True,
        state_version=state.version, ops_applied=ops_applied,
    )


def _infer_model(req: CompactRequest) -> str:
    # use the model named on the last assistant/user turn if the gateway passed it through
    for m in reversed(req.messages):
        if m.get("model"):
            return m["model"]
    return "default"
