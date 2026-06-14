"""Transcript splice — the safety-critical part.

Rule (from the desync discussion): keep system + the recent tail + every
tool_use/tool_result pair VERBATIM; only the OLD prefix is folded into a single
rendered state block. The kept tail boundary is snapped so we never start the
tail with an orphan `tool` message (a tool result whose assistant tool_call got
folded away) — that would 400 on the upstream and confuse the agent.

Operates on OpenAI Chat Completions messages (what llmconduit forwards
upstream): roles system|user|assistant|tool, optional tool_calls / tool_call_id.
"""
from __future__ import annotations

from typing import List, Tuple

from .schema import CompactState


def split_system(messages: List[dict]) -> Tuple[List[dict], List[dict]]:
    sys_msgs, body = [], []
    for m in messages:
        (sys_msgs if m.get("role") == "system" else body).append(m)
    return sys_msgs, body


def _tail_start(body: List[dict], keep_recent_turns: int) -> int:
    """Index in `body` where the verbatim tail begins.

    A 'turn' starts at a user message. Keep the last `keep_recent_turns` turns,
    then snap earlier so the tail never begins on an orphan tool result and an
    assistant-with-tool_calls is never separated from its tool messages.
    """
    user_idx = [i for i, m in enumerate(body) if m.get("role") == "user"]
    if len(user_idx) <= keep_recent_turns:
        return 0
    cut = user_idx[-keep_recent_turns]

    # snap back over any assistant(tool_calls) whose tool results are in the tail
    # and forward past orphan tool results at the boundary.
    while cut > 0 and body[cut].get("role") == "tool":
        cut -= 1  # pull the producing assistant message into the tail
    # if the message just before cut is assistant w/ tool_calls answered inside
    # the tail, include it too (keep the pair whole)
    if cut > 0:
        prev = body[cut - 1]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            cut -= 1
    return cut


def render_state_block(state: CompactState) -> str:
    L: List[str] = ["# Compacted working context",
                    "(Older turns were summarized into durable state; recent turns are verbatim below.)"]
    if state.objective:
        L.append(f"\n## Objective\n{state.objective}")
    if state.constraints:
        L.append("\n## Constraints / rules (do not violate)")
        L += [f"- {c}" for c in state.constraints]
    if state.decisions:
        L.append("\n## Decisions")
        L += [f"- {d.decision}" + (f" — {d.rationale}" if d.rationale else "") + f"  (#{d.id})"
              for d in state.decisions]
    open_t = [t for t in state.tasks if not t.done]
    done_t = [t for t in state.tasks if t.done]
    if open_t:
        L.append("\n## Open tasks")
        L += [f"- [ ] {t.task}  (#{t.id})" for t in open_t]
    if done_t:
        L.append("\n## Done")
        L += [f"- [x] {t.task}" + (f" ({t.outcome})" if t.outcome else "") + f"  (#{t.id})" for t in done_t]
    if state.files:
        L.append("\n## Files touched")
        L += [f"- `{f.path}` ({f.status}): {f.change}" for f in state.files]
    if state.facts:
        L.append("\n## Environment facts")
        L += [f"- {f}" for f in state.facts]
    if state.tool_outcomes:
        L.append("\n## Key tool outcomes")
        L += [f"- {o}" for o in state.tool_outcomes]
    if state.artifacts:
        L.append("\n## Artifacts produced")
        L += [f"- {a}" for a in state.artifacts]
    return "\n".join(L)


def splice(messages: List[dict], state: CompactState, keep_recent_turns: int):
    """Return (new_messages, folded_body_msgs).

    folded_body_msgs are the OLD turns to feed the extractor (the ones being
    replaced by the state block). new_messages is what to forward upstream:
        [system...] + [state block as a user message] + [verbatim tail]
    """
    sys_msgs, body = split_system(messages)
    cut = _tail_start(body, keep_recent_turns)
    folded, tail = body[:cut], body[cut:]
    if not folded:
        return messages, []  # nothing old enough to fold
    state_msg = {"role": "user",
                 "content": render_state_block(state),
                 "_compactor": True}  # marker; strip if your backend rejects extras
    return sys_msgs + [state_msg] + tail, folded
