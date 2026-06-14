# llmconduit-compactor

An **event-sourced context compactor for AI coding agents** (Codex / Claude Code),
exposed as an **external compactor** that llmconduit (or any gateway) can call.

It's the de-legal-ized lift of KLC's StateDelta-ledger compaction: the generic
mechanism — `ops → merge → materialized view → replace old turns` — tuned for the
state a *coding* agent actually needs, instead of legal case state.

## Why a compactor at the gateway at all? (read this first)
Compaction architecturally belongs in the **harness** (it holds state persistently
and knows task semantics). Use this gateway compactor when you **can't change the
harness**:

| Target | Best layer |
|---|---|
| **Codex** (open source) | harness — port the same `StateDelta` core into it |
| **Claude Code** (compaction is closed/untunable) | **gateway — this** |
| dumb/non-compacting clients | **gateway — this** |

Safety rule baked in: **keep system + the last N turns + all tool_use/tool_result
pairs verbatim; only the OLD prefix folds into a deterministic state block.** That
avoids harness↔model desync and keeps the upstream prefix byte-stable so SGLang's
KV cache still hits.

## How it works
1. Gateway forwards a request; if it's over `max_input_tokens`, it POSTs the chat
   messages to this service.
2. We split `system / body`, compute a tool-pair-safe fold point (keep last N turns),
   and **incrementally** extract a `StateDelta` from only the newly-archived turns
   (`extractor.py`, JSON-schema-guided, evidence-fenced against injection).
3. `merge()` folds it into the per-session `CompactState` (event-sourced; `replay()`
   reproduces it from the delta log).
4. We return `[system] + [rendered state block] + [verbatim tail]`.
5. Any failure → original messages returned unchanged (best-effort, no data loss).

## The contract (what to add to llmconduit)
```
POST /compact
 req : {session_id, messages[], max_input_tokens, keep_recent_turns,
        model?, upstream_base_url?, upstream_api_key?}
 resp: {messages[], compacted: bool, state_version, ops_applied}
```
The Rust side (config + hook + a built-in fallback) is sketched in
[`rust-hook/`](rust-hook/INTEGRATION.md).

## Run
```bash
pip install -r requirements.txt
# Model-agnostic: in the llmconduit flow the compactor summarizes with whatever
# model each request already targets (req.model). The var below is only a
# standalone fallback — point it at ANY OpenAI-compatible server.
COMPACTOR_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1 \
uvicorn compactor.server:app --host 127.0.0.1 --port 4100
```

## Testing
```bash
pip install -r requirements.txt fastapi httpx pytest
python -m pytest -q          # unit + /compact contract tests — NO network
# live end-to-end against your local model (extracts real durable state):
# model-agnostic: --model is auto-detected from the endpoint's /v1/models if omitted
python scripts/smoke_live.py --base http://127.0.0.1:8000/v1
```
- `tests/test_core.py` — merge/replay + tool-pair-safe splice (pydantic only).
- `tests/test_server.py` — the `/compact` contract with a stubbed extractor (no network): under-budget passthrough, over-budget compaction, system preserved, real shrinkage, incremental folding.
- `scripts/smoke_live.py` — builds a long synthetic coding session (decisions, file edits, a failing→passing test, a constraint), runs one real compaction pass, prints before/after token counts + the rendered state block. This is how you sanity-check a candidate model's extraction quality.
- **Full-stack (later):** drop `rust-hook/compaction.rs` into a llmconduit build, point `compaction.endpoint` at this server, and run Claude Code through llmconduit at the local model.

## KLC → generic mapping
| KLC (legal) | here (coding agent) |
|---|---|
| `CaseState` (facts/citations/posture) | `CompactState` (objective/decisions/files/tasks/facts/tool-outcomes/artifacts) |
| `AddFacts`/`AddCitations`/`PostureUpdate`… | `record_fact`/`note_file`/`record_decision`/`record_tool_outcome`… |
| `ComposerClient` / `llm_extractors` | any OpenAI-compatible endpoint (`extractor.py`) |
| sqlite `event_log` | in-memory delta log here; swap back to sqlite for durability |
| citation-completeness safeguard | (port as "preserve marked artifacts/handles" if needed) |

## Status
Runnable skeleton with real merge/replay/splice/extraction + tests. To make it
production-tuned for your stack: (1) persist the delta log (sqlite, like KLC),
(2) tune `keep_recent_turns`/caps per model, (3) optionally port KLC's audits
(task-drift / dropped-handle) as post-merge checks behind the same interface.
