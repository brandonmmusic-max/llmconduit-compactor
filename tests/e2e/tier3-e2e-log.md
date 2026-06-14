# llmconduit context compaction — Tier 3 end-to-end test log

**Date:** 2026-06-14  
**Gateway:** `llmconduit` (fork `brandonmmusic-max/llmconduit`, branch `feat/context-compaction`) — release binary, real HTTP path  
**Companion:** `brandonmmusic-max/llmconduit-compactor` — FastAPI `/compact`, in-memory event-sourced StateDelta ledger  
**Upstream model:** MiniMax-M3-NVFP4 (`brandonmusic/MiniMax-M3-NVFP4`), served on `127.0.0.1:9211` (vLLM/b12x, TP4 RTX PRO 6000)  

This is a real run through the actual gateway binary and a real local model — not a mock. Each case sends an over-budget transcript to llmconduit's `/v1/chat/completions`; the gateway's compaction hook detects it is over budget, calls the companion's `/compact`, and forwards the **compacted** messages to M3. Ground truth for "what M3 received" is llmconduit's own upstream request log (`upstream_request_log_path`), not the companion's self-report.

## Verdict

- **Compaction fired in both cases:** PASS
- **Fact preservation:** PASS — Case A 7/7, Case B 7/7
- **Tool-pair safety (no orphan tool messages forwarded):** PASS
- **Nothing lost across compaction:** the post-compaction model answers correctly recall every fact that was folded out of the verbatim transcript (files, pool size, retry policy + numbers, the off-by-one bug + fix, and the final test count).

## Test design (why this proves preservation)

A coding session is simulated where the load-bearing facts are stated in the **early** turns, then buried under bulky low-value turns (file dumps, `cargo test` output). The transcript ends with a mundane turn + a final question asking for a PR summary. Compaction keeps the last `keep_recent_turns=2` turns **verbatim** and folds everything older into a single state block.

The test is rigged so **every** load-bearing fact lives only in the folded region — the kept verbatim tail is deliberately mundane (a version-comment bump + the question) and contains none of the facts and no file paths. Therefore a correct final answer can **only** come from the compacted state block. The facts checked:

- files changed: `src/db.rs`, `src/config.rs`
- pooling/retry decision: pool size **32**, **max 5** attempts, exponential backoff **50ms → 2s**, full jitter
- the bug: off-by-one in the retry counter (`<=` should be `<`)
- final test result: **142 passed**

`max_input_tokens=2000`, `keep_recent_turns=2` for the test (production defaults are 96000 / 6).

---

## Case A — pure chat

### Before → after

| metric | value |
|---|---|
| input messages | 10 |
| input size | 22,454 chars (~5,613 tokens) |
| compaction triggered | True (state v1, 11 ops applied) |
| **messages forwarded to M3** | **5** (roles: system, user, user, assistant, user) |
| **size forwarded to M3** | **1,857 chars** (~92% smaller) |
| state block present in forwarded msgs | True |
| e2e latency | 9.69s |

### State block M3 actually received (verbatim, replaces the folded turns)

```markdown
# Compacted working context
(Older turns were summarized into durable state; recent turns are verbatim below.)

## Objective
Refactor DB layer in `orders`: add connection pool and retry-with-backoff for transient connection errors. Pool must cover p99 ~30 concurrent checkouts; retries must not hammer DB; public `Db` trait signature must not change.

## Constraints / rules (do not violate)
- Public `Db` trait signature must remain unchanged
- Retries must not hammer the DB (use jittered exponential backoff)
- Pool size must cover p99 ~30 concurrent checkouts

## Decisions
- Pool size = 32 (covers p99 ~30 with headroom)
- Max 5 attempts, exponential backoff 50ms->2s with full jitter
- Use `<` (not `<=`) in retry loop to cap at exactly 5 attempts; off-by-one fix after first test run

## Done
- [x] implement_pool_and_retry (Implemented in src/db.rs and src/config.rs; all 142 tests pass)

## Files touched
- `src/db.rs` (modified): Added Pool (size 32) and retry-with-backoff wrapper around connect(); public Db trait unchanged
- `src/config.rs` (modified): Added pool_size=32 and retry config (max_attempts=5, base_delay_ms=50, max_delay_ms=2000)

## Environment facts
- orders crate version: v0.3.9
- Test suite: 142 tests (case_000..case_141), all passing after retry-loop fix

## Key tool outcomes
- cargo test orders v0.3.9: first run FAILED due to off-by-one in retry loop (6 attempts instead of 5); fixed `<=` -> `<` in src/db.rs; re-run ALL 142 TESTS PASS
```

### M3's answer (post-compaction)

```
## PR Summary

**Files changed**
- `src/db.rs` — added Pool (size 32) and retry-with-backoff wrapper around `connect()`; public `Db` trait signature unchanged
- `src/config.rs` — added `pool_size=32` and retry config (`max_attempts=5`, `base_delay_ms=50`, `max_delay_ms=2000`)

**Key decisions**
- **Pool size: 32** — covers p99 ~30 concurrent checkouts with headroom
- **Retries: 5 attempts max**, exponential backoff **50ms → 2s with full jitter** — prevents hammering the DB on transient connection errors
- Off-by-one fix: `<=` → `<` in retry loop to cap at exactly 5 attempts

**Test result**
- `cargo test orders v0.3.9`: first run failed (6 attempts instead of 5); after fix, **all 142 tests pass**
```

### Preservation checks

| fact (folded out of the verbatim transcript) | recalled in answer |
|---|---|
| files src/db.rs + src/config.rs | PASS |
| pool size 32 | PASS |
| max 5 retries | PASS |
| backoff 50ms | PASS |
| backoff cap 2s | PASS |
| off-by-one fixed | PASS |
| 142 tests pass | PASS |

**7/7 preserved.**

---

## Case B — tool-heavy (tool_calls + tool results)

### Before → after

| metric | value |
|---|---|
| input messages | 26 |
| input size | 21,783 chars (~5,445 tokens) |
| tool calls / tool results in input | 8 / 8 |
| compaction triggered | True (state v1, 11 ops applied) |
| **messages forwarded to M3** | **7** (roles: system, user, user, assistant, tool, assistant, user) |
| **size forwarded to M3** | **2,193 chars** (~90% smaller) |
| state block present in forwarded msgs | True |
| tool messages kept verbatim in tail | 1 |
| e2e latency | 10.42s |

### Tool-pair safety (checked on exactly what M3 received)

- `tail_no_orphan_lead`: **PASS**
- `no_orphan_tool_results`: **PASS**
- `assistant_tool_calls_paired_with_results`: **PASS**

The kept tail includes an `assistant(tool_calls)` + its `tool` result as an intact pair (roles `... assistant, tool, assistant, ...`). The tail never begins on an orphan `tool` result, so the upstream never sees a tool result without its originating call.

### State block M3 actually received (verbatim, replaces the folded turns)

```markdown
# Compacted working context
(Older turns were summarized into durable state; recent turns are verbatim below.)

## Objective
Refactor DB layer in `orders`: add connection pool and retry-with-backoff for transient connection errors. Pool must cover p99 ~30 concurrent checkouts; retries must not hammer DB; public `Db` trait signature unchanged.

## Constraints / rules (do not violate)
- Public `Db` trait signature must not change.
- Pool must cover p99 ~30 concurrent checkouts (use size >= 30 with headroom).
- Retries must not hammer the DB (use exponential backoff with jitter, bounded max attempts).

## Decisions
- Use fixed-size connection pool of size 32 (covers ~30 p99 with headroom). — p99 is ~30 concurrent checkouts; 32 provides headroom without overprovisioning.
- Retry policy: max 5 attempts, exponential backoff 50ms -> 2s with full jitter. — Bounded attempts prevent hammering; full jitter avoids thundering herd.

## Files touched
- `src/db.rs` (modified): Added fixed-size pool (size 32) and retry-with-backoff wrapper around connect(); off-by-one in retry counter fixed (`<=` -> `<`).
- `src/config.rs` (modified): Added config knobs: pool_size=32; retry{max_attempts:5, base_delay_ms:50, max_delay_ms:2000}.

## Key tool outcomes
- [fail] cargo test: 141 passed, 1 failed (db::tests::retry_attempt_cap: expected 5 attempts, got 6; off-by-one `<=` should be `<`).
- [ok] cargo test after off-by-one fix: 142 passed, 0 failed.

## Artifacts produced
- orders v0.3.9 — Crate version compiled during test runs.
```

### M3's answer (post-compaction)

```
## PR Summary

**Files changed**
- `src/db.rs` — added fixed-size connection pool (size 32) and retry-with-backoff wrapper around `connect()`; fixed off-by-one in retry counter (`<=` → `<`); bumped module header to `// orders v0.4.0`.
- `src/config.rs` — added config knobs: `pool_size=32`; `retry{max_attempts:5, base_delay_ms:50, max_delay_ms:2000}`.

**Key decisions**
- **Pool size 32** — covers p99 ~30 concurrent checkouts with headroom, without overprovisioning.
- **Retry policy: max 5 attempts, exponential backoff 50ms → 2s with full jitter** — bounded attempts prevent hammering; full jitter avoids thundering herd.

**Final test result**
- `cargo test`: **142 passed, 0 failed** (after fixing the off-by-one retry counter bug).
```

### Preservation checks

| fact (folded out of the verbatim transcript) | recalled in answer |
|---|---|
| files src/db.rs + src/config.rs | PASS |
| pool size 32 | PASS |
| max 5 retries | PASS |
| backoff 50ms | PASS |
| backoff cap 2s | PASS |
| off-by-one fixed | PASS |
| 142 tests pass | PASS |

**7/7 preserved.**

---

## How to reproduce

1. Start an OpenAI-compatible model (any `/v1/chat/completions`); point the companion at it via `COMPACTOR_UPSTREAM_BASE_URL` + `COMPACTOR_MODEL`.
2. Run the companion: `uvicorn compactor.server:app --port 4100`.
3. Configure llmconduit with `compaction.enabled=true`, `compaction.endpoint=http://127.0.0.1:4100`, and a low `compaction.max_input_tokens` to force a trigger; start `llmconduit start`.
4. Send an over-budget chat request to the gateway; inspect `upstream_request_log_path` to see the compacted messages that were forwarded.

The runner used for this log is `tier3_runner.py` (builds both transcripts, drives the gateway, and reads llmconduit's upstream log for ground truth).
