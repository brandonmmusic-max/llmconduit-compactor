# llmconduit compaction hook — integration sketch (for the PR)

This is the small, generic, opt-in Rust side. It adds a `compaction` config
block and one call site; the actual compaction logic lives in an external
service (the Python companion here) or a builtin fallback.

## 1. config (`config.rs` + `config.yaml`)
Add to the gateway config struct:
```rust
#[serde(default)]
pub compaction: crate::compaction::CompactionConfig,
```
`config.yaml`:
```yaml
compaction:
  enabled: true
  mode: "external"                 # external | builtin
  endpoint: "http://127.0.0.1:4100"
  max_input_tokens: 96000          # trigger; set to your model's EFFECTIVE window
  keep_recent_turns: 6
  timeout_ms: 60000
```
Note: set `max_input_tokens` to the local model's *effective quality* window,
not its advertised max — that's the whole point of doing this for a model whose
harness over-trusts the context length.

## 2. module
Drop `compaction.rs` into `src/`, add `mod compaction;` in `lib.rs`. Deps you
likely already have: `serde`, `serde_json`, `reqwest`. (`reqwest` is used for
fallback upstreams already.)

## 3. call site (`upstream.rs` / `engine.rs`)
Right before the finalized OpenAI chat body is sent upstream (after
`responses_to_chat`, before the POST):
```rust
let session_id = headers
    .get("x-session-id").and_then(|v| v.to_str().ok()).map(str::to_string)
    .or_else(|| body.get("metadata").and_then(|m| m.get("session_id"))
                  .and_then(|v| v.as_str()).map(str::to_string))
    .unwrap_or_else(|| compaction::derive_session_id(
        body["messages"].as_array().map(Vec::as_slice).unwrap_or(&[])));

compaction::maybe_compact(&cfg.compaction, &mut body, &session_id).await;
```
`maybe_compact` is best-effort: over budget → calls the compactor and swaps
`body["messages"]`; on any error → leaves the body untouched. The user's
request never fails because of compaction.

## 4. the contract (what the external service must implement)
```
POST {endpoint}/compact
 req : {session_id, messages[], max_input_tokens, keep_recent_turns, model}
 resp: {messages[], compacted: bool, state_version, ops_applied}
```
`compacted=false` ⇒ gateway forwards the original messages.

## 5. builtin fallback (optional, nice-to-have for "batteries included")
Implement `builtin_chunk_summarize` (the `// TODO` stub): keep system + last N
turns + tool pairs verbatim; summarize each *fixed-boundary* older chunk via the
upstream model and cache by `hash(chunk)` so completed-chunk summaries stay
byte-identical (cache-stable). This gives compaction with no external process,
at lower fidelity than the StateDelta companion.

## Test surface
Add to `tests/gateway.rs`: (a) under budget ⇒ body unchanged; (b) over budget +
stub compactor ⇒ messages replaced; (c) compactor error ⇒ original forwarded;
(d) tail never starts with an orphan `tool` message.
```
```
