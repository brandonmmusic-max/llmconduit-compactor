//! Optional context compaction for llmconduit (sketch for the upstream PR).
//!
//! Hook point: in `upstream.rs`/`engine.rs`, right before the finalized
//! OpenAI chat body is sent to `/v1/chat/completions`, call
//! `maybe_compact(&cfg, &mut body, session_id).await`. It mutates `body["messages"]`
//! in place when over budget; on ANY error it leaves the body untouched
//! (best-effort, never fail the user's request).
//!
//! Two modes:
//!   - External: POST the messages to a compactor service implementing the
//!     contract (e.g. the Python `llmconduit-compactor`).
//!   - Builtin: a trivial chunked-summary fallback so the feature is
//!     batteries-included without an external process. (left as a TODO stub)

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Debug, Deserialize)]
pub struct CompactionConfig {
    #[serde(default)]
    pub enabled: bool,
    /// "external" | "builtin"
    #[serde(default = "default_mode")]
    pub mode: String,
    /// external compactor base URL, e.g. "http://127.0.0.1:4100"
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default = "default_max_input_tokens")]
    pub max_input_tokens: usize,
    #[serde(default = "default_keep_recent_turns")]
    pub keep_recent_turns: usize,
    /// network timeout for the compactor call
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_mode() -> String { "external".into() }
fn default_max_input_tokens() -> usize { 96_000 }
fn default_keep_recent_turns() -> usize { 6 }
fn default_timeout_ms() -> u64 { 60_000 }

impl Default for CompactionConfig {
    fn default() -> Self {
        Self { enabled: false, mode: default_mode(), endpoint: None,
               max_input_tokens: default_max_input_tokens(),
               keep_recent_turns: default_keep_recent_turns(),
               timeout_ms: default_timeout_ms() }
    }
}

/// Rough token estimate over the chat body (~4 chars/token).
fn estimate_tokens(messages: &[Value]) -> usize {
    let mut chars = 0usize;
    for m in messages {
        if let Some(s) = m.get("content").and_then(Value::as_str) {
            chars += s.len();
        } else if let Some(arr) = m.get("content").and_then(Value::as_array) {
            for b in arr {
                chars += b.get("text").and_then(Value::as_str).map(str::len).unwrap_or(0);
            }
        }
        if let Some(tc) = m.get("tool_calls") {
            chars += tc.to_string().len();
        }
    }
    (chars / 4).max(1)
}

/// Best-effort: mutate `body["messages"]` if over budget. Never errors out.
pub async fn maybe_compact(cfg: &CompactionConfig, body: &mut Value, session_id: &str) {
    if !cfg.enabled {
        return;
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else { return };
    if estimate_tokens(messages) <= cfg.max_input_tokens {
        return;
    }

    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();

    let new_messages = match cfg.mode.as_str() {
        "external" => match call_external(cfg, messages, session_id, &model).await {
            Ok(Some(m)) => m,
            _ => return, // failure or no-op -> forward original
        },
        // "builtin" => builtin_chunk_summarize(cfg, messages, &model).await, // TODO
        _ => return,
    };

    if let Some(obj) = body.as_object_mut() {
        obj.insert("messages".into(), Value::Array(new_messages));
    }
}

async fn call_external(
    cfg: &CompactionConfig,
    messages: &[Value],
    session_id: &str,
    model: &str,
) -> Result<Option<Vec<Value>>, Box<dyn std::error::Error + Send + Sync>> {
    let endpoint = cfg.endpoint.as_deref().ok_or("compaction.endpoint not set")?;
    let req = json!({
        "session_id": session_id,
        "messages": messages,
        "max_input_tokens": cfg.max_input_tokens,
        "keep_recent_turns": cfg.keep_recent_turns,
        "model": model,
    });
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(cfg.timeout_ms))
        .build()?;
    let resp: Value = client
        .post(format!("{}/compact", endpoint.trim_end_matches('/')))
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    if resp.get("compacted").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    Ok(resp.get("messages").and_then(Value::as_array).cloned())
}

/// Derive a stable session id when the client doesn't supply one: hash the
/// system prompt + first user message (immutable across a session).
pub fn derive_session_id(messages: &[Value]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for m in messages.iter() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "system" || role == "user" {
            m.get("content").and_then(Value::as_str).unwrap_or("").hash(&mut h);
            if role == "user" {
                break; // first user msg only
            }
        }
    }
    format!("sess-{:016x}", h.finish())
}
