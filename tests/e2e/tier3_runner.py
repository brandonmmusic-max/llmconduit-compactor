#!/usr/bin/env python3
"""Tier 3 e2e runner — proves llmconduit context compaction on BOTH
a pure-chat transcript and a tool-heavy (tool_calls + tool results) transcript,
end-to-end through the real gateway, and writes a reviewable markdown log.

For each case:
  - direct /compact call         -> contract fields {compacted, ops, version}
  - full e2e through llmconduit  -> M3's final answer
  - upstream request log         -> EXACTLY what M3 received (ground truth):
                                    state block, message shape, tool-pair safety
  - preservation checks          -> are the buried facts recalled?
"""
import json
import re
import time
import urllib.request

LLMCONDUIT = "http://127.0.0.1:4000/v1/chat/completions"
COMPANION = "http://127.0.0.1:4100/compact"
UPSTREAM_LOG = "/tmp/llmconduit_upstream.jsonl"
MODEL = "minimax-m3-nvfp4"
KEEP = 2
BUDGET = 2000

_CALL = [0]


def _post(url, payload, timeout=300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tool_call(name, args):
    _CALL[0] += 1
    cid = f"call_{_CALL[0]:03d}"
    return cid, {"role": "assistant", "content": None,
                 "tool_calls": [{"id": cid, "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool_result(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def fileblob(path, n):
    return "\n".join([f"// {path}"] + [f"    let _v{i} = ctx.get({i}).unwrap_or_default(); // boilerplate {i}" for i in range(n)])


def cargoblob(n):
    return "\n".join(["   Compiling orders v0.3.9"] + [f"test db::tests::case_{i:03d} ... ok" for i in range(n)])


# ---------- Case A: pure chat (no tools) ----------
def build_chat():
    return [
        {"role": "system", "content": "You are a precise senior Rust engineer in the `orders` service. Be concise and accurate."},
        {"role": "user", "content": "Refactor the DB layer in `orders`: add a connection pool and retry-with-backoff for transient connection errors. Constraints: pool must cover our p99 of ~30 concurrent checkouts; retries must NOT hammer the DB; do NOT change the public `Db` trait signature."},
        {"role": "assistant", "content": "Understood; keeping the public `Db` trait unchanged. Reading the current code.\n\n" + fileblob("src/db.rs", 50)},
        {"role": "user", "content": "Here are the files for reference. Implement it.\n\n" + fileblob("src/db.rs", 120) + "\n\n" + fileblob("src/config.rs", 90)},
        {"role": "assistant", "content": "Implemented. Changes:\n- `src/db.rs`: added a `Pool` with FIXED SIZE 32 (covers ~30 p99 with headroom) and wrapped `connect()` in a retry loop: MAX 5 ATTEMPTS, EXPONENTIAL BACKOFF 50ms->2s, full jitter. Public `Db` trait unchanged.\n- `src/config.rs`: added `pool_size` (32) and `retry` (max_attempts=5, base_delay_ms=50, max_delay_ms=2000).\nKEY DECISION: cap retries at 5 with jittered exp backoff 50ms->2s so we recover without hammering the DB; pool 32 from p99 ~30."},
        {"role": "user", "content": "Run the full test suite.\n\n" + cargoblob(120)},
        {"role": "assistant", "content": "First run FAILED: the retry loop did 6 attempts instead of 5 — an OFF-BY-ONE (`<=` should be `<`). Fixed it in `src/db.rs`. Re-ran: ALL 142 TESTS PASS."},
        {"role": "user", "content": "Thanks. Also bump the version comment at the top of the main DB module to 0.4.0."},
        {"role": "assistant", "content": "Done — header comment updated to `// orders v0.4.0`."},
        {"role": "user", "content": "Before we wrap up, write the PR summary: (1) which files did we change, (2) the key retry + pooling decision with the specific numbers, and (3) the final test result?"},
    ]


# ---------- Case B: tool-heavy (tool_calls + tool results) ----------
def build_tools():
    M = [{"role": "system", "content": "You are a precise senior Rust engineer in the `orders` service. Use tools to read/edit code and run tests. Be concise and accurate."}]
    M.append({"role": "user", "content": "Refactor the DB layer in `orders`: add a connection pool and retry-with-backoff for transient connection errors. Constraints: pool must cover our p99 of ~30 concurrent checkouts; retries must NOT hammer the DB; do NOT change the public `Db` trait signature."})
    c, m = tool_call("read_file", {"path": "src/db.rs"}); M += [m, tool_result(c, fileblob("src/db.rs", 90))]
    c, m = tool_call("read_file", {"path": "src/config.rs"}); M += [m, tool_result(c, fileblob("src/config.rs", 70))]
    M.append({"role": "assistant", "content": "Plan: fixed-size pool in `src/db.rs` + retry wrapper around `connect()`; config knobs in `src/config.rs`; keep public `Db` trait unchanged."})
    M.append({"role": "user", "content": "Looks right. Implement it."})
    c, m = tool_call("apply_patch", {"path": "src/db.rs", "patch": "add Pool size 32; retry max_attempts=5 backoff 50ms->2s full jitter; Db trait unchanged"}); M += [m, tool_result(c, "patch applied to src/db.rs")]
    c, m = tool_call("apply_patch", {"path": "src/config.rs", "patch": "pool_size=32; retry{max_attempts:5,base_delay_ms:50,max_delay_ms:2000}"}); M += [m, tool_result(c, "patch applied to src/config.rs")]
    M.append({"role": "assistant", "content": "Done. KEY DECISION: pool FIXED SIZE 32 (covers ~30 p99 with headroom); retry MAX 5 ATTEMPTS, EXPONENTIAL BACKOFF 50ms->2s, full jitter. Files: `src/db.rs`, `src/config.rs`. Public `Db` trait unchanged."})
    M.append({"role": "user", "content": "Run the full test suite."})
    c, m = tool_call("run_bash", {"cmd": "cargo test"}); M += [m, tool_result(c, cargoblob(120) + "\ntest db::tests::retry_attempt_cap ... FAILED\nassertion failed: expected 5 attempts, got 6 (off-by-one: `<=` should be `<`)\ntest result: FAILED. 141 passed; 1 failed")]
    c, m = tool_call("apply_patch", {"path": "src/db.rs", "patch": "fix off-by-one: `attempt <= max` -> `attempt < max`"}); M += [m, tool_result(c, "patch applied to src/db.rs (off-by-one fixed)")]
    c, m = tool_call("run_bash", {"cmd": "cargo test"}); M += [m, tool_result(c, cargoblob(142) + "\ntest result: ok. 142 passed; 0 failed")]
    M.append({"role": "assistant", "content": "Fixed an OFF-BY-ONE in the retry counter (6 attempts instead of 5; `<=` -> `<`) in `src/db.rs`. Re-ran: ALL 142 TESTS PASS."})
    M.append({"role": "user", "content": "Thanks. Also bump the version comment at the top of the main DB module to 0.4.0."})
    c, m = tool_call("apply_patch", {"path": "DB module header", "patch": "update header comment to // orders v0.4.0"}); M += [m, tool_result(c, "patch applied: header now // orders v0.4.0")]
    M.append({"role": "assistant", "content": "Done — header comment updated to `// orders v0.4.0`."})
    M.append({"role": "user", "content": "Before we wrap up, write the PR summary: (1) which files did we change, (2) the key retry + pooling decision with the specific numbers, and (3) the final test result?"})
    return M


def chars(msgs):
    tot = 0
    for m in msgs:
        c = m.get("content")
        tot += len(c) if isinstance(c, str) else (len(json.dumps(c)) if c else 0)
        if m.get("tool_calls"):
            tot += len(json.dumps(m["tool_calls"]))
    return tot


def strip_think(s):
    s = re.sub(r"<mm:think>.*?</mm:think>", "", s, flags=re.S)
    s = re.sub(r"^.*?</mm:think>", "", s, flags=re.S)  # leading orphan close tag
    return s.strip()


def read_forwarded():
    rows = [json.loads(l) for l in open(UPSTREAM_LOG) if l.strip()]
    if not rows:
        return None
    body = rows[-1]
    msgs = body.get("messages") or (body.get("body") or {}).get("messages")
    return msgs


def preservation(answer):
    a = answer.lower()
    return {
        "files src/db.rs + src/config.rs": "src/db.rs" in a and "src/config.rs" in a,
        "pool size 32": "32" in a,
        "max 5 retries": bool(re.search(r"\b5\b", a)),
        "backoff 50ms": "50" in a and ("ms" in a or "millis" in a),
        "backoff cap 2s": ("2s" in a) or ("2 s" in a) or ("2000" in a) or ("2 second" in a),
        "off-by-one fixed": ("off-by-one" in a) or ("< max" in a) or ("attempt <" in a) or ("strict" in a),
        "142 tests pass": "142" in a,
    }


def tool_safety(forwarded):
    tail = [m for m in (forwarded or []) if "Compacted working context" not in (m.get("content") or "") and m.get("role") != "system"]
    seen = set()
    orphan = False
    for m in tail:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            seen |= {tc.get("id") for tc in m["tool_calls"]}
        if m.get("role") == "tool" and m.get("tool_call_id") not in seen:
            orphan = True
    starts_clean = (not tail) or tail[0].get("role") != "tool"
    result_ids = {m.get("tool_call_id") for m in tail if m.get("role") == "tool"}
    complete = all(all(tc.get("id") in result_ids for tc in m["tool_calls"])
                   for m in tail if m.get("role") == "assistant" and m.get("tool_calls"))
    return {"tail_no_orphan_lead": starts_clean, "no_orphan_tool_results": not orphan,
            "assistant_tool_calls_paired_with_results": complete}


def run_case(name, messages, session):
    open(UPSTREAM_LOG, "w").close()  # truncate
    R = {"name": name, "in_messages": len(messages), "in_chars": chars(messages),
         "in_est_tokens": chars(messages) // 4,
         "tool_results": sum(1 for m in messages if m.get("role") == "tool"),
         "tool_calls": sum(len(m.get("tool_calls") or []) for m in messages)}
    comp = _post(COMPANION, {"session_id": session, "messages": messages,
                             "max_input_tokens": BUDGET, "keep_recent_turns": KEEP, "model": MODEL})
    R["compacted"] = comp.get("compacted")
    R["ops_applied"] = comp.get("ops_applied")
    R["state_version"] = comp.get("state_version")
    t0 = time.time()
    resp = _post(LLMCONDUIT, {"model": MODEL, "messages": messages, "max_tokens": 1500, "temperature": 0.2})
    R["e2e_latency_s"] = round(time.time() - t0, 2)
    raw = resp["choices"][0]["message"]["content"]
    R["answer_raw"] = raw
    R["answer"] = strip_think(raw) or raw
    fwd = read_forwarded()
    R["forwarded_messages"] = len(fwd) if fwd else None
    R["forwarded_roles"] = [m.get("role") for m in fwd] if fwd else None
    R["forwarded_chars"] = chars(fwd) if fwd else None
    R["state_block"] = next((m.get("content", "") for m in (fwd or []) if "Compacted working context" in (m.get("content") or "")), "")
    R["forwarded_has_state_block"] = bool(R["state_block"])
    R["forwarded_tool_msgs"] = sum(1 for m in (fwd or []) if m.get("role") == "tool")
    R["tool_pair_safety"] = tool_safety(fwd)
    pres = preservation(raw)
    R["preservation"] = pres
    R["preservation_passed"] = sum(pres.values())
    R["preservation_total"] = len(pres)
    return R


def main():
    cases = [run_case("Case A — pure chat", build_chat(), "tier3-chat"),
             run_case("Case B — tool-heavy (tool_calls + tool results)", build_tools(), "tier3-tools")]
    with open("/tmp/tier3_report.json", "w") as f:
        json.dump(cases, f, indent=2)
    print(json.dumps([{k: v for k, v in c.items() if k not in ("state_block", "answer_raw")} for c in cases], indent=2))


if __name__ == "__main__":
    main()
