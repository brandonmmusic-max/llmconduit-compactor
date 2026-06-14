#!/usr/bin/env python3
"""Tier 3 LARGE (>100K token) e2e test for llmconduit context compaction.

Same integrated path as the small test (client -> llmconduit -> /compact ->
compacted msgs -> M3), but the input is >100K tokens so it exercises CHUNKED
extraction. Two cases:

  Case A (chat):  real KY Supreme Court opinions (public record, CAP corpus) as
                  the document under review.
  Case B (tools): real llmconduit Rust source (public OSS) read via tool calls.

Into each we plant 5 unique "needle" facts at increasing depth (~2/26/50/74/96%
of the folded content). The final question asks the model to recall all 5 by
tag + summarize. Recall-by-depth measures lost-in-the-middle across compaction;
the needles live ONLY in the folded region, so a correct answer can only come
from the compacted state.

We also pull the RAW state-delta ops (companion /state endpoint) so the exact
ops the extractor emitted can be reviewed.
"""
import json
import re
import time
import urllib.request

LLMCONDUIT = "http://127.0.0.1:4000/v1/chat/completions"
COMPANION = "http://127.0.0.1:4100"
UPSTREAM_LOG = "/tmp/llmconduit_upstream.jsonl"
MODEL = "minimax-m3-nvfp4"
KEEP = 2
BUDGET = 32000  # simulate a 32K effective window receiving >100K tokens

CAP_DIR = "/media/brandonmusic/nvme0n1p3/Users/brand/Downloads/kentucky_legal_counsel_local/app/data/cap_ky_plaintext"
RUST_DIR = "/tmp/llmconduit-work/src"

_CALL = [0]

NEEDLE_FRACS = [0.02, 0.26, 0.50, 0.74, 0.96]
TAGS = ["ALPHA-7731", "BRAVO-4420", "CHARLIE-9015", "DELTA-2286", "ECHO-6673"]


def _post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tool_call(name, args):
    _CALL[0] += 1
    cid = f"call_{_CALL[0]:03d}"
    return cid, {"role": "assistant", "content": None,
                 "tool_calls": [{"id": cid, "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool_result(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


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
    s = re.sub(r"^.*?</mm:think>", "", s, flags=re.S)
    return s.strip()


CHAT_NEEDLES = [
    f"[ATTORNEY WORK-PRODUCT NOTE ALPHA, tag {TAGS[0]} — appellate strategy: our lead assignment of error is the trial court's handling of the accomplice-testimony corroboration instruction.]",
    f"[ATTORNEY WORK-PRODUCT NOTE BRAVO, tag {TAGS[1]} — retain Dr. Helga Vongsawad as our independent forensic timeline expert before the next status conference.]",
    f"[ATTORNEY WORK-PRODUCT NOTE CHARLIE, tag {TAGS[2]} — decision: do NOT raise the venue-change issue on appeal; it was waived below and would dilute the stronger claims.]",
    f"[ATTORNEY WORK-PRODUCT NOTE DELTA, tag {TAGS[3]} — file the supplemental authority letter citing the 2019 amendment by the March 14 deadline.]",
    f"[ATTORNEY WORK-PRODUCT NOTE ECHO, tag {TAGS[4]} — settlement floor authorized by client is no less than a sentence-modification to life without parole; never disclose this figure to opposing counsel.]",
]

TOOL_NEEDLES = [
    f"FINDING ALPHA (tag {TAGS[0]}): every API surface (Anthropic /v1/messages, /v1/responses, /v1/chat/completions) converges on engine.rs `run_turn` — that single function is the only place to hook cross-cutting behavior like compaction.",
    f"FINDING BRAVO (tag {TAGS[1]}): the Anthropic<->Responses adapters are the riskiest code; tool_use/tool_result block pairing is reconstructed there and is the most likely source of desync bugs.",
    f"FINDING CHARLIE (tag {TAGS[2]}): upstream.rs implements failover with a per-endpoint cooldown; a dead upstream is skipped for `upstream_failure_cooldown_secs` before retry.",
    f"FINDING DELTA (tag {TAGS[3]}): config is persisted vs runtime-split (PersistedConfig -> Config via from_persisted); any new knob must be threaded through BOTH or it silently defaults.",
    f"FINDING ECHO (tag {TAGS[4]}): streaming SSE is assembled in http.rs; the chat-completions delta format and the Anthropic event format are produced from one internal event stream.",
]


def load_chat_doc():
    import os
    files = ["sw3d_348_0627-01.txt", "sw3d_192_0350-01.txt"]
    parts = []
    for f in files:
        p = os.path.join(CAP_DIR, f)
        if os.path.exists(p):
            parts.append(open(p, encoding="utf-8", errors="replace").read())
    doc = "\n\n===== NEXT CONTROLLING AUTHORITY =====\n\n".join(parts)
    # insert needles at depth fractions on paragraph boundaries
    L = len(doc)
    out = doc
    # insert from last to first so offsets stay valid
    for frac, needle in sorted(zip(NEEDLE_FRACS, CHAT_NEEDLES), reverse=True):
        pos = int(L * frac)
        nl = out.find("\n", pos)
        if nl == -1:
            nl = pos
        out = out[:nl] + "\n\n" + needle + "\n\n" + out[nl:]
    return out


def build_chat():
    doc = load_chat_doc()
    half = len(doc) // 2
    cut = doc.find("\n", half)
    d1, d2 = doc[:cut], doc[cut:]
    return [
        {"role": "system", "content": "You are an appellate attorney's research assistant. Analyze the provided authorities and the attorney's work-product notes carefully and accurately."},
        {"role": "user", "content": "I'm preparing a Kentucky criminal appeal. Here is the first part of my case file — controlling authority plus my work-product notes. Read it carefully; I'll send the rest next.\n\n" + d1},
        {"role": "assistant", "content": "Received the first part. I've read the authority and your interleaved work-product notes. Send the remainder and I'll analyze the whole file together."},
        {"role": "user", "content": "Here is the remainder of the case file.\n\n" + d2 + "\n\nGive me a brief initial read."},
        {"role": "assistant", "content": "Initial read: this is a capital/serious-felony appeal turning heavily on accomplice testimony and the corroboration rules, with several preserved evidentiary issues. I've noted your work-product annotations throughout."},
        {"role": "user", "content": "Good. Separately, please use Bluebook style for any citations in the final work product."},
        {"role": "assistant", "content": "Understood — Bluebook style for all citations in the final work product."},
        {"role": "user", "content": "Before we continue: (1) list ALL of my attorney work-product notes — give the tag code AND the substance of each one; and (2) give me a 4-5 sentence summary of the lead opinion's holding."},
    ]


def read_rust(fname):
    import os
    p = os.path.join(RUST_DIR, fname)
    return open(p, encoding="utf-8", errors="replace").read()


def build_tools():
    files = ["engine.rs", "adapters/responses_to_anthropic.rs", "adapters/anthropic_to_responses.rs",
             "adapters/responses_to_chat.rs", "upstream.rs", "config.rs", "adapters/chat_to_responses.rs",
             "http.rs", "monitor.rs", "adapters/chat_completions.rs"]
    M = [{"role": "system", "content": "You are a senior Rust engineer doing a codebase study of `llmconduit`. Use tools to read files. Record concrete findings as you go."}]
    M.append({"role": "user", "content": "Study the llmconduit codebase and explain how a request flows through it end to end. Read the key source files and note important findings as you go."})
    # interleave reads with finding-needles at increasing depth
    needle_after = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4}  # after reading file index -> emit needle k
    for idx, f in enumerate(files):
        c, m = tool_call("read_file", {"path": f"src/{f}"})
        M += [m, tool_result(c, f"// src/{f}\n" + read_rust(f))]
        if idx in needle_after:
            M.append({"role": "assistant", "content": TOOL_NEEDLES[needle_after[idx]]})
    M.append({"role": "assistant", "content": "Done reading the core files; I have the full request-flow picture and recorded findings ALPHA through ECHO."})
    M.append({"role": "user", "content": "Thanks. Quick aside: what Rust edition is this crate on?"})
    M.append({"role": "assistant", "content": "It targets Rust edition 2024."})
    M.append({"role": "user", "content": "Now: (1) list ALL findings you recorded — tag code AND substance of each; and (2) summarize in 4-5 sentences how a request flows through llmconduit end to end."})
    return M


def needle_recall(answer):
    a = answer
    return {tag: (tag in a) for tag in TAGS}


def tool_safety(forwarded):
    tail = [m for m in (forwarded or []) if "Compacted working context" not in (m.get("content") or "") and m.get("role") != "system"]
    seen = set(); orphan = False
    for m in tail:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            seen |= {tc.get("id") for tc in m["tool_calls"]}
        if m.get("role") == "tool" and m.get("tool_call_id") not in seen:
            orphan = True
    starts_clean = (not tail) or tail[0].get("role") != "tool"
    return {"tail_no_orphan_lead": starts_clean, "no_orphan_tool_results": not orphan}


def run_case(name, messages, session):
    open(UPSTREAM_LOG, "w").close()
    R = {"name": name, "in_messages": len(messages), "in_chars": chars(messages),
         "in_est_tokens": chars(messages) // 4,
         "tool_results": sum(1 for m in messages if m.get("role") == "tool"),
         "tool_calls": sum(len(m.get("tool_calls") or []) for m in messages)}
    # direct /compact (introspection session) -> contract + lets us read raw ops
    t0 = time.time()
    comp = _post(COMPANION + "/compact", {"session_id": session, "messages": messages,
                                          "max_input_tokens": BUDGET, "keep_recent_turns": KEEP, "model": MODEL})
    R["extract_latency_s"] = round(time.time() - t0, 1)
    R["compacted"] = comp.get("compacted")
    R["ops_applied"] = comp.get("ops_applied")
    R["state_version"] = comp.get("state_version")
    R["compacted_messages"] = len(comp.get("messages", []))
    R["compacted_chars"] = chars(comp.get("messages", []))
    # raw delta ops for review
    st = _get(f"{COMPANION}/state/{session}")
    R["raw_state"] = st.get("state")
    R["raw_deltas"] = st.get("deltas")
    R["n_ops"] = st.get("n_ops")
    # full e2e through llmconduit
    t0 = time.time()
    resp = _post(LLMCONDUIT, {"model": MODEL, "messages": messages, "max_tokens": 3000, "temperature": 0.2})
    R["e2e_latency_s"] = round(time.time() - t0, 1)
    raw = resp["choices"][0]["message"]["content"]
    R["answer_raw"] = raw
    R["answer"] = strip_think(raw) or raw
    fwd = [json.loads(l) for l in open(UPSTREAM_LOG) if l.strip()]
    fwd = (fwd[-1].get("messages") if fwd else None)
    R["forwarded_messages"] = len(fwd) if fwd else None
    R["forwarded_roles"] = [m.get("role") for m in fwd] if fwd else None
    R["forwarded_chars"] = chars(fwd) if fwd else None
    R["forwarded_state_block"] = next((m.get("content", "") for m in (fwd or []) if "Compacted working context" in (m.get("content") or "")), "")
    R["tool_pair_safety"] = tool_safety(fwd) if R["tool_calls"] else None
    R["needle_recall"] = needle_recall(raw)
    R["needles_recalled"] = sum(needle_recall(raw).values())
    return R


def main():
    cases = [run_case("Case A — chat, >100K (KY appellate case file)", build_chat(), "tier3-large-chat"),
             run_case("Case B — tool-heavy, >100K (llmconduit codebase study)", build_tools(), "tier3-large-tools")]
    json.dump(cases, open("/tmp/tier3_large_report.json", "w"), indent=2)
    for c in cases:
        print(f"\n### {c['name']}")
        print(f"  input: {c['in_messages']} msgs, {c['in_chars']:,} ch (~{c['in_est_tokens']:,} tok); tool_calls={c['tool_calls']}")
        print(f"  compacted={c['compacted']} ops={c['ops_applied']} n_ops_raw={c['n_ops']} extract={c['extract_latency_s']}s")
        print(f"  forwarded to M3: {c['forwarded_messages']} msgs, {c['forwarded_chars']:,} ch  (state block: {bool(c['forwarded_state_block'])})")
        if c["tool_pair_safety"]:
            print(f"  tool_pair_safety: {c['tool_pair_safety']}")
        print(f"  needle recall by depth {NEEDLE_FRACS}: {[c['needle_recall'][t] for t in TAGS]}  -> {c['needles_recalled']}/5")
        print(f"  e2e_latency={c['e2e_latency_s']}s")


if __name__ == "__main__":
    main()
