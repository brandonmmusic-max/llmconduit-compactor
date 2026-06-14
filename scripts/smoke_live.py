"""Live smoke test against ANY OpenAI-compatible endpoint (model-agnostic).

Builds a long synthetic coding-agent transcript (decisions, file edits, a
failing then passing test, a constraint), runs ONE real compaction pass, and
prints before/after token estimate + the rendered state block. Confirms the
extractor pulls durable state out of whatever model you point it at.

    python scripts/smoke_live.py --base http://127.0.0.1:8000/v1
    # --model is auto-detected from the endpoint's /v1/models if omitted.
"""
import argparse
import asyncio
import json

import httpx

from compactor.extractor import extract_state_delta
from compactor.merge import merge
from compactor.schema import CompactState
from compactor.transcript import _tail_start, render_state_block, split_system
from compactor.triggers import estimate_tokens


def big_transcript():
    msgs = [{"role": "system", "content": "You are a coding agent in a Rust repo."}]
    events = [
        ("user", "Add a retry wrapper around the DB calls in src/db.rs. "
                 "Hard rule: never block the async runtime."),
        ("assistant", "Plan: add tokio-retry with exponential backoff in src/db.rs."),
        ("tool_call", "read_file src/db.rs"),
        ("tool", "pub async fn query(...) { ... } // ~200 lines"),
        ("assistant", "Decision: tokio-retry crate, backoff 50ms..2s, max 5 attempts."),
        ("user", "Also bump the connection pool to 32."),
        ("assistant", "Edited src/db.rs (added with_retry); set pool=32 in src/config.rs."),
        ("tool_call", "cargo test"),
        ("tool", "FAILED db::tests::retry_gives_up — expected Err after 5 tries"),
        ("assistant", "Off-by-one in the retry counter; fixing."),
        ("tool_call", "cargo test"),
        ("tool", "test result: ok. 142 passed; 0 failed"),
    ]
    for role, content in events:
        if role == "tool_call":
            msgs.append({"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": content.split()[0],
                                          "arguments": json.dumps({"cmd": content})}}]})
        elif role == "tool":
            msgs.append({"role": "tool", "tool_call_id": "c1", "content": content})
        else:
            msgs.append({"role": role, "content": content + "\n" + ("context filler. " * 350)})
    return msgs


async def detect_model(base: str, key: str | None) -> str | None:
    """Pick the first served model from /v1/models — keeps this model-agnostic."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(timeout=15) as cx:
            r = await cx.get(f"{base.rstrip('/')}/models", headers=headers)
            r.raise_for_status()
            data = r.json().get("data") or []
            return data[0]["id"] if data else None
    except Exception:
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="OpenAI-compatible base url, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None, help="served model name (auto-detected from /v1/models if omitted)")
    ap.add_argument("--key", default=None)
    ap.add_argument("--keep-recent", type=int, default=3)
    a = ap.parse_args()

    model = a.model or await detect_model(a.base, a.key)
    if not model:
        print("FAIL: no --model given and could not auto-detect from "
              f"{a.base}/models. Pass --model <served-name>.")
        return
    print(f"using model: {model}")

    msgs = big_transcript()
    sys_msgs, body = split_system(msgs)
    fold_point = _tail_start(body, a.keep_recent)
    folded = body[:fold_point]
    print(f"BEFORE: ~{estimate_tokens(msgs):,} tok, {len(msgs)} msgs "
          f"(folding {len(folded)} old msgs, keeping last {a.keep_recent} turns)")

    state = CompactState.empty("smoke")
    delta = await extract_state_delta(state, folded, base_url=a.base,
                                      model=model, api_key=a.key)
    if delta is None:
        print("FAIL: extractor returned None (check --base reachable / model usable).")
        return
    state = merge(state, delta)
    block = render_state_block(state)
    new = sys_msgs + [{"role": "user", "content": block}] + body[fold_point:]

    print(f"AFTER : ~{estimate_tokens(new):,} tok, {len(new)} msgs "
          f"({len(delta.ops)} ops -> state v{state.version})")
    print("\n----- rendered state block -----\n" + block + "\n--------------------------------")

    ok = bool(state.objective or state.decisions or state.files or state.constraints)
    print(f"\n{'OK' if ok else 'WARN'}: durable state "
          f"{'extracted' if ok else 'MISSING'}; recent turns + tool pairs kept verbatim.")


if __name__ == "__main__":
    asyncio.run(main())
