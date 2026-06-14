"""llmconduit-compactor: an event-sourced context compactor for AI coding
agents, exposed as an external compactor for llmconduit (or any gateway).

De-legal-ized lift of KLC's StateDelta-ledger compaction: the generic mechanism
(ops -> merge -> materialized view -> replace old turns) tuned for coding-agent
state instead of legal case state.
"""
from .schema import CompactState, StateDelta, SCHEMA_VERSION
from .merge import merge, replay
from .triggers import estimate_tokens, should_compact
from .transcript import splice, render_state_block, split_system
from .extractor import extract_state_delta, build_extractor_messages

__all__ = [
    "CompactState", "StateDelta", "SCHEMA_VERSION",
    "merge", "replay",
    "estimate_tokens", "should_compact",
    "splice", "render_state_block", "split_system",
    "extract_state_delta", "build_extractor_messages",
]
