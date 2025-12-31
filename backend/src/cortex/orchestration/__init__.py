"""Orchestration module for Cortex.

Implements §10 of the Canonical Blueprint.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    # Graphs
    "graph_answer_question": "cortex.orchestration.graphs",
    "graph_draft_email": "cortex.orchestration.graphs",
    "graph_summarize_thread": "cortex.orchestration.graphs",
    # Nodes and tools
    "tool_email_get_thread": "cortex.orchestration.nodes",
    "extract_document_mentions": "cortex.orchestration.nodes",
    "node_handle_error": "cortex.orchestration.nodes",
    "node_assemble_context": "cortex.orchestration.nodes",
    "node_classify_query": "cortex.orchestration.nodes",
    "node_retrieve_context": "cortex.orchestration.nodes",
    "node_query_graph": "cortex.orchestration.nodes",
    "node_generate_answer": "cortex.orchestration.nodes",
    "node_prepare_draft_query": "cortex.orchestration.nodes",
    "node_draft_email_initial": "cortex.orchestration.nodes",
    "node_critique_draft": "cortex.orchestration.nodes",
    "node_improve_draft": "cortex.orchestration.nodes",
    "node_audit_draft": "cortex.orchestration.nodes",
    "node_select_attachments": "cortex.orchestration.nodes",
    "node_load_thread": "cortex.orchestration.nodes",
    "node_summarize_analyst": "cortex.orchestration.nodes",
    "node_summarize_critic": "cortex.orchestration.nodes",
    "node_summarize_improver": "cortex.orchestration.nodes",
    "node_summarize_final": "cortex.orchestration.nodes",
    # States
    "AnswerQuestionState": "cortex.orchestration.states",
    "DraftEmailState": "cortex.orchestration.states",
    "SummarizeThreadState": "cortex.orchestration.states",
}

__all__ = [
    # Public API: graphs and states only
    "graph_answer_question",
    "graph_draft_email",
    "graph_summarize_thread",
    "AnswerQuestionState",
    "DraftEmailState",
    "SummarizeThreadState",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if not module_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
