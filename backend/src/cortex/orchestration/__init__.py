"""
Orchestration module for Cortex.

Implements §10 of the Canonical Blueprint.
"""
from cortex.orchestration.graphs import (
    graph_answer_question,
    graph_draft_email,
    graph_summarize_thread,
)
from cortex.orchestration.nodes import (  # Helper tools (§10.2); Answer graph nodes; Draft graph nodes; Summarize graph nodes
    extract_document_mentions,
    node_assemble_context,
    node_audit_draft,
    node_classify_query,
    node_critique_draft,
    node_draft_email_initial,
    node_generate_answer,
    node_handle_error,
    node_improve_draft,
    node_load_thread,
    node_prepare_draft_query,
    node_retrieve_context,
    node_summarize_analyst,
    node_summarize_critic,
    node_summarize_final,
    node_summarize_improver,
    tool_email_get_thread,
)
from cortex.orchestration.states import (
    AnswerQuestionState,
    DraftEmailState,
    SummarizeThreadState,
)

__all__ = [
    # Tools
    "tool_email_get_thread",
    "extract_document_mentions",
    # Nodes
    "node_handle_error",
    "node_assemble_context",
    "node_classify_query",
    "node_retrieve_context",
    "node_generate_answer",
    "node_prepare_draft_query",
    "node_draft_email_initial",
    "node_critique_draft",
    "node_improve_draft",
    "node_audit_draft",
    "node_load_thread",
    "node_summarize_analyst",
    "node_summarize_critic",
    "node_summarize_improver",
    "node_summarize_final",
    # States
    "AnswerQuestionState",
    "DraftEmailState",
    "SummarizeThreadState",
    # Graphs
    "graph_answer_question",
    "graph_draft_email",
    "graph_summarize_thread",
]
