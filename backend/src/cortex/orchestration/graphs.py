"""
LangGraph Definitions.

Implements §10.1 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from cortex.orchestration.states import AnswerQuestionState, DraftEmailState, SummarizeThreadState
from cortex.orchestration.nodes import (
    node_classify_query,
    node_retrieve_context,
    node_assemble_context,
    node_generate_answer,
    node_handle_error,
    node_prepare_draft_query,
    node_draft_email_initial,
    node_critique_draft,
    node_improve_draft,
    node_audit_draft,
    node_load_thread,
    node_summarize_analyst,
    node_summarize_critic,
    node_summarize_improver,
    node_summarize_final
)

logger = logging.getLogger(__name__)


def _check_error(state: dict) -> str:
    """Check if state has error and route accordingly."""
    if state.get("error"):
        return "handle_error"
    return "continue"


def _route_by_classification(state: dict) -> str:
    """Route based on query classification."""
    classification = state.get("classification")
    if not classification:
        return "retrieve"
    
    # Could add specialized routing for navigational vs semantic queries
    # For now, always go to retrieve
    return "retrieve"


def build_answer_graph() -> StateGraph:
    """
    Build graph_answer_question.
    
    Blueprint §10.1:
    * Search -> Answer
    """
    workflow = StateGraph(AnswerQuestionState)
    
    # Define nodes
    workflow.add_node("classify", node_classify_query)
    workflow.add_node("retrieve", node_retrieve_context)
    workflow.add_node("assemble", node_assemble_context)
    workflow.add_node("generate", node_generate_answer)
    workflow.add_node("handle_error", node_handle_error)
    
    # Define edges with error handling
    workflow.set_entry_point("classify")
    
    # Classification -> route by type (or continue to retrieve)
    workflow.add_conditional_edges(
        "classify",
        _route_by_classification,
        {"retrieve": "retrieve", "handle_error": "handle_error"}
    )
    
    # Retrieve -> check error or continue to assemble
    workflow.add_conditional_edges(
        "retrieve",
        _check_error,
        {"continue": "assemble", "handle_error": "handle_error"}
    )
    
    workflow.add_edge("assemble", "generate")
    
    # Generate -> check error or end
    workflow.add_conditional_edges(
        "generate",
        _check_error,
        {"continue": END, "handle_error": "handle_error"}
    )
    
    workflow.add_edge("handle_error", END)
    
    return workflow


def should_improve(state: DraftEmailState) -> str:
    """Determine if draft needs improvement."""
    critique = state.critique
    iteration = state.iteration_count
    
    if iteration >= 3: # Max iterations
        return "audit"
        
    if not critique:
        return "audit"
        
    # Check if there are major issues
    has_major_issues = any(issue.severity in ["major", "critical"] for issue in critique.issues)
    
    if has_major_issues:
        return "improve"
        
    return "audit"


def build_draft_graph() -> StateGraph:
    """
    Build graph_draft_email.
    
    Blueprint §10.3:
    1. Prepare query
    2. Gather context
    3. Draft initial
    4. Critique
    5. Audit
    6. Improve (loop)
    7. Select attachments
    8. Finalize
    """
    workflow = StateGraph(DraftEmailState)
    
    # Nodes
    workflow.add_node("prepare_query", node_prepare_draft_query)
    # Reuse retrieval nodes from answer graph, but we need to map state keys if they differ
    # DraftEmailState has 'explicit_query' mapped to 'query' by prepare_query node
    # And it has 'retrieval_results' and 'assembled_context' same as AnswerQuestionState
    # So we can reuse the nodes if they operate on compatible state dicts.
    # node_retrieve_context expects 'query' in state. node_prepare_draft_query puts it there.
    
    workflow.add_node("retrieve", node_retrieve_context)
    workflow.add_node("assemble", node_assemble_context)
    workflow.add_node("draft_initial", node_draft_email_initial)
    workflow.add_node("critique", node_critique_draft)
    workflow.add_node("improve", node_improve_draft)
    workflow.add_node("audit", node_audit_draft)
    workflow.add_node("handle_error", node_handle_error)
    
    # Edges
    workflow.set_entry_point("prepare_query")
    workflow.add_edge("prepare_query", "retrieve")
    workflow.add_edge("retrieve", "assemble")
    workflow.add_edge("assemble", "draft_initial")
    workflow.add_edge("draft_initial", "critique")
    
    # Conditional edge for loop
    workflow.add_conditional_edges(
        "critique",
        should_improve,
        {
            "improve": "improve",
            "audit": "audit"
        }
    )
    
    workflow.add_edge("improve", "critique")
    workflow.add_edge("audit", END)
    
    return workflow


def build_summarize_graph() -> StateGraph:
    """
    Build graph_summarize_thread.
    
    Blueprint §10.4:
    1. Load thread
    2. Analyst (facts ledger)
    3. Critic
    4. Improver (optional)
    5. Merge metadata
    6. Finalize
    """
    workflow = StateGraph(SummarizeThreadState)
    
    # Nodes
    workflow.add_node("load_thread", node_load_thread)
    workflow.add_node("analyst", node_summarize_analyst)
    workflow.add_node("critic", node_summarize_critic)
    workflow.add_node("improver", node_summarize_improver)
    workflow.add_node("finalize", node_summarize_final)
    workflow.add_node("handle_error", node_handle_error)
    
    # Edges
    workflow.set_entry_point("load_thread")
    workflow.add_edge("load_thread", "analyst")
    workflow.add_edge("analyst", "critic")
    workflow.add_edge("critic", "improver")
    workflow.add_edge("improver", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow


# -----------------------------------------------------------------------------
# Canonical Blueprint Aliases (§10.1)
# -----------------------------------------------------------------------------
# Blueprint references graphs as graph_answer_question, graph_draft_email, 
# graph_summarize_thread. These aliases provide canonical naming.

graph_answer_question = build_answer_graph
"""Canonical alias for build_answer_graph per Blueprint §10.1."""

graph_draft_email = build_draft_graph
"""Canonical alias for build_draft_graph per Blueprint §10.3."""

graph_summarize_thread = build_summarize_graph
"""Canonical alias for build_summarize_graph per Blueprint §10.4."""