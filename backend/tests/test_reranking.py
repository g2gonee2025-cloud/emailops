from cortex.retrieval.reranking import _candidate_summary_text, rerank_results
from cortex.retrieval.results import SearchResultItem


def test_candidate_summary_text_full_metadata():
    item = SearchResultItem(
        chunk_id="1",
        score=0.5,
        content="Meeting at 5",
        metadata={
            "sender": "alice@example.com",
            "date": "2023-10-27",
            "subject": "Project Sync",
        },
    )
    result = _candidate_summary_text(item)
    expected = "From: alice@example.com | Date: 2023-10-27 | Subject: Project Sync | Content: Meeting at 5"
    assert result == expected


def test_candidate_summary_text_partial_metadata():
    item = SearchResultItem(
        chunk_id="2", score=0.5, content="Just content", metadata={"subject": "Hello"}
    )
    result = _candidate_summary_text(item)
    expected = "Subject: Hello | Content: Just content"
    assert result == expected


def test_rerank_results_alpha_blending():
    items = [
        SearchResultItem(chunk_id="1", score=0, lexical_score=1.0, vector_score=0.0),
        SearchResultItem(chunk_id="2", score=0, lexical_score=0.0, vector_score=1.0),
    ]

    # Alpha 0.5 -> Should be equal (0.5 each)
    reranked = rerank_results(items, alpha=0.5)
    assert reranked[0].rerank_score == 0.5
    assert reranked[1].rerank_score == 0.5

    # Alpha 1.0 -> Vector wins (Item 2)
    reranked_mem = rerank_results(items, alpha=1.0)
    # Item 2 has vector=1.0 -> score 1.0
    # Item 1 has vector=0.0 -> score 0.0
    assert reranked_mem[0].chunk_id == "2"
    assert reranked_mem[0].rerank_score == 1.0
