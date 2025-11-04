"""
P0-13 to P0-22 FIX: Vertex AI Agent Builder integration.

Implements Google Cloud's Discovery Engine / Agent Builder APIs for:
- Grounded generation with citations
- Fact checking and grounding verification
- Document ranking with semantic understanding
- Session management for conversational search
- Multi-turn search with context preservation

Based on official Google documentation:
- https://cloud.google.com/generative-ai-app-builder/docs/samples
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from google.api_core import retry

# Optional Discovery Engine import
try:
    from google.cloud import discoveryengine_v1beta as discoveryengine
except ImportError:
    try:
        from google.cloud import discoveryengine  # type: ignore
    except ImportError:
        discoveryengine = None  # type: ignore

from .core_config import EmailOpsConfig
from .core_exceptions import ProviderError, ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "AgentSession",
    "GroundedAnswer",
    "GroundingCheck",
    "RankedDocument",
    "VertexAgentBuilder",
]


@dataclass
class AgentSession:
    """
    P0-15 FIX: Represents a conversational search session.

    Attributes:
        session_id: Unique session identifier
        name: Full resource name (projects/{project}/locations/{location}/collections/{collection}/dataStores/{datastore}/sessions/{session})
        state: Session state dict
        created_at: When session was created
    """
    session_id: str
    name: str
    state: dict[str, Any]
    created_at: str


@dataclass
class GroundedAnswer:
    """
    P0-14 FIX: Answer with grounding information from Agent Builder.

    Attributes:
        answer: Generated answer text
        citations: List of source citations with confidence
        grounding_supports: Grounding evidence from sources
        grounding_score: Confidence score (0.0-1.0)
    """
    answer: str
    citations: list[dict[str, Any]]
    grounding_supports: list[dict[str, Any]]
    grounding_score: float


@dataclass
class GroundingCheck:
    """
    P0-16 FIX: Fact verification result.

    Attributes:
        is_grounded: Whether claim is supported by sources
        confidence: Confidence score
        supporting_sources: List of supporting document IDs
        unsupported_claims: Claims not found in sources
    """
    is_grounded: bool
    confidence: float
    supporting_sources: list[str]
    unsupported_claims: list[str]


@dataclass
class RankedDocument:
    """
    P0-17 FIX: Document with semantic ranking score.

    Attributes:
        document_id: Document identifier
        relevance_score: Semantic relevance (0.0-1.0)
        rank: Position in ranked list
        metadata: Document metadata
    """
    document_id: str
    relevance_score: float
    rank: int
    metadata: dict[str, Any]


class VertexAgentBuilder:
    """
    P0-13 to P0-22 FIX: Vertex AI Agent Builder client.

    Provides production-grade integration with Google Cloud Discovery Engine
    for grounded generation, fact checking, and conversational search.

    Usage:
        config = EmailOpsConfig.load()
        builder = VertexAgentBuilder(
            project_id=config.gcp_project,
            location=config.gcp_region,
            datastore_id="emailops-datastore"
        )

        # Grounded Q&A
        answer = builder.answer_query("What is our coverage limit?")

        # Fact checking
        check = builder.check_grounding(
            answer="Coverage limit is $1M",
            sources=[doc1, doc2]
        )
    """

    def __init__(
        self,
        project_id: str,
        location: str = "global",
        datastore_id: str | None = None,
        collection_id: str = "default_collection",
    ):
        """
        Initialize Vertex Agent Builder client.

        Args:
            project_id: GCP project ID
            location: Location (default: global)
            datastore_id: Discovery Engine datastore ID
            collection_id: Collection ID (default: default_collection)

        Raises:
            ValidationError: If required parameters missing
            ProviderError: If initialization fails
        """
        if not project_id:
            raise ValidationError("project_id is required", field="project_id")

        self.project_id = project_id
        self.location = location
        self.datastore_id = datastore_id
        self.collection_id = collection_id

        # Initialize clients
        try:
            self._search_client = discoveryengine.SearchServiceClient()
            self._conversational_search_client = discoveryengine.ConversationalSearchServiceClient()
            self._document_service_client = discoveryengine.DocumentServiceClient()

            logger.info(
                "Initialized Vertex Agent Builder for project=%s, location=%s",
                project_id,
                location
            )
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Vertex Agent Builder: {e}",
                provider="vertex_agent_builder",
                retryable=False
            ) from e

    def _get_serving_config(self) -> str:
        """Get full serving config resource name."""
        if not self.datastore_id:
            raise ValidationError("datastore_id must be set for search operations")

        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/collections/{self.collection_id}/dataStores/{self.datastore_id}"
            f"/servingConfigs/default_config"
        )

    def answer_query(
        self,
        query: str,
        session_id: str | None = None,
        *,
        max_citations: int = 5,
        temperature: float = 0.2,
    ) -> GroundedAnswer:
        """
        P0-13 FIX: Answer query with grounding from Discovery Engine.

        Uses Vertex AI Agent Builder to generate answers grounded in your
        document corpus with automatic citation extraction.

        Args:
            query: User question
            session_id: Optional session for multi-turn
            max_citations: Max citations to return
            temperature: Generation temperature

        Returns:
            GroundedAnswer with answer text, citations, grounding info

        Raises:
            ProviderError: If API call fails

        Example:
            answer = builder.answer_query("What is our D&O coverage?")
            print(answer.answer)
            for cite in answer.citations:
                print(f"  - {cite['document_id']}: {cite['snippet']}")
        """
        if not query or len(query.strip()) < 3:
            raise ValidationError("Query must be at least 3 characters", field="query")

        logger.debug(
            "answer_query request prepared", session_id=session_id, temperature=temperature
        )

        try:
            # Build request
            request = discoveryengine.AnswerQueryRequest(
                serving_config=self._get_serving_config(),
                query=discoveryengine.Query(text=query),
                session=session_id,
                answer_generation_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
                    model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
                        model_version="stable"
                    ),
                    prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
                        preamble="You are an expert assistant. Answer based solely on the provided documents."
                    ),
                    include_citations=True,
                    answer_language_code="en",
                ),
            )

            # Call API with retry
            response = self._search_client.answer_query(
                request=request,
                retry=retry.Retry(deadline=30.0)
            )

            # Extract answer and grounding
            answer_text = response.answer.answer_text if hasattr(response, "answer") else ""
            citations = []
            grounding_supports = []
            grounding_score = 0.0

            if hasattr(response.answer, "citations"):
                for cite in response.answer.citations[:max_citations]:
                    citations.append({
                        "document_id": cite.sources[0].reference_id if cite.sources else "unknown",
                        "snippet": cite.sources[0].snippet if cite.sources else "",
                        "confidence": "high" if cite.confidence > 0.8 else "medium" if cite.confidence > 0.5 else "low",
                    })

            if hasattr(response.answer, "grounding_supports"):
                for support in response.answer.grounding_supports:
                    grounding_supports.append({
                        "document_id": support.document_id if hasattr(support, "document_id") else "",
                        "segment": support.segment if hasattr(support, "segment") else "",
                        "score": float(support.confidence_score) if hasattr(support, "confidence_score") else 0.0,
                    })

            if hasattr(response.answer, "grounding_score"):
                grounding_score = float(response.answer.grounding_score)

            logger.info(
                "answer_query: query_len=%d, answer_len=%d, citations=%d, grounding_score=%.2f",
                len(query),
                len(answer_text),
                len(citations),
                grounding_score
            )

            return GroundedAnswer(
                answer=answer_text,
                citations=citations,
                grounding_supports=grounding_supports,
                grounding_score=grounding_score
            )

        except Exception as e:
            logger.error("answer_query failed: %s", e)
            raise ProviderError(
                f"Vertex Agent Builder answer_query failed: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def grounded_generation_streaming(
        self,
        query: str,
        session_id: str | None = None,
    ) -> Iterator[str]:
        """
        P0-14 FIX: Stream grounded generation responses.

        Yields answer text chunks as they're generated, enabling
        real-time UI updates.

        Args:
            query: User question
            session_id: Optional session for context

        Yields:
            Answer text chunks

        Example:
            for chunk in builder.grounded_generation_streaming("What's our policy?"):
                print(chunk, end="", flush=True)
        """
        if not query or len(query.strip()) < 3:
            raise ValidationError("Query must be at least 3 characters", field="query")

        try:
            request = discoveryengine.AnswerQueryRequest(
                serving_config=self._get_serving_config(),
                query=discoveryengine.Query(text=query),
                session=session_id,
                answer_generation_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
                    model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
                        model_version="stable"
                    ),
                    include_citations=True,
                ),
            )

            # Stream response
            for response in self._search_client.answer_query(request=request, stream=True):
                if hasattr(response, "answer") and hasattr(response.answer, "answer_text"):
                    yield response.answer.answer_text

        except Exception as e:
            logger.error("grounded_generation_streaming failed: %s", e)
            raise ProviderError(
                f"Streaming generation failed: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def create_session(
        self,
        session_id: str | None = None,
        user_pseudo_id: str | None = None,
    ) -> AgentSession:
        """
        P0-15 FIX: Create new conversational search session.

        Sessions maintain context across multiple queries for
        conversational search experiences.

        Args:
            session_id: Optional custom session ID (auto-generated if None)
            user_pseudo_id: Optional anonymous user identifier

        Returns:
            AgentSession object

        Example:
            session = builder.create_session()
            # Use session.name in subsequent queries
        """
        try:
            parent = (
                f"projects/{self.project_id}/locations/{self.location}"
                f"/collections/{self.collection_id}/dataStores/{self.datastore_id}"
            )

            session = discoveryengine.Session(
                user_pseudo_id=user_pseudo_id or "anonymous",
                state=discoveryengine.Session.State.IN_PROGRESS,
            )

            request = discoveryengine.CreateSessionRequest(
                parent=parent,
                session=session,
            )

            response = self._conversational_search_client.create_session(
                request=request,
                retry=retry.Retry(deadline=10.0)
            )

            logger.info("Created session: %s", response.name)

            generated_session_id = response.name.split("/")[-1]
            effective_session_id = session_id or generated_session_id
            session_state: dict[str, Any] = {}
            if session_id and session_id != generated_session_id:
                session_state["generated_session_id"] = generated_session_id

            return AgentSession(
                session_id=effective_session_id,
                name=response.name,
                state=session_state,
                created_at=datetime.now(UTC).isoformat()
            )

        except Exception as e:
            logger.error("create_session failed: %s", e)
            raise ProviderError(
                f"Failed to create session: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def update_session(
        self,
        session_name: str,
        state: dict[str, Any],
    ) -> AgentSession:
        """
        P0-15 FIX: Update session state.

        Args:
            session_name: Full session resource name
            state: State dict to merge

        Returns:
            Updated AgentSession
        """
        try:
            session = discoveryengine.Session(
                name=session_name,
                state=state,
            )

            request = discoveryengine.UpdateSessionRequest(
                session=session,
            )

            response = self._conversational_search_client.update_session(request=request)

            return AgentSession(
                session_id=session_name.split("/")[-1],
                name=response.name,
                state=state,
                created_at=datetime.now(UTC).isoformat()
            )

        except Exception as e:
            logger.error("update_session failed: %s", e)
            raise ProviderError(
                f"Failed to update session: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def list_sessions(
        self,
        page_size: int = 50,
    ) -> list[AgentSession]:
        """
        P0-15 FIX: List all sessions for this datastore.

        Args:
            page_size: Max sessions per page

        Returns:
            List of AgentSession objects
        """
        try:
            parent = (
                f"projects/{self.project_id}/locations/{self.location}"
                f"/collections/{self.collection_id}/dataStores/{self.datastore_id}"
            )

            request = discoveryengine.ListSessionsRequest(
                parent=parent,
                page_size=page_size,
            )

            sessions = []
            for response in self._conversational_search_client.list_sessions(request=request):
                sessions.append(AgentSession(
                    session_id=response.name.split("/")[-1],
                    name=response.name,
                    state={},
                    created_at=str(response.create_time) if hasattr(response, "create_time") else ""
                ))

            return sessions

        except Exception as e:
            logger.error("list_sessions failed: %s", e)
            raise ProviderError(
                f"Failed to list sessions: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def check_grounding(
        self,
        answer_candidate: str,
        facts: list[str],
        *,
        citation_threshold: float = 0.5,
    ) -> GroundingCheck:
        """
        P0-16 FIX: Verify answer is grounded in provided facts.

        Uses Vertex AI grounding API to check if generated answer
        is supported by source documents. Critical for fact-checking.

        Args:
            answer_candidate: Generated answer to verify
            facts: List of fact strings from source documents
            citation_threshold: Min confidence for grounding (0.0-1.0)

        Returns:
            GroundingCheck with verification results

        Example:
            check = builder.check_grounding(
                answer_candidate="Coverage limit is $1M",
                facts=["Policy doc states limit of $1M", "Another fact"]
            )
            if not check.is_grounded:
                print("UNGROUNDED CLAIMS:", check.unsupported_claims)
        """
        if not answer_candidate or not answer_candidate.strip():
            raise ValidationError("answer_candidate cannot be empty", field="answer_candidate")

        if not facts:
            raise ValidationError("facts list cannot be empty", field="facts")

        try:
            # Build grounding request
            request = discoveryengine.CheckGroundingRequest(
                grounding_config=f"projects/{self.project_id}/locations/{self.location}/groundingConfigs/default_grounding_config",
                answer_candidate=answer_candidate,
                facts=facts,
            )

            response = self._search_client.check_grounding(
                request=request,
                retry=retry.Retry(deadline=15.0)
            )

            # Parse response
            is_grounded = bool(response.grounded) if hasattr(response, "grounded") else False
            confidence = float(response.confidence) if hasattr(response, "confidence") else 0.0

            if is_grounded and confidence < citation_threshold:
                is_grounded = False

            supporting_sources = []
            if hasattr(response, "cited_chunks"):
                supporting_sources = [
                    chunk.chunk_id if hasattr(chunk, "chunk_id") else ""
                    for chunk in response.cited_chunks
                ]

            unsupported_claims = []
            if hasattr(response, "claims") and not is_grounded:
                unsupported_claims = [
                    claim.text if hasattr(claim, "text") else str(claim)
                    for claim in response.claims
                    if not getattr(claim, "grounded", False)
                ]

            logger.info(
                "check_grounding: is_grounded=%s, confidence=%.2f, supporting=%d, unsupported=%d",
                is_grounded,
                confidence,
                len(supporting_sources),
                len(unsupported_claims)
            )

            return GroundingCheck(
                is_grounded=is_grounded,
                confidence=confidence,
                supporting_sources=supporting_sources,
                unsupported_claims=unsupported_claims
            )

        except Exception as e:
            logger.error("check_grounding failed: %s", e)
            raise ProviderError(
                f"Grounding check failed: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def rank_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        top_k: int = 10,
    ) -> list[RankedDocument]:
        """
        P0-17 FIX: Rank documents by semantic relevance to query.

        Uses Vertex AI semantic ranking (not just keyword matching)
        to reorder documents by relevance.

        Args:
            query: Search query
            documents: List of dicts with 'id' and 'content' keys
            top_k: Number of top results to return

        Returns:
            List of RankedDocument objects in relevance order

        Example:
            docs = [
                {"id": "doc1", "content": "Policy covers $1M"},
                {"id": "doc2", "content": "Unrelated content"},
            ]
            ranked = builder.rank_documents("coverage limit", docs)
            # ranked[0] will be doc1 (more relevant)
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty", field="query")

        if not documents:
            return []

        try:
            # Build ranking request
            ranking_config = f"projects/{self.project_id}/locations/{self.location}/rankingConfigs/default_ranking_config"

            records = []
            for doc in documents:
                if not isinstance(doc, dict):
                    continue
                records.append(
                    discoveryengine.RankingRecord(
                        id=str(doc.get("id", "")),
                        title=str(doc.get("title", "")),
                        content=str(doc.get("content", ""))
                    )
                )

            request = discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model="semantic-ranker-512@latest",
                query=query,
                records=records,
                top_n=top_k,
            )

            response = self._search_client.rank(
                request=request,
                retry=retry.Retry(deadline=15.0)
            )

            # Parse ranked results
            ranked = []
            for idx, record in enumerate(response.records[:top_k]):
                ranked.append(RankedDocument(
                    document_id=record.id,
                    relevance_score=float(record.score) if hasattr(record, "score") else 0.0,
                    rank=idx + 1,
                    metadata={}
                ))

            logger.info("rank_documents: input=%d, output=%d", len(documents), len(ranked))

            return ranked

        except Exception as e:
            logger.error("rank_documents failed: %s", e)
            raise ProviderError(
                f"Document ranking failed: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def get_datastore_info(self) -> dict[str, Any]:
        """
        P0-18 FIX: Get datastore information.

        Returns metadata about the configured datastore including
        document count, schema, and configuration.

        Returns:
            Dict with datastore info
        """
        if not self.datastore_id:
            raise ValidationError("datastore_id must be set")

        try:
            name = (
                f"projects/{self.project_id}/locations/{self.location}"
                f"/collections/{self.collection_id}/dataStores/{self.datastore_id}"
            )

            request = discoveryengine.GetDataStoreRequest(name=name)

            datastore = self._document_service_client.get_data_store(request=request)

            return {
                "name": datastore.name,
                "display_name": datastore.display_name if hasattr(datastore, "display_name") else "",
                "content_config": str(datastore.content_config) if hasattr(datastore, "content_config") else "",
                "create_time": str(datastore.create_time) if hasattr(datastore, "create_time") else "",
            }

        except Exception as e:
            logger.error("get_datastore_info failed: %s", e)
            raise ProviderError(
                f"Failed to get datastore info: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e

    def multi_turn_search(
        self,
        query: str,
        session: AgentSession,
        *,
        page_size: int = 10,
    ) -> tuple[list[dict[str, Any]], AgentSession]:
        """
        P0-19 FIX: Perform conversational search with session context.

        Maintains conversation context across queries for natural
        follow-up questions.

        Args:
            query: Current question
            session: Existing session with history
            page_size: Max results to return

        Returns:
            Tuple of (search_results, updated_session)

        Example:
            session = builder.create_session()
            results1, session = builder.multi_turn_search("What's our D&O coverage?", session)
            results2, session = builder.multi_turn_search("What about E&O?", session)
            # Second query uses context from first
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty", field="query")

        try:
            request = discoveryengine.ConverseConversationRequest(
                name=session.name,
                query=discoveryengine.TextInput(input=query),
                serving_config=self._get_serving_config(),
            )

            response = self._conversational_search_client.converse_conversation(
                request=request,
                retry=retry.Retry(deadline=30.0)
            )

            # Extract search results
            results = []
            if hasattr(response, "search_results"):
                for result in response.search_results[:page_size]:
                    results.append({
                        "document_id": result.id if hasattr(result, "id") else "",
                        "content": result.document.struct_data if hasattr(result, "document") else {},
                        "relevance_score": float(result.relevance_score) if hasattr(result, "relevance_score") else 0.0,
                    })

            # Update session state
            updated_session = AgentSession(
                session_id=session.session_id,
                name=session.name,
                state=session.state,
                created_at=session.created_at
            )

            logger.info("multi_turn_search: query_len=%d, results=%d", len(query), len(results))

            return results, updated_session

        except Exception as e:
            logger.error("multi_turn_search failed: %s", e)
            raise ProviderError(
                f"Multi-turn search failed: {e}",
                provider="vertex_agent_builder",
                retryable=True
            ) from e


# Factory function for easy instantiation
def create_agent_builder(config: EmailOpsConfig | None = None) -> VertexAgentBuilder:
    """
    Factory function to create VertexAgentBuilder from EmailOpsConfig.

    Args:
        config: EmailOpsConfig instance (loads default if None)

    Returns:
        Configured VertexAgentBuilder instance

    Example:
        builder = create_agent_builder()
        answer = builder.answer_query("What is our coverage?")
    """
    if config is None:
        config = EmailOpsConfig.load()

    return VertexAgentBuilder(
        project_id=config.gcp.gcp_project,
        location=config.gcp.gcp_region,
        datastore_id=None,  # Must be configured by caller
    )
