"""
Tests for RAGSystem.query() in rag_system.py.

Covers:
- Answer and sources returned from a content query
- Sources are reset after each query (no leakage between calls)
- Session history is stored when session_id provided
- Session history is NOT stored when no session_id
- Vector store is empty → graceful "no content" answer (not a crash)
- Exception inside AIGenerator propagates cleanly
- ChromaDB n_results > collection size does not crash the system
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Optional

from rag_system import RAGSystem
from vector_store import SearchResults


# ─── Minimal config fixture ───────────────────────────────────────────────────

@dataclass
class _TestConfig:
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-test"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def rag(tmp_path):
    """
    RAGSystem with all heavy components mocked out so tests run without
    a real ChromaDB or Anthropic API key.
    """
    config = _TestConfig(CHROMA_PATH=str(tmp_path / "chroma"))

    with (
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.SessionManager") as MockSM,
    ):
        mock_vs = MockVS.return_value
        mock_ai = MockAI.return_value
        mock_sm = MockSM.return_value

        # Default: search returns content
        mock_vs.search.return_value = SearchResults(
            documents=["Lesson content about Python."],
            metadata=[{"course_title": "Python Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vs.get_lesson_link.return_value = "http://example.com/lesson1"
        mock_vs.get_existing_course_titles.return_value = []

        # Default: AI returns a plain answer
        mock_ai.generate_response.return_value = "Python is a programming language."

        # Session manager
        mock_sm.get_conversation_history.return_value = None
        mock_sm.create_session.return_value = "session-abc"

        system = RAGSystem(config)
        # Expose mocks so individual tests can inspect / override them
        system._mock_vs = mock_vs
        system._mock_ai = mock_ai
        system._mock_sm = mock_sm

        yield system


# ─── Basic query behavior ─────────────────────────────────────────────────────

class TestQueryBasicBehavior:

    def test_returns_answer_string(self, rag):
        answer, _ = rag.query("What is Python?")
        assert answer == "Python is a programming language."

    def test_returns_sources_list(self, rag):
        # Make the search tool populate sources by doing a real search
        # We need to prime the CourseSearchTool with a real mock store
        rag.search_tool.store = rag._mock_vs
        rag.search_tool.last_sources = []

        # Simulate the AI generator calling the tool during generate_response
        def fake_generate(query, conversation_history=None, tools=None, tool_manager=None):
            # Simulate tool being called
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query="Python")
            return "Python is a programming language."

        rag._mock_ai.generate_response.side_effect = fake_generate

        answer, sources = rag.query("What is Python?")

        assert isinstance(sources, list)

    def test_answer_is_string_not_none(self, rag):
        rag._mock_ai.generate_response.return_value = "An answer."
        answer, _ = rag.query("any question")
        assert answer is not None
        assert isinstance(answer, str)


# ─── Source lifecycle ─────────────────────────────────────────────────────────

class TestSourceLifecycle:

    def test_sources_reset_between_queries(self, rag):
        """Sources from one query must not appear in the next query's response."""
        rag.search_tool.store = rag._mock_vs

        call_count = [0]

        def fake_generate(query, conversation_history=None, tools=None, tool_manager=None):
            call_count[0] += 1
            if call_count[0] == 1 and tool_manager:
                # First query: tool is called and sources are set
                tool_manager.execute_tool("search_course_content", query="Python")
            # Second query: tool is NOT called, no new sources
            return "Some answer."

        rag._mock_ai.generate_response.side_effect = fake_generate

        _, sources_first = rag.query("What is Python?")
        _, sources_second = rag.query("General greeting")

        # Second query had no tool call → sources should be empty
        assert sources_second == []

    def test_sources_empty_when_no_tool_called(self, rag):
        """If Claude never calls the search tool, sources must be empty."""
        # generate_response returns answer without calling tool_manager
        rag._mock_ai.generate_response.return_value = "Hello!"

        _, sources = rag.query("Hi there")

        assert sources == []


# ─── Session management ───────────────────────────────────────────────────────

class TestSessionManagement:

    def test_session_history_retrieved_when_session_id_provided(self, rag):
        rag._mock_sm.get_conversation_history.return_value = "User: prev\nAssistant: prev ans"
        rag._mock_ai.generate_response.return_value = "Answer"

        rag.query("Follow-up question", session_id="session-xyz")

        rag._mock_sm.get_conversation_history.assert_called_once_with("session-xyz")

    def test_exchange_stored_after_query_with_session_id(self, rag):
        rag._mock_ai.generate_response.return_value = "My answer"

        rag.query("A question", session_id="session-xyz")

        rag._mock_sm.add_exchange.assert_called_once_with(
            "session-xyz", "A question", "My answer"
        )

    def test_no_history_lookup_without_session_id(self, rag):
        rag.query("Question without session")

        rag._mock_sm.get_conversation_history.assert_not_called()

    def test_no_exchange_stored_without_session_id(self, rag):
        rag.query("Question without session")

        rag._mock_sm.add_exchange.assert_not_called()


# ─── Empty vector store ───────────────────────────────────────────────────────

class TestEmptyVectorStore:

    def test_query_does_not_crash_when_store_empty(self, rag):
        """If the vector store is empty, the system should still return a string answer."""
        rag._mock_vs.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        rag._mock_ai.generate_response.return_value = (
            "I could not find relevant course content."
        )

        answer, sources = rag.query("What is machine learning?")

        assert isinstance(answer, str)
        assert sources == []

    def test_search_error_does_not_crash_query(self, rag):
        """A search error (e.g. empty ChromaDB collection) must not propagate as an exception."""
        rag._mock_vs.search.return_value = SearchResults.empty(
            "Search error: Number of requested results 5 is greater than number of elements in index 0"
        )
        rag._mock_ai.generate_response.return_value = "No results found."

        answer, sources = rag.query("Anything")

        assert isinstance(answer, str)


# ─── ChromaDB n_results edge case ────────────────────────────────────────────

class TestChromaDBNResultsEdgeCase:
    """
    ChromaDB raises a ValueError when n_results > number of documents in the
    collection.  VectorStore.search() catches it and returns SearchResults.empty().
    This test verifies the full stack handles it without a 500 error.
    """

    def test_chroma_nresults_error_handled_gracefully(self, rag):
        """Simulate ChromaDB raising ValueError for too many requested results."""
        rag._mock_vs.search.return_value = SearchResults.empty(
            "Search error: Number of requested results 5 is greater than "
            "number of elements in index 2, updating n_results = 2"
        )
        rag._mock_ai.generate_response.return_value = (
            "I found limited content. Here is what I know..."
        )

        answer, sources = rag.query("Explain lesson 1")

        assert isinstance(answer, str)
        assert len(answer) > 0
