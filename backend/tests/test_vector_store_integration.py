"""
Integration tests for VectorStore against the real ChromaDB on disk.

These tests use the actual backend/chroma_db (no mocks) to verify that
the vector store populated from the docs/ folder can execute queries
without errors.

Run from the project root:
    uv run pytest backend/tests/test_vector_store_integration.py -v
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults

# Resolve the chroma_db path relative to backend/
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BACKEND_DIR, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RESULTS = 5


@pytest.fixture(scope="module")
def store():
    """Real VectorStore pointing at the persisted chroma_db directory."""
    if not os.path.exists(CHROMA_PATH):
        pytest.skip(f"chroma_db not found at {CHROMA_PATH} — run the app first to ingest docs")
    return VectorStore(CHROMA_PATH, EMBEDDING_MODEL, MAX_RESULTS)


# ─── Sanity: data is present ──────────────────────────────────────────────────

class TestDataPresence:

    def test_at_least_one_course_ingested(self, store):
        count = store.get_course_count()
        assert count > 0, (
            "No courses in ChromaDB. Start the app once so docs/ are ingested, "
            "then re-run these tests."
        )

    def test_course_titles_non_empty(self, store):
        titles = store.get_existing_course_titles()
        assert len(titles) > 0


# ─── search() without filters ─────────────────────────────────────────────────

class TestSearchNoFilter:

    def test_search_returns_SearchResults_instance(self, store):
        result = store.search("What is this course about?")
        assert isinstance(result, SearchResults)

    def test_search_has_no_error(self, store):
        result = store.search("introduction to the course")
        assert result.error is None, f"Unexpected search error: {result.error}"

    def test_search_returns_documents(self, store):
        result = store.search("introduction to the course")
        assert not result.is_empty(), "Expected search results but got none"

    def test_documents_and_metadata_lengths_match(self, store):
        result = store.search("lesson content")
        assert len(result.documents) == len(result.metadata)
        assert len(result.documents) == len(result.distances)

    def test_metadata_contains_course_title(self, store):
        result = store.search("course overview")
        for meta in result.metadata:
            assert "course_title" in meta, f"Metadata missing 'course_title': {meta}"

    def test_n_results_does_not_exceed_max(self, store):
        result = store.search("anything")
        assert len(result.documents) <= MAX_RESULTS


# ─── search() with n_results > collection size ───────────────────────────────

class TestNResultsEdgeCase:

    def test_search_with_large_n_results_does_not_raise(self, store):
        """
        ChromaDB raises ValueError when n_results > number of elements.
        VectorStore.search() must catch this and return SearchResults.empty().
        This is a known crash path that shows as 'query failed' in the UI.
        """
        # Request far more results than any collection would have
        result = store.search("test query", limit=9999)
        # Should NOT propagate an exception — must return SearchResults
        assert isinstance(result, SearchResults)
        # Either an error message OR valid documents — not an uncaught exception
        if result.error:
            # Error was caught and wrapped — acceptable
            assert "Search error" in result.error or len(result.error) > 0
        else:
            assert isinstance(result.documents, list)


# ─── search() with course_name filter ────────────────────────────────────────

class TestSearchWithCourseFilter:

    def test_search_with_valid_course_name(self, store):
        """Filter by the first known course title — should return results."""
        titles = store.get_existing_course_titles()
        if not titles:
            pytest.skip("No courses in store")

        first_title = titles[0]
        result = store.search("content", course_name=first_title)
        assert isinstance(result, SearchResults)
        assert result.error is None, f"Unexpected error: {result.error}"

    def test_search_with_unknown_course_name_returns_error(self, store):
        """Querying a non-existent course should return SearchResults.empty() with error."""
        result = store.search("anything", course_name="NONEXISTENT_COURSE_XYZ_123")
        assert isinstance(result, SearchResults)
        # Expect either an error message or empty results
        assert result.error is not None or result.is_empty()

    def test_results_belong_to_filtered_course(self, store):
        """All returned chunks must belong to the requested course."""
        titles = store.get_existing_course_titles()
        if not titles:
            pytest.skip("No courses in store")

        first_title = titles[0]
        result = store.search("lesson", course_name=first_title)
        if result.is_empty() or result.error:
            pytest.skip(f"No results for course '{first_title}'")

        for meta in result.metadata:
            assert meta.get("course_title") == first_title, (
                f"Expected course_title='{first_title}' but got '{meta.get('course_title')}'"
            )


# ─── get_lesson_link ──────────────────────────────────────────────────────────

class TestGetLessonLink:

    def test_get_lesson_link_does_not_raise(self, store):
        """Should return None or a string, never raise."""
        result = store.get_lesson_link("Any Course Title", 1)
        assert result is None or isinstance(result, str)
