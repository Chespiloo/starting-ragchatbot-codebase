"""
Tests for CourseSearchTool.execute() in search_tools.py.

Covers:
- Basic search returning formatted results
- Empty results handling
- Search error propagation
- Source tracking and deduplication
- Filter arguments (course_name, lesson_number) passed to vector store
- ToolManager source collection and reset
"""
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


def make_results(docs, metas):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
    )


# ─── Basic search behavior ────────────────────────────────────────────────────

class TestExecuteBasicBehavior:

    def test_returns_formatted_text_on_match(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Python is a high-level language."],
            [{"course_title": "Python Basics", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "http://example.com/lesson1"

        result = tool.execute("What is Python?")

        assert "Python Basics" in result
        assert "Python is a high-level language." in result

    def test_returns_no_content_message_when_results_empty(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        result = tool.execute("What is Python?")

        assert "No relevant content found" in result

    def test_returns_error_string_when_search_errors(self, tool, mock_store):
        mock_store.search.return_value = SearchResults.empty(
            "Search error: collection has 0 items"
        )

        result = tool.execute("What is Python?")

        assert "Search error" in result

    def test_multiple_chunks_all_appear_in_output(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Chunk A content.", "Chunk B content."],
            [
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": 2},
            ],
        )
        mock_store.get_lesson_link.return_value = None

        result = tool.execute("overview")

        assert "Chunk A content." in result
        assert "Chunk B content." in result


# ─── Source tracking ─────────────────────────────────────────────────────────

class TestSourceTracking:

    def test_last_sources_populated_after_search(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Some content"],
            [{"course_title": "Python Basics", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "http://example.com/lesson1"

        tool.execute("What is Python?")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["name"] == "Python Basics - Lesson 1"
        assert tool.last_sources[0]["url"] == "http://example.com/lesson1"

    def test_duplicate_lesson_sources_are_deduplicated(self, tool, mock_store):
        """Two chunks from the same lesson should yield only one source entry."""
        mock_store.search.return_value = make_results(
            ["Chunk 1", "Chunk 2"],
            [
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 1},
            ],
        )
        mock_store.get_lesson_link.return_value = "http://example.com/lesson1"

        tool.execute("Python")

        assert len(tool.last_sources) == 1

    def test_sources_from_different_lessons_all_tracked(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            ["Chunk 1", "Chunk 2"],
            [
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2},
            ],
        )
        mock_store.get_lesson_link.side_effect = [
            "http://example.com/lesson1",
            "http://example.com/lesson2",
        ]

        tool.execute("Python")

        assert len(tool.last_sources) == 2

    def test_last_sources_empty_when_no_results(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        tool.execute("Python")

        assert tool.last_sources == []

    def test_source_name_without_lesson_number(self, tool, mock_store):
        """Chunks with no lesson_number should still produce a source entry."""
        mock_store.search.return_value = make_results(
            ["General content"],
            [{"course_title": "Python Basics"}],  # no lesson_number key
        )

        tool.execute("Python")

        # Should not crash; source name is just the course title
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["name"] == "Python Basics"


# ─── Filter arguments forwarded to vector store ──────────────────────────────

class TestFilterArguments:

    def test_course_name_forwarded_to_store(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        tool.execute("overview", course_name="Python Basics")

        mock_store.search.assert_called_once_with(
            query="overview",
            course_name="Python Basics",
            lesson_number=None,
        )

    def test_lesson_number_forwarded_to_store(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        tool.execute("intro", lesson_number=3)

        mock_store.search.assert_called_once_with(
            query="intro",
            course_name=None,
            lesson_number=3,
        )

    def test_empty_result_with_course_filter_mentions_course_name(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        result = tool.execute("overview", course_name="Unknown Course")

        assert "Unknown Course" in result

    def test_empty_result_with_lesson_filter_mentions_lesson(self, tool, mock_store):
        mock_store.search.return_value = make_results([], [])

        result = tool.execute("intro", lesson_number=7)

        assert "7" in result


# ─── ToolManager integration ──────────────────────────────────────────────────

class TestToolManager:

    def test_get_last_sources_returns_tool_sources(self, mock_store):
        tool = CourseSearchTool(mock_store)
        tool.last_sources = [{"name": "Course A - Lesson 1", "url": None}]

        manager = ToolManager()
        manager.register_tool(tool)

        assert manager.get_last_sources() == [{"name": "Course A - Lesson 1", "url": None}]

    def test_reset_sources_clears_tool_sources(self, mock_store):
        tool = CourseSearchTool(mock_store)
        tool.last_sources = [{"name": "Course A - Lesson 1", "url": None}]

        manager = ToolManager()
        manager.register_tool(tool)
        manager.reset_sources()

        assert tool.last_sources == []
        assert manager.get_last_sources() == []

    def test_execute_tool_routes_to_correct_tool(self, mock_store):
        mock_store.search.return_value = make_results(
            ["Result content"],
            [{"course_title": "Some Course", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)
        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test query")

        assert "Result content" in result

    def test_execute_unknown_tool_returns_error(self, mock_store):
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()
