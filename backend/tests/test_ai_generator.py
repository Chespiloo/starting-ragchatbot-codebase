"""
Tests for AIGenerator in ai_generator.py.

Covers:
- Direct text response (no tools)
- Tool-use flow: Claude triggers tool, result fed back, final answer returned
- Tool result messages are structured correctly for the Anthropic API
- Second API call (after tool execution) does NOT include tools parameter
- Behaviour when stop_reason is tool_use but tool_manager is absent
"""
import pytest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator


@pytest.fixture
def generator():
    return AIGenerator(api_key="test-key", model="claude-test-model")


def make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name: str, tool_id: str, input_dict: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input_dict
    return block


def make_response(stop_reason: str, *content_blocks):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = list(content_blocks)
    return resp


# ─── Direct (no-tool) responses ──────────────────────────────────────────────

class TestDirectResponse:

    def test_returns_text_when_no_tool_use(self, generator):
        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.return_value = make_response(
                "end_turn", make_text_block("Direct answer")
            )

            result = generator.generate_response("What is 2+2?")

        assert result == "Direct answer"
        assert mock_create.call_count == 1

    def test_api_called_with_correct_model(self, generator):
        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.return_value = make_response(
                "end_turn", make_text_block("OK")
            )

            generator.generate_response("test")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-test-model"

    def test_conversation_history_injected_into_system_prompt(self, generator):
        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.return_value = make_response(
                "end_turn", make_text_block("OK")
            )

            generator.generate_response("test", conversation_history="User: hello\nAssistant: hi")

        system = mock_create.call_args.kwargs["system"]
        assert "hello" in system


# ─── Tool-use flow ────────────────────────────────────────────────────────────

class TestToolUseFlow:

    def test_tool_executed_when_claude_requests_it(self, generator):
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search result text"

        tool_call = make_tool_use_block(
            "search_course_content", "toolu_001", {"query": "Python intro"}
        )
        first_response = make_response("tool_use", tool_call)
        second_response = make_response("end_turn", make_text_block("Final answer"))

        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.side_effect = [first_response, second_response]

            result = generator.generate_response(
                "Tell me about Python",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

        assert result == "Final answer"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python intro"
        )

    def test_two_api_calls_made_during_tool_use(self, generator):
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        first_response = make_response(
            "tool_use",
            make_tool_use_block("search_course_content", "toolu_001", {"query": "test"}),
        )
        second_response = make_response("end_turn", make_text_block("Done"))

        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.side_effect = [first_response, second_response]

            generator.generate_response(
                "test",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

        assert mock_create.call_count == 2

    def test_second_call_does_not_include_tools(self, generator):
        """After tool execution the second API call must NOT pass tools — prevents infinite loop."""
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        first_response = make_response(
            "tool_use",
            make_tool_use_block("search_course_content", "toolu_001", {"query": "test"}),
        )
        second_response = make_response("end_turn", make_text_block("Done"))

        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.side_effect = [first_response, second_response]

            generator.generate_response(
                "test",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

        second_call_kwargs = mock_create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs

    def test_tool_result_message_structure_is_valid(self, generator):
        """
        The messages list sent in the second call must contain:
        1. The original user message
        2. The assistant's tool-use response
        3. A user message with type=tool_result carrying the tool output
        """
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Fetched course content"

        tool_block = make_tool_use_block(
            "search_course_content", "toolu_abc", {"query": "intro"}
        )
        first_response = make_response("tool_use", tool_block)
        second_response = make_response("end_turn", make_text_block("Answer"))

        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.side_effect = [first_response, second_response]

            generator.generate_response(
                "What is the intro?",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

        messages = mock_create.call_args_list[1].kwargs["messages"]

        # Message 0: original user query
        assert messages[0]["role"] == "user"
        assert "What is the intro?" in messages[0]["content"]

        # Message 1: assistant tool-use
        assert messages[1]["role"] == "assistant"

        # Message 2: tool result
        assert messages[2]["role"] == "user"
        tool_result_content = messages[2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "toolu_abc"
        assert tool_result_content[0]["content"] == "Fetched course content"

    def test_multiple_tool_calls_all_executed(self, generator):
        """If Claude returns two tool_use blocks, both must be executed."""
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        block1 = make_tool_use_block("search_course_content", "t1", {"query": "A"})
        block2 = make_tool_use_block("search_course_content", "t2", {"query": "B"})
        first_response = make_response("tool_use", block1, block2)
        second_response = make_response("end_turn", make_text_block("Answer"))

        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.side_effect = [first_response, second_response]

            generator.generate_response(
                "test",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

        assert mock_tool_manager.execute_tool.call_count == 2


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_tool_use_without_tool_manager_does_not_crash(self, generator):
        """
        If Claude returns stop_reason=tool_use but no tool_manager was provided,
        the generator should not raise an AttributeError on a non-text content block.
        """
        tool_block = make_tool_use_block("search_course_content", "t1", {"query": "test"})
        response = make_response("tool_use", tool_block)

        with patch.object(generator.client.messages, "create") as mock_create:
            mock_create.return_value = response

            # Without tool_manager the code falls through to `response.content[0].text`
            # on a tool_use block — which has no .text attribute → AttributeError
            try:
                result = generator.generate_response("test query")
                # If it reaches here, result should at least be a string
                assert isinstance(result, str)
            except AttributeError as exc:
                pytest.fail(
                    f"AIGenerator raised AttributeError when stop_reason=tool_use "
                    f"but no tool_manager provided: {exc}"
                )
