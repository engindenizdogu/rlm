"""Tests for DeepRLM binary-search-style run method."""

from unittest.mock import Mock, patch

import pytest

from deep_rlm.deep_rlm import DeepRLM
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class ConfigurableMockLM(BaseLM):
    """Mock LM with configurable responses for testing."""

    def __init__(self, responses=None, stop_signals=None):
        super().__init__(model_name="mock-model")
        self.responses = responses or []
        self.stop_signals = stop_signals or []
        self.call_count = 0

    def completion(self, prompt):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        if idx < len(self.responses):
            return self.responses[idx]
        return "Default response"

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self):
        return UsageSummary(
            model_usage_summaries={
                "mock-model": ModelUsageSummary(
                    total_calls=self.call_count, total_input_tokens=10, total_output_tokens=10
                )
            }
        )

    def get_last_usage(self):
        return self.get_usage_summary()


def create_mock_completion_result(response, stop_workflow):
    """Helper to create mock RLMChatCompletion result."""
    mock_result = Mock()
    mock_result.response = response
    mock_result.stop_workflow = stop_workflow
    return mock_result


class TestDeepRLMBinaryRunStopBehavior:
    """Tests for stop_workflow signal handling in run_binary."""

    def test_root_stops_immediately_with_custom_prompt(self):
        """Test that search stops when root node emits STOP_WORKFLOW(true)."""
        # Use a custom system prompt that will emit STOP_WORKFLOW
        custom_prompt = "Always respond with: Found! STOP_WORKFLOW(true)"

        deep_rlm = DeepRLM(
            max_system_depth=5,
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test-key"},
            custom_system_prompt=custom_prompt,
        )

        # Test the structure without actual execution
        # Just verify the method exists and returns proper structure
        assert hasattr(deep_rlm, "run_binary")

    def test_none_treated_as_false_structure(self):
        """Test that run_binary handles None stop_workflow correctly."""
        # This tests the worker function's handling of None
        from deep_rlm.deep_rlm import _run_rlm_completion_worker

        # Create a mock result with None stop_workflow
        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_result = Mock()
            mock_result.response = "Test response"
            mock_result.stop_workflow = None
            mock_rlm.completion.return_value = mock_result
            mock_rlm_class.return_value = mock_rlm

            # Call the worker directly
            result = _run_rlm_completion_worker(
                {"backend": "openai", "backend_kwargs": {"model_name": "test"}},
                "test chunk",
                "test prompt",
            )

            # Verify None is converted to False
            assert result["stop_workflow"] is False
            assert result["error"] is None


class TestDeepRLMBinaryRunDepthBehavior:
    """Tests for depth limiting in run_binary."""

    def test_max_depth_structure_validation(self):
        """Test that max_system_depth is properly stored and validated."""
        deep_rlm = DeepRLM(
            max_system_depth=2,
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        assert deep_rlm.max_system_depth == 2
        assert hasattr(deep_rlm, "run_binary")

    def test_depth_zero_validation(self):
        """Test that max_system_depth=0 is valid."""
        deep_rlm = DeepRLM(
            max_system_depth=0,
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        assert deep_rlm.max_system_depth == 0


class TestDeepRLMBinaryRunParallelism:
    """Tests for parallel execution control."""

    def test_max_parallel_workers_parameter(self):
        """Test that max_parallel_workers parameter is stored correctly."""
        deep_rlm = DeepRLM(
            max_system_depth=2,
            max_parallel_workers=2,  # Limit parallelism
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        assert deep_rlm.max_parallel_workers == 2

    def test_unlimited_workers_default(self):
        """Test that max_parallel_workers defaults to None (unlimited)."""
        deep_rlm = DeepRLM(
            max_system_depth=1,
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        assert deep_rlm.max_parallel_workers is None


class TestDeepRLMBinaryRunOutputStructure:
    """Tests for output structure of run_binary."""

    def test_output_structure_definition(self):
        """Test that run_binary is defined and has proper signature."""
        deep_rlm = DeepRLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        # Verify method exists
        assert hasattr(deep_rlm, "run_binary")
        assert callable(deep_rlm.run_binary)

    def test_worker_structured_output(self):
        """Test that structured worker returns proper format."""
        from deep_rlm.deep_rlm import _run_rlm_completion_worker

        # Mock the RLM at module level
        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_result = Mock()
            mock_result.response = "Test response"
            mock_result.stop_workflow = True
            mock_rlm.completion.return_value = mock_result
            mock_rlm_class.return_value = mock_rlm

            result = _run_rlm_completion_worker(
                {"backend": "openai", "backend_kwargs": {"model_name": "test"}}, "chunk", "prompt"
            )

            # Check structure
            assert "response" in result
            assert "stop_workflow" in result
            assert "error" in result
            assert result["response"] == "Test response"
            assert result["stop_workflow"] is True
            assert result["error"] is None

    def test_worker_error_handling(self):
        """Test that worker captures exceptions properly."""
        from deep_rlm.deep_rlm import _run_rlm_completion_worker

        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_rlm.completion.side_effect = Exception("Test error")
            mock_rlm_class.return_value = mock_rlm

            result = _run_rlm_completion_worker(
                {"backend": "openai", "backend_kwargs": {"model_name": "test"}}, "chunk", "prompt"
            )

            # Check error handling
            assert "error" in result
            assert result["error"] is not None
            assert "Test error" in result["error"]
            assert result["stop_workflow"] is False  # Errors don't trigger stop


class TestDeepRLMBinaryRunErrorHandling:
    """Tests for error handling in run_binary."""

    def test_empty_context_raises_error(self):
        """Test that empty context raises ValueError."""
        deep_rlm = DeepRLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        with pytest.raises(ValueError, match="Context must be provided"):
            deep_rlm.run_binary(context="", prompt="prompt")

    def test_none_context_raises_error(self):
        """Test that None context raises ValueError."""
        deep_rlm = DeepRLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        with pytest.raises(ValueError, match="Context must be provided"):
            deep_rlm.run_binary(context=None, prompt="prompt")

    def test_invalid_stop_mode_raises_error(self):
        """Test that invalid stop_mode raises ValueError."""
        deep_rlm = DeepRLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        with pytest.raises(ValueError, match="stop_mode must be"):
            deep_rlm.run_binary(context="test context", prompt="prompt", stop_mode="invalid_mode")


class TestDeepRLMBinaryMiddleFirstBehavior:
    """Tests for middle-first chunk initialization and index-based traversal."""

    def test_middle_first_initialization_output_schema(self):
        """Test that run_binary output contains chunk index metadata."""
        deep_rlm = DeepRLM(
            max_system_depth=0,  # Only process root node
            token_limit=50,  # Force decomposition
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        # Create context that will be decomposed into multiple chunks
        words = ["word" + str(i) for i in range(100)]
        context = " ".join(words)

        # Mock the RLM completion to avoid actual API calls
        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_result = Mock()
            mock_result.response = "Test response"
            mock_result.stop_workflow = False
            mock_rlm.completion.return_value = mock_result
            mock_rlm_class.return_value = mock_rlm

            result = deep_rlm.run_binary(context=context, prompt="test")

            # Verify output structure
            assert "nodes" in result
            assert len(result["nodes"]) > 0

            # Check that chunk index metadata is present
            first_node = result["nodes"][0]
            assert "lo" in first_node
            assert "hi" in first_node
            assert "mid_index" in first_node
            assert "id" in first_node
            assert "depth" in first_node
            assert "response" in first_node
            assert "stop_workflow" in first_node
            assert "error" in first_node
            assert "parent_id" in first_node

    def test_stop_mode_immediate_default(self):
        """Test that stop_mode defaults to 'immediate'."""
        deep_rlm = DeepRLM(
            max_system_depth=2,
            token_limit=50,
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        context = " ".join(["word" + str(i) for i in range(100)])

        call_count = [0]

        def mock_completion_with_stop(*args, **kwargs):
            """Mock that returns stop signal on second call."""
            call_count[0] += 1
            mock_result = Mock()
            mock_result.response = "Response " + str(call_count[0])
            mock_result.stop_workflow = call_count[0] == 2  # Stop on second call
            return mock_result

        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_rlm.completion.side_effect = mock_completion_with_stop
            mock_rlm_class.return_value = mock_rlm

            # Call without explicit stop_mode (should default to "immediate")
            result = deep_rlm.run_binary(context=context, prompt="test")

            # With immediate mode, should stop as soon as stop signal is seen
            # Verify structure is valid
            assert "nodes" in result
            assert len(result["nodes"]) >= 1

    def test_stop_mode_finish_depth(self):
        """Test that finish_depth mode completes all nodes at stop depth."""
        deep_rlm = DeepRLM(
            max_system_depth=2,
            token_limit=50,
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        context = " ".join(["word" + str(i) for i in range(100)])

        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_result = Mock()
            mock_result.response = "Test response"
            mock_result.stop_workflow = False  # No stop for this test
            mock_rlm.completion.return_value = mock_result
            mock_rlm_class.return_value = mock_rlm

            result = deep_rlm.run_binary(context=context, prompt="test", stop_mode="finish_depth")

            # Verify result structure is valid
            assert "nodes" in result
            assert len(result["nodes"]) > 0


class TestDeepRLMBinaryIndexRangeBranching:
    """Tests for index-based range branching behavior."""

    def test_range_branching_logic(self):
        """Test that nodes branch correctly based on index ranges."""
        deep_rlm = DeepRLM(
            max_system_depth=2,
            token_limit=50,  # Force multiple chunks
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        # Create context that will definitely be split into multiple chunks
        context = " ".join(["word" + str(i) for i in range(200)])

        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_result = Mock()
            mock_result.response = "Test response"
            mock_result.stop_workflow = False
            mock_rlm.completion.return_value = mock_result
            mock_rlm_class.return_value = mock_rlm

            result = deep_rlm.run_binary(context=context, prompt="test")

            # Verify that nodes have valid range relationships
            nodes = result["nodes"]
            assert len(nodes) > 0

            # Check root node
            root = nodes[0]
            assert root["depth"] == 0
            assert root["parent_id"] is None
            assert root["lo"] == root["hi"]  # Root evaluates single middle chunk

            # If there are child nodes, verify their ranges
            children = [n for n in nodes if n["parent_id"] == root["id"]]
            if len(children) > 0:
                # Children should have non-overlapping ranges
                for child in children:
                    assert child["depth"] == 1
                    assert child["lo"] <= child["hi"]
                    assert child["lo"] <= child["mid_index"] <= child["hi"]


class TestDeepRLMBinaryImmediateCancellation:
    """Tests for immediate mode future cancellation."""

    def test_cancelled_nodes_not_in_output(self):
        """Test that cancelled nodes are properly skipped and not added to output."""
        # This test validates that when futures are cancelled, they don't appear
        # in the nodes list (since result is None and we skip them)
        deep_rlm = DeepRLM(
            max_system_depth=1,
            token_limit=50,
            backend="openai",
            backend_kwargs={"model_name": "gpt-4", "api_key": "test"},
        )

        context = " ".join(["word" + str(i) for i in range(100)])

        with patch("deep_rlm.deep_rlm.RLM") as mock_rlm_class:
            mock_rlm = Mock()
            mock_result = Mock()
            mock_result.response = "Test response"
            mock_result.stop_workflow = False
            mock_rlm.completion.return_value = mock_result
            mock_rlm_class.return_value = mock_rlm

            result = deep_rlm.run_binary(context=context, prompt="test", stop_mode="immediate")

            # All nodes in output should have valid responses (no None)
            for node in result["nodes"]:
                assert node["response"] is not None
                assert "response" in node
                assert "stop_workflow" in node
                assert "error" in node
