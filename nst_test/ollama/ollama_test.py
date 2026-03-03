"""
Test file for Ollama client integration.

SETUP REQUIRED:
1. Install Ollama from https://ollama.ai
2. Start Ollama server: ollama serve (runs on http://localhost:11434)
3. Pull the required model: ollama pull qwen2.5:7b
4. Run this file: python nst/ollama_test.py
"""

import asyncio

from rlm.clients.ollama import OllamaClient


def test_basic_completion():
    """Test basic string prompt completion."""
    print("\n=== Test: Basic Completion ===")
    client = OllamaClient(model_name="qwen2.5:7b")

    prompt = "What is 2 + 2?"
    response = client.completion(prompt)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ Basic completion test passed")


def test_message_list_completion():
    """Test completion with message list format."""
    print("\n=== Test: Message List Completion ===")
    client = OllamaClient(model_name="qwen2.5:7b")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"},
    ]
    response = client.completion(messages)

    print(f"Messages: {messages}")
    print(f"Response: {response}")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ Message list completion test passed")


def test_model_override():
    """Test model name override at completion time."""
    print("\n=== Test: Model Override ===")
    client = OllamaClient(model_name="qwen2.5:7b")

    # This should use qwen2.5:7b even though we try to override
    # (if you have another model, you can test with it)
    prompt = "Who are you?"
    response = client.completion(prompt, model="qwen2.5:7b")

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ Model override test passed")


async def test_async_completion():
    """Test async completion."""
    print("\n=== Test: Async Completion ===")
    client = OllamaClient(model_name="qwen2.5:7b")

    prompt = "List three colors"
    response = await client.acompletion(prompt)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ Async completion test passed")


def test_multi_turn_conversation():
    """Test multi-turn conversation."""
    print("\n=== Test: Multi-Turn Conversation ===")
    client = OllamaClient(model_name="qwen2.5:7b")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ]
    response = client.completion(messages)

    print(f"Messages: {messages}")
    print(f"Response: {response}")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✓ Multi-turn conversation test passed")


def test_usage_tracking():
    """Test usage tracking."""
    print("\n=== Test: Usage Tracking ===")
    client = OllamaClient(model_name="qwen2.5:7b")

    # Make a completion
    response = client.completion("Hello!")
    print(f"Response: {response}")

    # Get last usage
    last_usage = client.get_last_usage()
    print(f"Last usage: {last_usage}")
    assert last_usage.total_calls == 1

    # Make another completion
    response = client.completion("How are you?")
    print(f"Response: {response}")

    # Get total usage summary
    summary = client.get_usage_summary()
    print(f"Usage summary: {summary}")
    assert len(summary.model_usage_summaries) > 0
    assert "qwen2.5:7b" in summary.model_usage_summaries
    model_summary = summary.model_usage_summaries["qwen2.5:7b"]
    assert model_summary.total_calls == 2
    print(f"Total calls: {model_summary.total_calls}")
    print(f"Total input tokens: {model_summary.total_input_tokens}")
    print(f"Total output tokens: {model_summary.total_output_tokens}")
    print("✓ Usage tracking test passed")


def test_missing_model_error():
    """Test error when model name is missing."""
    print("\n=== Test: Missing Model Error ===")
    client = OllamaClient()  # No model name provided

    try:
        client.completion("Hello")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"Expected error: {e}")
        assert "Model name is required" in str(e)
        print("✓ Missing model error test passed")


def test_invalid_prompt_type_error():
    """Test error with invalid prompt type."""
    print("\n=== Test: Invalid Prompt Type Error ===")
    client = OllamaClient(model_name="qwen2.5:7b")

    try:
        client.completion(123)  # Invalid type
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"Expected error: {e}")
        assert "Invalid prompt type" in str(e)
        print("✓ Invalid prompt type error test passed")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Ollama Client Tests")
    print("=" * 60)

    try:
        test_basic_completion()
        test_message_list_completion()
        test_model_override()
        await test_async_completion()
        test_multi_turn_conversation()
        test_usage_tracking()
        test_missing_model_error()
        test_invalid_prompt_type_error()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
