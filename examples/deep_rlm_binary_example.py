"""
Example demonstrating DeepRLM's middle-first binary search run method.

This example shows how to use run_binary() which:
- First decomposes context into chunks using simple_decomposition
- Starts from the middle chunk (efficient for large contexts)
- Branches left and right by index ranges if STOP_WORKFLOW(true) not found
- Processes nodes level-by-level (BFS) until stop signal or max depth
- Returns structured node data with chunk indices
- Supports configurable stop modes: "immediate" or "finish_depth"
"""

import os

from dotenv import load_dotenv

from deep_rlm.deep_rlm import DeepRLM


def main():
    load_dotenv()

    # Sample context with multiple sections
    context = """
    Section 1: Introduction
    This document contains important information about project requirements.
    
    Section 2: Technical Details
    The system should handle concurrent requests efficiently.
    
    Section 3: Security Requirements
    Authentication must use OAuth2 with JWT tokens.
    
    Section 4: Performance Goals
    The API should respond within 200ms for 95% of requests.
    
    Section 5: Conclusion
    The magic number is 7531. This is the answer you're looking for.
    STOP_WORKFLOW(true)
    """

    # Initialize DeepRLM with binary search capability
    deep_rlm = DeepRLM(
        max_system_depth=3,  # Maximum tree depth (0 = root only)
        max_parallel_workers=4,  # Limit concurrent RLM instances
        token_limit=100,  # Token limit per chunk (for binary splits)
        backend="openai",
        backend_kwargs={
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        environment="local",
        max_iterations=2,  # RLM iterations per node
        verbose=False,
    )

    print("\n=== Example: Middle-First Binary Search Run ===\n")

    # Run binary search with default "immediate" stop mode
    # The method will:
    # 1. Decompose context into chunks based on token_limit
    # 2. Start from middle chunk index
    # 3. If stop signal not found, branch to left and right ranges
    # 4. Evaluate midpoint of each range
    # 5. Stop immediately when first stop signal found (immediate mode)
    result = deep_rlm.run_binary(
        context=context,
        prompt="Find the magic number in this document. When you find it, respond with STOP_WORKFLOW(true).",
        stop_mode="immediate",  # Can also use "finish_depth" to complete current depth
    )

    print("\n=== Results ===\n")

    # Structured node data
    print(f"Total nodes executed: {len(result['nodes'])}")
    print(f"Max depth reached: {max(n['depth'] for n in result['nodes'])}")

    # Check if any node found the answer
    stopped_nodes = [n for n in result["nodes"] if n["stop_workflow"]]
    if stopped_nodes:
        print(
            f"\n✓ Stop signal found at node {stopped_nodes[0]['id']} (depth {stopped_nodes[0]['depth']})"
        )
        print(f"Response: {stopped_nodes[0]['response'][:200]}...")
    else:
        print("\n✗ No stop signal found")

    # Display tree structure with chunk indices
    print("\n=== Node Tree ===")
    for node in result["nodes"]:
        indent = "  " * node["depth"]
        status = "✓ STOPPED" if node["stop_workflow"] else "→ searching"
        parent = f"(parent: {node['parent_id']})" if node["parent_id"] is not None else "(root)"
        chunk_info = f"[chunks {node['lo']}-{node['hi']}, mid={node['mid_index']}]"
        print(f"{indent}Node {node['id']} {parent} {chunk_info}: {status}")


if __name__ == "__main__":
    main()
