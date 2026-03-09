import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from rlm import RLM
from .tokenizer import count_words, tokenize


def _run_rlm_completion_worker(rlm_kwargs, chunk, root_prompt):
    """
    Worker function for ProcessPoolExecutor to run a single RLM completion.
    Returns structured data including stop_workflow signal. Must be at module level to be picklable.
    """
    try:
        rlm = RLM(**rlm_kwargs)
        result = rlm.completion(prompt=chunk, root_prompt=root_prompt)
        return {
            "response": result.response,
            "stop_workflow": result.stop_workflow if result.stop_workflow is not None else False,
            "error": None,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "response": f"Error: {str(e)}",
            "stop_workflow": False,
            "error": str(e),
        }


class DeepRLM:
    def __init__(
        self,
        num_rlms_in_depth=2,
        max_system_depth=10,
        token_limit=1000,
        max_parallel_workers=None,
        **rlm_kwargs,
    ):
        """
        Initialize DeepRLM with RLM parameters.

        Args:
            num_rlms_in_depth: Maximum number of RLMs in depth for recursive reasoning (if applicable).
            max_system_depth: Maximum system depth for recursive reasoning (if applicable).
            token_limit: Optional token limit for a single RLM.
            max_parallel_workers: Maximum number of parallel workers for concurrent execution. None means unlimited.
            **rlm_kwargs: All parameters are forwarded to RLM initialization.
                See RLM.__init__() for available parameters:
                - backend, backend_kwargs
                - environment, environment_kwargs
                - depth, max_depth, max_iterations
                - custom_system_prompt
                - other_backends, other_backend_kwargs
                - logger, verbose, persistent
        """
        self.num_rlms_in_depth = num_rlms_in_depth
        self.max_system_depth = max_system_depth
        self.token_limit = token_limit
        self.max_parallel_workers = max_parallel_workers
        self.rlm_kwargs = rlm_kwargs
        print("\n=============== DEEP RLM ================")
        print(
            f"Initialized DeepRLM with\n\tnum_rlms_in_depth={num_rlms_in_depth},\n\tmax_system_depth={max_system_depth},\n\ttoken_limit={token_limit}\n\tmax_parallel_workers={max_parallel_workers or 'unlimited'}\n"
        )

    def simple_decomposition(self, context, token_multiplier=1.5):
        """
        A simple context decomposition method that splits the context into chunks based on a token limit. This
        is a heuristic approach and can be improved with more sophisticated methods.

        Args:
            context: The input context to be decomposed.
            token_multiplier: A multiplier to allow some flexibility in determining if the context is atomic
                or needs decomposition. For example, if token_limit is 1000 and token_multiplier is 1.5, then
                contexts with up to 1500 tokens will not be decomposed.

        Returns:
            A list of context chunks that are within the token limit.
        """
        words, num_words = count_words(context)
        if num_words == 0:
            raise ValueError("Context must not be empty.")

        num_tokens = tokenize(
            context, model=self.rlm_kwargs.get("backend_kwargs", {}).get("model_name", "gpt-5-nano")
        )
        tokens_per_word = num_tokens / num_words
        print(
            f"Read context with word count: {num_words}, token count: {num_tokens}, tokens per word: {tokens_per_word:.2f}"
        )

        decomposed_context = []
        # Check if task is atomic. If so, no need to decompose. `token_multiplier` is a simple way to allow
        # some flexibility to avoid splitting when the input has 105 tokens and token_limit is 100, for example.
        if self.token_limit is not None and num_tokens <= self.token_limit * token_multiplier:
            decomposed_context.append(context)
        else:
            words_per_chunk = max(1, int(self.token_limit / tokens_per_word))
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i : i + words_per_chunk]
                chunk = " ".join(chunk_words)
                decomposed_context.append(chunk)

        return decomposed_context

    def run(self, context, prompt=None):
        """
        Run DeepRLM with the given context and prompt.

        Processes decomposed context chunks across multiple depth layers,
        with each layer running up to num_rlms_in_depth RLM instances concurrently.

        Args:
            context: The input context to be decomposed and processed.
            prompt: Optional root prompt to be passed to each RLM.completion call.

        Returns:
            Dictionary mapping RLM identifiers ("RLM-0", "RLM-1", etc.) to their response strings.
        """
        if not context:
            raise ValueError("Context must be provided.")

        decomposed_context = self.simple_decomposition(context)
        num_chunks = len(decomposed_context)

        print(f"Number of decomposed chunks: {num_chunks}")
        for i, item in enumerate(decomposed_context):
            print(f"Chunk {i}: {item[:100]}..." if len(item) > 100 else f"Chunk {i}: {item}")

        # Calculate number of layers needed
        num_layers = math.ceil(num_chunks / self.num_rlms_in_depth)
        print(f"Number of layers required: {num_layers}")

        # Validate against max_system_depth
        if num_layers > self.max_system_depth:
            raise ValueError(
                f"Number of required layers ({num_layers}) exceeds max_system_depth ({self.max_system_depth}). "
                f"Reduce context size or increase max_system_depth."
            )

        results = {}
        global_rlm_index = 0

        # Process chunks layer by layer
        for layer_idx in range(num_layers):
            start_idx = layer_idx * self.num_rlms_in_depth
            end_idx = min(start_idx + self.num_rlms_in_depth, num_chunks)
            layer_chunks = decomposed_context[start_idx:end_idx]

            print(f"\n=== Processing Layer {layer_idx + 1}/{num_layers} ===")
            print(f"Chunks {start_idx} to {end_idx - 1} ({len(layer_chunks)} RLMs)")

            # Execute all RLMs in this layer concurrently using separate processes
            with ProcessPoolExecutor(max_workers=len(layer_chunks)) as executor:
                # Submit tasks to process pool
                futures = [
                    executor.submit(
                        _run_rlm_completion_worker, self.rlm_kwargs, layer_chunks[i], prompt
                    )
                    for i in range(len(layer_chunks))
                ]

                # Collect results as they complete
                for i, future in enumerate(futures):
                    result = future.result()
                    results[f"RLM-{global_rlm_index + i}"] = result["response"]

            global_rlm_index += len(layer_chunks)
            print(f"Layer {layer_idx + 1} completed.")

        print(f"\n=== All {num_layers} layers completed ===")
        print(f"Total RLMs executed: {len(results)}")

        return results

    def run_binary(self, context, prompt=None, stop_mode="immediate"):
        """
        Run DeepRLM with middle-first chunk binary search based on stop_workflow signals.

        First decomposes context using simple_decomposition, then starts from the middle chunk.
        If STOP_WORKFLOW(true) is not found, branches left and right by index ranges with midpoint
        evaluation. Continues branching until a node emits STOP_WORKFLOW(true) or max_system_depth
        is reached.

        Uses BFS (breadth-first search) traversal: all nodes at depth d are processed
        in parallel before moving to depth d+1.

        Args:
            context: The input context to be processed.
            prompt: Optional root prompt to be passed to each RLM.completion call.
            stop_mode: How to handle STOP_WORKFLOW(true) signals. Options:
                - "immediate": Stop as soon as first stop signal is seen (best-effort cancellation)
                - "finish_depth": Complete all nodes at stop depth before returning

        Returns:
            Dictionary with one key:
            - "nodes": List of node records with {"id", "depth", "response", "stop_workflow",
                       "error", "parent_id", "lo", "hi", "mid_index"}
        """
        if not context:
            raise ValueError("Context must be provided.")

        if stop_mode not in ("immediate", "finish_depth"):
            raise ValueError(f"stop_mode must be 'immediate' or 'finish_depth', got '{stop_mode}'")

        print("\n======= BINARY SEARCH RUN =======")
        print(f"Max system depth: {self.max_system_depth}")
        print(f"Max parallel workers: {self.max_parallel_workers or 'unlimited'}")
        print(f"Stop mode: {stop_mode}")

        # Decompose context into chunks first
        chunks = self.simple_decomposition(context)
        num_chunks = len(chunks)
        print(f"\nDecomposed into {num_chunks} chunks")
        for i, chunk in enumerate(chunks):
            preview = (
                chunk[:80].replace("\n", " ") + "..."
                if len(chunk) > 80
                else chunk.replace("\n", " ")
            )
            print(f"  Chunk {i}: {preview}")

        if num_chunks == 0:
            raise ValueError("simple_decomposition returned empty chunk list")

        # Node structure: {"id", "depth", "lo", "hi", "mid_index", "chunk", "parent_id"}
        nodes = []
        node_id_counter = 0

        # Initialize frontier with middle chunk spanning full range
        mid_index = num_chunks // 2
        frontier = [
            {
                "id": node_id_counter,
                "depth": 0,
                "lo": 0,
                "hi": num_chunks - 1,
                "mid_index": mid_index,
                "chunk": chunks[mid_index],
                "parent_id": None,
            }
        ]
        node_id_counter += 1
        print(f"\nStarting from middle chunk: index {mid_index} (range [0, {num_chunks - 1}])")

        global_stop = False

        while frontier and not global_stop:
            current_depth = frontier[0]["depth"]

            if current_depth > self.max_system_depth:
                print(f"\n=== Reached max_system_depth ({self.max_system_depth}) ===")
                break

            print(f"\n=== Processing Depth {current_depth} ===")
            print(f"Frontier size: {len(frontier)} nodes")

            # Determine worker limit for this batch
            max_workers = len(frontier)
            if self.max_parallel_workers is not None:
                max_workers = min(max_workers, self.max_parallel_workers)

            # Execute nodes using as_completed for immediate cancellation on stop
            executor = ProcessPoolExecutor(max_workers=max_workers)
            future_to_node = {
                executor.submit(
                    _run_rlm_completion_worker, self.rlm_kwargs, node["chunk"], prompt
                ): node
                for node in frontier
            }

            results_by_node_id = {}

            try:
                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    result = future.result()
                    results_by_node_id[node["id"]] = result

                    # Check for immediate stop
                    if result["stop_workflow"] is True and stop_mode == "immediate":
                        print(
                            f"  --> Node {node['id']} emitted STOP_WORKFLOW(true), cancelling remaining tasks"
                        )
                        # Cancel all pending futures
                        for pending_future in future_to_node:
                            if pending_future is not future and not pending_future.done():
                                cancelled = pending_future.cancel()
                                if cancelled:
                                    pending_node = future_to_node[pending_future]
                                    print(
                                        f"  --> Cancelled pending node {pending_node['id']} [chunk {pending_node['mid_index']}]"
                                    )
                        break
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

            # Build results_batch in original frontier order (for deterministic node IDs)
            results_batch = [results_by_node_id.get(node["id"]) for node in frontier]

            # Process results and build next frontier
            next_frontier = []
            stop_signal_seen = False

            for _i, (node, result) in enumerate(zip(frontier, results_batch, strict=True)):
                # Skip cancelled nodes (result is None)
                if result is None:
                    print(f"  Node {node['id']} [chunk {node['mid_index']}]: cancelled")
                    continue

                node_record = {
                    "id": node["id"],
                    "depth": node["depth"],
                    "response": result["response"],
                    "stop_workflow": result["stop_workflow"],
                    "error": result["error"],
                    "parent_id": node["parent_id"],
                    "lo": node["lo"],
                    "hi": node["hi"],
                    "mid_index": node["mid_index"],
                }
                nodes.append(node_record)

                print(
                    f"  Node {node['id']} [chunk {node['mid_index']}]: stop_workflow={result['stop_workflow']}, error={result['error'] is not None}"
                )

                # Check for stop condition
                if result["stop_workflow"] is True:
                    print(f"  --> Node {node['id']} emitted STOP_WORKFLOW(true)")
                    stop_signal_seen = True
                    if stop_mode == "immediate":
                        global_stop = True
                        break

                # If not stopped and not at max depth, branch based on parent's range
                if node["depth"] < self.max_system_depth and result["error"] is None:
                    mid = node["mid_index"]

                    # Determine left and right ranges within parent's bounds
                    left_lo = node["lo"]
                    left_hi = mid - 1
                    right_lo = mid + 1
                    right_hi = node["hi"]

                    # Create child nodes if ranges are valid
                    if left_lo <= left_hi:
                        left_mid = (left_lo + left_hi) // 2
                        left_node = {
                            "id": node_id_counter,
                            "depth": node["depth"] + 1,
                            "lo": left_lo,
                            "hi": left_hi,
                            "mid_index": left_mid,
                            "chunk": chunks[left_mid],
                            "parent_id": node["id"],
                        }
                        node_id_counter += 1
                        next_frontier.append(left_node)
                        print(
                            f"  --> Node {node['id']} branching left: node {left_node['id']} [chunks {left_lo}-{left_hi}, mid={left_mid}]"
                        )

                    if right_lo <= right_hi:
                        right_mid = (right_lo + right_hi) // 2
                        right_node = {
                            "id": node_id_counter,
                            "depth": node["depth"] + 1,
                            "lo": right_lo,
                            "hi": right_hi,
                            "mid_index": right_mid,
                            "chunk": chunks[right_mid],
                            "parent_id": node["id"],
                        }
                        node_id_counter += 1
                        next_frontier.append(right_node)
                        print(
                            f"  --> Node {node['id']} branching right: node {right_node['id']} [chunks {right_lo}-{right_hi}, mid={right_mid}]"
                        )

                    if left_lo > left_hi and right_lo > right_hi:
                        print(f"  --> Node {node['id']} has no valid branches remaining")

            print(f"Depth {current_depth} completed.")

            # Handle finish_depth mode
            if stop_mode == "finish_depth" and stop_signal_seen:
                print(
                    f"Stop signal seen at depth {current_depth}, stopping after completing this depth (finish_depth mode)"
                )
                global_stop = True

            frontier = next_frontier

        print("\n=== Binary search completed ===")
        print(f"Total nodes executed: {len(nodes)}")
        print(f"Max depth reached: {max(n['depth'] for n in nodes) if nodes else 0}")
        print(f"Global stop triggered: {global_stop}")

        return {"nodes": nodes}
