import math
from concurrent.futures import ProcessPoolExecutor
from rlm import RLM
from .tokenizer import tokenize, count_words

def _run_rlm_completion_worker(rlm_kwargs, chunk, root_prompt):
    """
    Worker function for ProcessPoolExecutor to run a single RLM completion.
    Must be at module level to be picklable.
    """
    try:
        rlm = RLM(**rlm_kwargs)
        result = rlm.completion(prompt=chunk, root_prompt=root_prompt)
        return result.response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

class DeepRLM:
    def __init__(self, num_rlms_in_depth = 2, max_system_depth = 10, token_limit = 1000, **rlm_kwargs):
        """
        Initialize DeepRLM with RLM parameters.
        
        Args:
            num_rlms_in_depth: Maximum number of RLMs in depth for recursive reasoning (if applicable).
            max_system_depth: Maximum system depth for recursive reasoning (if applicable).
            token_limit: Optional token limit for a single RLM.
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
        self.rlm_kwargs = rlm_kwargs
        print("\n=============== DEEP RLM ================")
        print(f"Initialized DeepRLM with\n\tnum_rlms_in_depth={num_rlms_in_depth},\n\tmax_system_depth={max_system_depth},\n\ttoken_limit={token_limit}\n")

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

        num_tokens = tokenize(context, model=self.rlm_kwargs.get("backend_kwargs", {}).get("model_name", "gpt-5-nano"))
        tokens_per_word = num_tokens / num_words
        print(f"Read context with word count: {num_words}, token count: {num_tokens}, tokens per word: {tokens_per_word:.2f}")

        decomposed_context = []
        # Check if task is atomic. If so, no need to decompose. `token_multiplier` is a simple way to allow
        # some flexibility to avoid splitting when the input has 105 tokens and token_limit is 100, for example.
        if self.token_limit is not None and num_tokens <= self.token_limit * token_multiplier:
            decomposed_context.append(context)
        else:
            words_per_chunk = max(1, int(self.token_limit / tokens_per_word))
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i+words_per_chunk]
                chunk = ' '.join(chunk_words)
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
                        _run_rlm_completion_worker,
                        self.rlm_kwargs,
                        layer_chunks[i],
                        prompt
                    )
                    for i in range(len(layer_chunks))
                ]
                
                # Collect results as they complete
                for i, future in enumerate(futures):
                    response = future.result()
                    results[f"RLM-{global_rlm_index + i}"] = response
            
            global_rlm_index += len(layer_chunks)
            print(f"Layer {layer_idx + 1} completed.")
        
        print(f"\n=== All {num_layers} layers completed ===")
        print(f"Total RLMs executed: {len(results)}")
        
        return results