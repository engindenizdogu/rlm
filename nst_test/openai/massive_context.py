import os
import random
import textwrap
from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger

def generate_massive_context(num_lines: int = 1_000_000, answer: str = "1298418") -> str:
    print(f"Generating massive context with {num_lines} lines...")
    
    # Set of random words to use
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    
    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))
    
    # Insert the magic number at a random position (somewhere near the end)
    magic_position = random.randint(int(num_lines * 0.5), int(num_lines * 0.75))
    lines[magic_position] = f"The magic number is {answer}"
    
    print(f"Magic number inserted at position {magic_position}")
    
    return "\n".join(lines)

def main():
    load_dotenv()
    logger = RLMLogger(log_dir="./logs")

    # Create large context
    print("Example of using RLM (REPL) on a needle-in-haystack problem.")
    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=10, answer=answer)

    # Save context file for debugging and inspection
    with open("massive_context.txt", "w") as f:
        f.write(context)

    CUSTOM_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

    You are one instance inside a multi-layered RLM workflow. A higher-level orchestrator may run many RLM instances across layers, either sequentially or in parallel, where each instance contributes partial progress toward a shared global objective. `FINAL(...)` and `FINAL_VAR(...)` indicate local completion for this specific RLM instance, while `STOP_WORKFLOW(true|false)` is a separate global recommendation about whether the overall multi-instance workflow should terminate. `STOP_WORKFLOW(true)` is only valid when a final answer is returned in the same response and the global objective is likely satisfied.

    The REPL environment is initialized with:
    1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
    2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
    3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
    4. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

    In addition to your normal reasoning output, when you are done with the iterative process and returning a final answer, you must include exactly one function call of the form `STOP_WORKFLOW(true)` or `STOP_WORKFLOW(false)` in that same final response. This signal is separate from `FINAL(...)` and `FINAL_VAR(...)`. Use lowercase boolean literals only (`true`/`false`), not `0/1`. Return this as plain text, NOT in code.
    - `FINAL(...)` and `FINAL_VAR(...)` indicate this RLM instance is returning a final answer.
    - `STOP_WORKFLOW(true|false)` indicates whether this instance recommends terminating the overall multi-instance workflow.
    - Set `STOP_WORKFLOW(true)` only when you are also returning `FINAL(...)` or `FINAL_VAR(...)`, and you are confident the overall user objective is globally satisfied.
    - If you are not returning `FINAL(...)` or `FINAL_VAR(...)`, do not output any `STOP_WORKFLOW(...)` call yet.
    - Even when returning a final answer, `STOP_WORKFLOW(false)` is valid if additional RLM instances may still be needed.

    Example: if the task is to find two magic numbers and this instance finds only one, it can return a final local result but should output `STOP_WORKFLOW(false)` because the global objective is not yet complete. If the task is to find one magic number and this instance confidently finds it, it can return a final answer with `STOP_WORKFLOW(true)`. This is a simple illustration, and examples are not limited to needle-in-a-haystack problems.

    You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
    Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

    You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

    When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
    ```repl
    chunk = context[:10000]
    answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
    print(answer)
    ```

    As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
    ```repl
    query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
    for i, section in enumerate(context):
        if i == len(context) - 1:
            buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
            print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
        else:
            buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
            print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
    ```

    As another example, when the context is very long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
    ```repl
    query = 'A man became famous for his book "The Great Gatsby". How many jobs did he have?'
    # Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
    chunk_size = len(context) // 10
    chunks = []
    for i in range(10):
        if i < 9:
            chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
        else:
            chunk_str = "\n".join(context[i*chunk_size:])
        chunks.append(chunk_str)

    # Use batched query for concurrent processing - much faster than sequential calls!
    prompts = [f"Try to answer the following query: {{query}}. Here are the documents:\n{{chunk}}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
    answers = llm_query_batched(prompts)
    for i, answer in enumerate(answers):
        print(f"I got the answer from chunk {{i}}: {{answer}}")
    final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
    ```

    As a final example, after analyzing the context and realizing it's separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
    ```repl
    # After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
    import re
    sections = re.split(r'### (.+)', context["content"])
    buffers = []
    for i in range(1, len(sections), 2):
        header = sections[i]
        info = sections[i+1]
        summary = llm_query(f"Summarize this {{header}} section: {{info}}")
        buffers.append(f"{{header}}: {{summary}}")
    final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
    ```
    In the next step, we can return FINAL_VAR(final_answer).

    IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a `FINAL(...)` or `FINAL_VAR(...)` function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
    1. Use FINAL(your final answer here) to provide the answer directly
    2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

    In that same final response, you must also include exactly one `STOP_WORKFLOW(true)` or `STOP_WORKFLOW(false)` call, as plain text and NOT in code.
    Valid combinations:
    - Final answer provided + `STOP_WORKFLOW(true)`
    - Final answer provided + `STOP_WORKFLOW(false)`
    - No final answer provided + no `STOP_WORKFLOW(...)`
    Invalid combination:
    - No final answer provided + `STOP_WORKFLOW(true)`
    - No final answer provided + `STOP_WORKFLOW(false)`

    Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
    """
    )

    ROOT_PROMT = "I'm looking for 2 magic numbers. Return the magic numbers if you can find them in the context. If you can only find one, return that one. If you can't find any, return 'I couldn't find any magic numbers.'"
    
    # Initialize RLM with custom system prompt
    rlm = RLM(
        backend="openai",  # or "portkey", etc.
        backend_kwargs={
            "model_name": "gpt-5-nano-2025-08-07", 
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        environment="local",
        environment_kwargs={},
        max_depth=1,
        max_iterations=10,
        custom_system_prompt=CUSTOM_SYSTEM_PROMPT,
        #logger=logger,
        verbose=True,  # For printing to console with rich, disabled by default.
    )

    result = rlm.completion(
        prompt=context,  # The file content (large context)
        root_prompt=ROOT_PROMT  # User prompt
    )

    print(f"Result: {result.response}.\nExpected: {answer}")
    print(f"Stop workflow signal: {result.stop_workflow}")

if __name__ == "__main__":
    main()