import os
import sys
import random
import textwrap
from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from context_generation.context_generator import generate_massive_context, insert_location_needle


def main():
    """
    # Create large context
    print("Example of using RLM (REPL) on a needle-in-haystack problem.")
    context = generate_massive_context(num_lines=10000, number_of_needles=2)
    context = insert_location_needle(context)
    
    # Save context file for debugging and inspection
    with open("../openai/input/massive_context.txt", "w") as f:
        f.write(context)

    # Read context file
    with open("../openai/input/massive_context.txt") as f:
        context = f.read()
    """

    # Read context file (pdf-to-text output)
    with open("../inputs/Speech and Language Processing - Daniel Jurafsky.txt") as f:
        context = f.read()

    CUSTOM_SYSTEM_PROMPT = textwrap.dedent(
    """You are currently executing an RLM instance inside a multi-layered RLM workflow. You are not the orchestrator, and you cannot control or inspect other instances directly. A higher-level orchestrator may run many RLM instances across layers, either sequentially or in parallel, where each instance contributes partial progress toward a shared global objective.
    
    You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, and you are strongly encouraged to use them as much as possible. You will be queried iteratively until you provide a final answer. One RLM instance may make many recursive sub-LM calls (via `llm_query` and `llm_query_batched`) across multiple iterations before returning a final answer.

    The REPL environment is initialized with:
    1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
    2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
    3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
    4. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

    You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. You are encouraged to ask for structured outputs from your sub-LLM calls (e.g. JSON with particular fields) to help you track information and maintain state in your REPL environment through variables. You can use these variables as buffers to build up your final answer. An example of a structured output format is:
    ```json
    {
        "answer": "The answer to the question",
        "reasoning": "The reasoning behind the answer"
    }
    ```

    Remember, your sub-LLM calls might not always find the answer, and that's okay! You can use them to gather information, analyze the context, and build up to your final answer iteratively.

    Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

    You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful, they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

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

    COMPLETION SIGNALS (IMPORTANT):
    Local completion (FINAL|FINAL_VAR):
    When you are done with the iterative process, you MUST provide a final answer inside a `FINAL(...)` or `FINAL_VAR(...)` function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
    1. Use FINAL(your final answer here) to provide the answer directly
    2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

    Global workflow recommendation (STOP_WORKFLOW):
    When you are returning `FINAL(...)` or `FINAL_VAR(...)`, you must also include exactly one `STOP_WORKFLOW(true)` or `STOP_WORKFLOW(false)` call in that same final response. Use lowercase boolean literals only (`true`/`false`), not `0/1`. Return this as plain text, NOT in code.
    - `FINAL(...)` and `FINAL_VAR(...)` indicate this RLM instance is returning a final answer.
    - `STOP_WORKFLOW(true|false)` indicates whether this instance recommends terminating the overall multi-instance workflow.
    - Set `STOP_WORKFLOW(true)` only when you are also returning `FINAL(...)` or `FINAL_VAR(...)`, and you are confident the overall user objective is globally satisfied.
    - If you are not returning `FINAL(...)` or `FINAL_VAR(...)`, do not output any `STOP_WORKFLOW(...)` call yet.
    - Even when returning a final answer, `STOP_WORKFLOW(false)` is valid if additional RLM instances may still be needed.

    Example: if the task is to find two magic numbers and this instance finds only one, it can return a final local result but should output `STOP_WORKFLOW(false)` because the global objective is not yet complete. If the task is to find one magic number and this instance confidently finds it, it can return a final answer with `STOP_WORKFLOW(true)`. This is a simple illustration, and examples are not limited to needle-in-a-haystack problems.

    IMPORTANT: Once you decide to return a final answer, do not include any more code blocks or LLM calls because that can cause confusion about what your final answer is. If you want to do any additional reasoning or analysis before answering, you are encouraged to use additional iterations for code execution.
    
    Valid combinations:
    - Final answer provided + `STOP_WORKFLOW(true)`
    - Final answer provided + `STOP_WORKFLOW(false)`
    - No final answer provided + no `STOP_WORKFLOW(...)`
    Invalid combination:
    - No final answer provided + `STOP_WORKFLOW(true)`
    - No final answer provided + `STOP_WORKFLOW(false)`

    Think step by step carefully, plan, and execute this plan immediately in your response, do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
    """
    )

    #ROOT_PROMPT = "I'm looking for a number in the context."

    #ROOT_PROMPT = "I'm looking for a number. I know that there is ONLY one number in the entire context. Return the number if you can find it in your chunk. If you can't find any, return 'I couldn't find any magic numbers.'"

    #ROOT_PROMPT = "I'm looking for two numbers in the context."

    #ROOT_PROMPT = "Where is Stevens Institute of Technology located?"

    #ROOT_PROMPT = "Based on the given context, answer the following question: Where is Stevens Institute of Technology located?"

    #ROOT_PROMPT = "Based on the given context and the conversation, answer the following question: Where do they think Stevens Institute of Technology is located?"

    #ROOT_PROMPT = "Based on the given context, answer the following question: What are the two complementary pillars of Unity Catalog Business Semantics as described in the document?"

    #ROOT_PROMPT = "Based on the given context, answer the following question. Even tough this is a short document, split it into five chunks and analyze each chunk iteratively in the REPL environment with `llm_query` to make sure you don't miss any details. Question: What are the two complementary pillars of Unity Catalog Business Semantics as described in the document?"

    ROOT_PROMPT = "Based on the given context, answer the following question: What is the main idea behind the logit lens?"

    # Initialize RLM with custom system prompt
    load_dotenv()
    logger = RLMLogger(log_dir="./logs")
    rlm = RLM(
        backend="openai",  # or "portkey", etc.
        backend_kwargs={
            #"model_name": "gpt-5-nano-2025-08-07",
            "model_name": "gpt-5-mini-2025-08-07",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        environment="local",
        environment_kwargs={},
        max_depth=1,
        max_iterations=25,
        custom_system_prompt=CUSTOM_SYSTEM_PROMPT,
        logger=logger,
        verbose=True,  # For printing to console with rich, disabled by default.
    )

    result = rlm.completion(
        prompt=context,  # The file content (large context)
        root_prompt=ROOT_PROMPT  # User prompt
    )

    print(f"Result: {result.response}")
    #print(f"Expected answer: {answer}")
    print(f"Stop workflow signal: {result.stop_workflow}")

if __name__ == "__main__":
    main()
