# Changelog — 2026-03-04

## Summary
- Added `nst_test/pdf-search/pdf_to_text.py` for pdf to txt convertion.
- RLMs could return a final answer before executing all code blocks within the environment. Updated the if statement to prevent that in `rlm/core/rlm.py`:
  # There should be no code blocks left to run
  if final_answer is not None and not iteration.code_blocks:

CUSTOM_SYSTEM_PROMPT updated, added:
  IMPORTANT: Once you decide to return a final answer, do not include any more code blocks or LLM calls because that can cause confusion about what your final answer is. If you want to do any additional reasoning or analysis before answering, you are encouraged to use additional iterations for code execution.

  AND

  You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. You are encouraged to ask for structured outputs from your sub-LLM calls (e.g. JSON with particular fields) to help you track information and maintain state in your REPL environment through variables. You can use these variables as buffers to build up your final answer. An example of a structured output format is:
  ```json
  {
        "answer": "The answer to the question",
        "reasoning": "The reasoning behind the answer"
  }
  ```

  AND replaced

  "When you are returnin a final answer" with "When you are returning `FINAL(...)` or `FINAL_VAR(...)`," (to prevent confusion)

  AND

  "Remember, your sub-LLM calls might not always find the answer, and that's okay! You can use them to gather information, analyze the context, and build up to your final answer iteratively."
