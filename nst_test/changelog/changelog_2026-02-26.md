# Changelog — 2026-02-26

## Summary
Added STOP_WORKFLOW support for multi-layer RLM runs.

## Updated files
- `nst_test/docs/custom_system_prompt.txt`
  - Added STOP_WORKFLOW output rules and valid/invalid combinations.
- `nst_test/openai/custom_system_prompt.py`
  - Added the same STOP_WORKFLOW rules for the OpenAI test prompt.
- `rlm/utils/parsing.py`
  - Added `find_stop_workflow(text) -> bool | None`.
  - Returns `None` when tag is missing, malformed, or duplicated.
- `rlm/core/rlm.py`
  - Parses STOP_WORKFLOW directly with `find_stop_workflow(...)`.
  - Passes parsed value into `RLMChatCompletion.stop_workflow`.
  - Stores parsed `stop_workflow` on each `RLMIteration` before logging.
- `rlm/core/types.py`
  - Added `stop_workflow` to `RLMChatCompletion` serialization/deserialization.
  - Fixed dataclass field ordering.
  - Added `stop_workflow` to `RLMIteration` so it is included in iteration logs.
- `tests/test_parsing.py`
  - Added tests for valid/invalid STOP_WORKFLOW patterns.
- `tests/test_types.py`
  - Added coverage for `RLMIteration.stop_workflow` defaults and serialization.

## Validation
- `pytest tests/test_parsing.py` → 33 passed.
- `pytest tests/test_types.py tests/test_parsing.py` → 50 passed.
