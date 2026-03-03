# Changelog — 2026-03-02

## Summary
Added binary-search-style run method to DeepRLM with STOP_WORKFLOW support.

## Updated files
- `deep_rlm/deep_rlm.py`
  - Added `_run_rlm_completion_worker_structured` returning structured data with stop_workflow
  - Added `binary_split` method for exact 2-chunk splits
  - Added `run_binary` method with BFS traversal and early stop on STOP_WORKFLOW(true)
  - Added `max_parallel_workers` parameter to control concurrent execution
- `tests/test_deep_rlm_binary.py`
  - Added 15 tests covering binary split, stop behavior, depth limits, parallelism, and output structure
- `examples/deep_rlm_binary_example.py`
  - Added example demonstrating binary search usage

## Behavior
- Starts with single RLM on full context, branches into left/right halves if answer not found
- Uses BFS level-by-level expansion with parallel execution per depth
- Stops globally when any node emits STOP_WORKFLOW(true)
- Limits depth via max_system_depth (independent of num_rlms_in_depth)

## TODO
- Test `run_binary`
- Tree visualization