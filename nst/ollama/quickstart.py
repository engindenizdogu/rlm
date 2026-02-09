import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="ollama",  # or "portkey", etc.
    #backend_kwargs={"model_name": "qwen2.5:7b"},
    backend_kwargs={"model_name": "qwen3:8b"},
    environment="local",
    environment_kwargs={},
    max_depth=1,
    max_iterations=15,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

result = rlm.completion("Print me the first 100 powers of two, each on a newline.")

print(result)