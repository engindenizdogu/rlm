import os
import time

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="openai",  # or "portkey", etc.
    backend_kwargs={
        "model_name": "gpt-5-nano-2025-08-07",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    environment="local",
    environment_kwargs={},
    max_depth=0,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

start = time.time()
result = rlm.completion("Print me the first 100 powers of two, each on a newline.")
end = time.time()
print(f"Time taken: {end - start} seconds")
print(result)
