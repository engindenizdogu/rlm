import os
import time
import random
from dotenv import load_dotenv
from rlm.logger import RLMLogger
from deep_rlm.context_generator import generate_massive_context
from deep_rlm.deep_rlm import DeepRLM

def main():
    load_dotenv()
    logger = RLMLogger(log_dir="./logs")

    # Generate 'massive' context
    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=100, answer=answer)

    # Save context file for debugging and inspection
    with open("massive_context.txt", "w") as f:
        f.write(context)

    # Initialize DeepRLM
    deepRLM = DeepRLM(
        num_rlms_in_depth = 2,  # Maximum number of RLMs in depth for recursive reasoning
        max_system_depth = 10,  # Maximum system depth for recursive reasoning
        token_limit = 200,      # Token limit for a single RLM.
        backend="openai",
        backend_kwargs={
            "model_name": "gpt-5-nano-2025-08-07", 
            "api_key": os.getenv("OPENAI_API_KEY"),
            },
        environment="local",
        environment_kwargs={},
        max_depth=1,
        max_iterations=3,
        logger=logger,
        verbose=False,
    )

    start = time.time()
    results = deepRLM.run(
        context=context,    # The file content (large context)
        prompt="I'm looking for a magic number. I'm not sure if it's in this chunk, but tell me if you can find it."    # User prompt
    )
    end = time.time()

    print(f"\nExecution time: {end - start:.2f} seconds")
    print("=========== RESULTS ===========")
    print(results)

if __name__ == "__main__":
    main()