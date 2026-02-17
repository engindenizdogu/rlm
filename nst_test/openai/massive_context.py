import os
import random
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

    print("Example of using RLM (REPL) on a needle-in-haystack problem.")
    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=100, answer=answer)

    # Save context file for debugging and inspection
    with open("massive_context.txt", "w") as f:
        f.write(context)

    rlm = RLM(
        backend="openai",  # or "portkey", etc.
        backend_kwargs={
            "model_name": "gpt-5-nano-2025-08-07", 
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        environment="local",
        environment_kwargs={},
        max_depth=1,
        max_iterations=5,
        logger=logger,
        verbose=True,  # For printing to console with rich, disabled by default.
    )

    result = rlm.completion(
        prompt=context,  # The file content (large context)
        root_prompt="I'm looking for a magic number. I'm not sure if it's in this chunk, but tell me if you can find it."  # User prompt
    )

    print(f"Result: {result}. Expected: {answer}")

if __name__ == "__main__":
    main()