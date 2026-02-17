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
    NUM_LINES = 50  # For testing, we can use a smaller number. Adjust as needed.
    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=NUM_LINES, answer=answer)

    # Save context file for debugging and inspection
    with open("massive_context.txt", "w") as f:
        f.write(context)

    rlm = RLM(
        backend="ollama",
        backend_kwargs={"model_name": "qwen3:8b"}, #qwen2.5:7b, qwen2.5-coder:7b, qwen3:8b, deepseek-r1:8b
        environment="local",
        environment_kwargs={},
        max_depth=1,
        max_iterations=30,
        logger=logger,
        verbose=True,  # For printing to console with rich, disabled by default.
    )

    result = rlm.completion(
        prompt=context,  # The file content (large context)
        root_prompt="I'm looking for a magic number. What is it?"  # User prompt
    )

    print(f"Result: {result}. Expected: {answer}")

if __name__ == "__main__":
    main()