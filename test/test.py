import random

def generate_massive_context(num_lines: int = 1_000_000, answer: str = "1298418") -> str:
    print(f"Generating massive context with {num_lines} lines...")
    
    # Set of random words to use
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    
    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))
    
    # Insert the magic number at a random position (somewhere in the middle)
    magic_position = random.randint(int(num_lines*0.4), int(num_lines*0.6))
    lines[magic_position] = f"The magic number is {answer}"
    
    print(f"Magic number inserted at position {magic_position}")
    
    return "\n".join(lines)

# Test for the 'generate_massive_context' function. Generate and write to a .txt file
def main():
    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=100, answer=answer)
    with open("massive_context.txt", "w") as f:
        f.write(context)

if __name__ == "__main__":
    main()