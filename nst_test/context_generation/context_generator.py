import random

# A list of false locations to make the search more difficult
STEVENS_LOCATIONS = [
    "Stevens Institute of Technology is located in Amsterdam.",
    "Stevens Institute of Technology is located in Berlin.",
    "Stevens Institute of Technology is located in Tokyo.",
    "Stevens Institute of Technology is located in Sydney.",
    "Stevens Institute of Technology is located in Toronto.",
    "Stevens Institute of Technology is located in Paris.",
    "Stevens Institute of Technology is located in London.",
    "Stevens Institute of Technology is located in Madrid.",
    "Stevens Institute of Technology is located in Rome.",
    "Stevens Institute of Technology is located in Istanbul.",
    "Stevens Institute of Technology is located in Mumbai.",
]

def generate_massive_context(num_lines: int = 1_000_000, number_of_needles: int = 1) -> str:
    print(f"Generating massive context with {num_lines} lines...")

    # Set of random words to use
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    # Insert the magic number at a random position (somewhere near the end)
    magic_positions = []
    for _ in range(number_of_needles):
        magic_position = random.randint(int(num_lines * 0.1), int(num_lines * 0.9))
        magic_positions.append(magic_position)
        lines[magic_position] = f"The magic number is {random.randint(1000000, 9999999)}"

    print(f"Magic numbers inserted at positions {magic_positions}")

    return "\n".join(lines)

def insert_location_needle(context: str) -> str:
    # Select a random false location and insert it into a random position in the context
    false_location = random.choice(STEVENS_LOCATIONS)
    lines = context.split('\n')
    insert_position = random.randint(0, len(lines) - 1)
    lines.insert(insert_position, false_location)
    return '\n'.join(lines)

# Test
def main():
    print("Example of using RLM (REPL) on a needle-in-haystack problem.")
    context = generate_massive_context(num_lines=100, number_of_needles=2)
    context = insert_location_needle(context)
    
    # Save context file for debugging and inspection
    with open("../openai/input/massive_context.txt", "w") as f:
        f.write(context)

    print("Massive context generated and saved to input/massive_context.txt")

if __name__ == "__main__":
    main()