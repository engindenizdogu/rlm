from pathlib import Path
import tiktoken

def tokenize(input: str | Path, model: str = "gpt-5-nano-2025-08-07") -> int:
    """
    Tokenize a string or text file path and return the token count.

    Args:
        input: The input string or path to a .txt file
        model: The model name to use for encoding (default: "gpt-5-nano")

    Returns:
        Number of tokens
    """
    if isinstance(input, Path) and input.is_file():
        text = input.read_text(encoding="utf-8")
    elif isinstance(input, str):
        try:
            path = Path(input)
            if path.is_file():
                text = path.read_text(encoding="utf-8")
            else:
                text = input
        except (OSError, ValueError):
            text = input
    else:
        text = str(input)

    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def count_words(input: str | Path) -> int:
    """
    Count words in a string or a text file path.

    Args:
        input: The input string or path to a .txt file
    
    Returns:
        Number of words
    """
    if isinstance(input, Path) and input.is_file():
        text = input.read_text(encoding="utf-8")
    elif isinstance(input, str):
        try:
            path = Path(input)
            if path.is_file():
                text = path.read_text(encoding="utf-8")
            else:
                text = input
        except (OSError, ValueError):
            text = input
    else:
        text = str(input)

    words = text.split()
    return words, len(words)
