#!/usr/bin/env python3
"""
Script to find a specific sentence in a PDF document.
Supports pypdf, PyPDF2, and pdfplumber libraries.
"""

import re
import sys
from pathlib import Path


def extract_text_pypdf(pdf_path):
    """Extract text using pypdf library."""
    import pypdf

    pages_text = []
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages_text.append((i + 1, text))
    return pages_text


def extract_text_pypdf2(pdf_path):
    """Extract text using PyPDF2 library."""
    import PyPDF2

    pages_text = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages_text.append((i + 1, text))
    return pages_text


def extract_text_pdfplumber(pdf_path):
    """Extract text using pdfplumber library."""
    import pdfplumber

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            pages_text.append((i + 1, text))
    return pages_text


def get_pdf_extractor():
    """Determine which PDF library is available and return the appropriate extractor."""
    try:
        import pypdf
        return extract_text_pypdf, "pypdf"
    except ImportError:
        pass

    try:
        import PyPDF2
        return extract_text_pypdf2, "PyPDF2"
    except ImportError:
        pass

    try:
        import pdfplumber
        return extract_text_pdfplumber, "pdfplumber"
    except ImportError:
        pass

    return None, None


def normalize_text(text):
    """Normalize text by removing extra whitespace and line breaks."""
    # Replace multiple whitespace/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_sentence_in_pdf(pdf_path, search_sentence, case_sensitive=False):
    """
    Find a sentence in a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        search_sentence: The sentence to search for
        case_sensitive: Whether the search should be case-sensitive
    
    Returns:
        List of tuples (page_number, context) where the sentence was found
    """
    extractor, library_name = get_pdf_extractor()

    if extractor is None:
        print("Error: No PDF library found. Please install one of:")
        print("  - pypdf:       pip install pypdf")
        print("  - PyPDF2:      pip install PyPDF2")
        print("  - pdfplumber:  pip install pdfplumber")
        sys.exit(1)

    print(f"Using {library_name} to read PDF...")

    # Extract text from all pages
    pages_text = extractor(pdf_path)

    # Normalize the search sentence
    search_normalized = normalize_text(search_sentence)
    if not case_sensitive:
        search_normalized = search_normalized.lower()

    # Search for the sentence
    results = []
    for page_num, page_text in pages_text:
        # Normalize the page text
        text_normalized = normalize_text(page_text)
        if not case_sensitive:
            text_normalized = text_normalized.lower()

        # Check if the sentence is in this page
        if search_normalized in text_normalized:
            # Find the position and extract context
            pos = text_normalized.find(search_normalized)

            # Extract context (100 chars before and after)
            start = max(0, pos - 100)
            end = min(len(text_normalized), pos + len(search_normalized) + 100)
            context = text_normalized[start:end]

            # Highlight the found sentence
            context = context.replace(search_normalized, f">>>{search_normalized}<<<")

            results.append((page_num, context))

    return results


def main():
    # Default values
    pdf_path = "Speech and Language Processing - Daniel Jurafsky, James H. Martin.pdf"
    search_sentence = "Chinese has about 100,000 Chinese characters"

    # Parse command line arguments
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        search_sentence = sys.argv[2]

    # Convert to Path object
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print(f"Searching for: '{search_sentence}'")
    print(f"In PDF: {pdf_file.name}")
    print("-" * 80)

    # Search for the sentence
    results = find_sentence_in_pdf(pdf_file, search_sentence)

    if results:
        print(f"\n✓ Found {len(results)} occurrence(s):\n")
        for page_num, context in results:
            print(f"Page {page_num}:")
            print(f"  ...{context}...")
            print()
    else:
        print("\n✗ Sentence not found in the PDF.")
        print("\nTips:")
        print("  - Try searching for a shorter phrase")
        print("  - Check for typos or formatting differences")
        print("  - The sentence might be split across pages or have unusual spacing")


if __name__ == "__main__":
    main()
