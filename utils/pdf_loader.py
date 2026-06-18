from pathlib import Path
import pymupdf4llm


def load_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF.

    Args:
        pdf_path = Path to PDF

    Returns:
        Extracted markdown text
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found : {pdf_path}")

    markdown_text: str = pymupdf4llm.to_markdown(str(path))
    return markdown_text
