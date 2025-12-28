import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize user input and titles to a canonical form.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text
