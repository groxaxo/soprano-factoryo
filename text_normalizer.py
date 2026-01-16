"""
Text normalization for Soprano training.

Provides consistent text normalization for Spanish and other language support.
Key transformations:
- Case-insensitive (converts to lowercase)
- ñ → ni mapping for Spanish support
- Removes accents from vowels (á→a, é→e, í→i, ó→o, ú→u, ü→u)
- Strips brackets [] from text
"""
import re
import unicodedata

NORMALIZER_VERSION = "1.0.0"


def normalize_text(text: str) -> str:
    """
    Normalize text for Soprano training.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text with:
        - Lowercase characters
        - ñ replaced with ni
        - Accented vowels normalized (á→a, etc.)
        - Brackets [] removed
    """
    # Convert to lowercase
    text = text.lower()

    # Spanish normalization: ñ → ni (must be done before accent removal)
    text = text.replace('ñ', 'ni')

    # Remove accents from vowels using Unicode normalization
    # NFD decomposition separates base characters from combining diacritical marks
    text = unicodedata.normalize('NFD', text)
    # Remove combining diacritical marks (accent marks)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    # Remove square brackets (but keep content inside)
    text = re.sub(r'[\[\]]', '', text)

    return text
