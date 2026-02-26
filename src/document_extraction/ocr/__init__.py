"""Module OCR pour l'extraction de texte."""

from .tesseract_ocr import (
    TesseractOCR,
    TesseractNotFoundError,
    OCRError,
    OCRResult,
    WordBox,
    get_available_languages,
    is_tesseract_available,
)

__all__ = [
    "TesseractOCR",
    "TesseractNotFoundError",
    "OCRError",
    "OCRResult",
    "WordBox",
    "get_available_languages",
    "is_tesseract_available",
]
