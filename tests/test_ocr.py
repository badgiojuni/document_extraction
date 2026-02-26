"""Tests pour le module OCR."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from document_extraction.ocr import (
    TesseractOCR,
    OCRResult,
    WordBox,
    is_tesseract_available,
)


# Skip tous les tests si Tesseract n'est pas installé
pytestmark = pytest.mark.skipif(
    not is_tesseract_available(),
    reason="Tesseract n'est pas installé",
)


class TestOCRResult:
    """Tests pour OCRResult."""

    def test_is_empty_true(self):
        """Test résultat vide."""
        result = OCRResult(
            text="   ",
            confidence=0.0,
            language="fra",
            word_count=0,
        )
        assert result.is_empty() is True

    def test_is_empty_false(self):
        """Test résultat non vide."""
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            language="eng",
            word_count=2,
        )
        assert result.is_empty() is False

    def test_get_lines(self):
        """Test extraction des lignes."""
        result = OCRResult(
            text="Ligne 1\n\nLigne 2\n  \nLigne 3",
            confidence=0.9,
            language="fra",
            word_count=6,
        )
        lines = result.get_lines()
        assert lines == ["Ligne 1", "Ligne 2", "Ligne 3"]


class TestWordBox:
    """Tests pour WordBox."""

    def test_properties(self):
        """Test les propriétés calculées."""
        box = WordBox(
            text="test",
            confidence=0.95,
            left=10,
            top=20,
            width=50,
            height=15,
        )
        assert box.right == 60
        assert box.bottom == 35


class TestTesseractOCR:
    """Tests pour TesseractOCR."""

    @pytest.fixture
    def ocr(self) -> TesseractOCR:
        """Crée une instance TesseractOCR."""
        return TesseractOCR(lang="eng", psm=6, oem=3)

    @pytest.fixture
    def text_image(self) -> Image.Image:
        """Crée une image avec du texte."""
        # Créer une image blanche
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)

        # Dessiner du texte noir
        try:
            # Essayer d'utiliser une police système
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except (OSError, IOError):
            # Fallback sur la police par défaut
            font = ImageFont.load_default()

        draw.text((20, 30), "Hello World Test 123", fill="black", font=font)

        return img

    @pytest.fixture
    def blank_image(self) -> Image.Image:
        """Crée une image blanche vide."""
        return Image.new("RGB", (200, 100), color="white")

    def test_init_default(self):
        """Test l'initialisation par défaut."""
        ocr = TesseractOCR()
        assert ocr.lang == "fra"
        assert ocr.psm == 6
        assert ocr.oem == 3

    def test_init_custom(self):
        """Test l'initialisation personnalisée."""
        ocr = TesseractOCR(lang="eng+fra", psm=3, oem=1)
        assert ocr.lang == "eng+fra"
        assert ocr.psm == 3
        assert ocr.oem == 1

    def test_extract_text(self, ocr, text_image):
        """Test l'extraction de texte basique."""
        result = ocr.extract_text(text_image)

        assert isinstance(result, OCRResult)
        assert result.language == "eng"
        assert result.processing_time_ms is not None
        assert result.processing_time_ms > 0
        # Le texte devrait contenir au moins quelques caractères reconnus
        assert len(result.text) > 0

    def test_extract_blank_image(self, ocr, blank_image):
        """Test avec une image vide."""
        result = ocr.extract_text(blank_image)

        assert isinstance(result, OCRResult)
        # Une image blanche devrait donner peu ou pas de texte
        assert result.word_count <= 1

    def test_extract_words_with_boxes(self, ocr, text_image):
        """Test l'extraction avec bounding boxes."""
        boxes = ocr.extract_words_with_boxes(text_image)

        assert isinstance(boxes, list)
        # Devrait trouver au moins quelques mots
        if boxes:
            box = boxes[0]
            assert isinstance(box, WordBox)
            assert box.confidence >= 0
            assert box.confidence <= 1
            assert box.width > 0
            assert box.height > 0

    def test_extract_from_multiple(self, ocr, text_image):
        """Test l'extraction multi-pages."""
        images = [text_image, text_image]
        result = ocr.extract_from_multiple(images)

        assert isinstance(result, OCRResult)
        # Le texte combiné devrait contenir le séparateur de page
        assert "Page 2" in result.text or result.word_count > 0

    def test_extract_from_multiple_empty(self, ocr):
        """Test avec liste vide."""
        result = ocr.extract_from_multiple([])

        assert result.text == ""
        assert result.word_count == 0

    def test_extract_from_file_not_found(self, ocr):
        """Test avec fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            ocr.extract_text_from_file("/nonexistent/image.png")


class TestHelperFunctions:
    """Tests pour les fonctions utilitaires."""

    def test_is_tesseract_available(self):
        """Test la détection de Tesseract."""
        # Ce test s'exécute seulement si Tesseract est disponible
        assert is_tesseract_available() is True
