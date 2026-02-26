"""Tests pour le module de preprocessing."""

import numpy as np
import pytest
from PIL import Image

from document_extraction.preprocessing import (
    PDFConverter,
    ImageEnhancer,
    is_pdf,
    is_image,
)


class TestPDFConverter:
    """Tests pour PDFConverter."""

    def test_init_default_values(self):
        """Test l'initialisation avec les valeurs par défaut."""
        converter = PDFConverter()
        assert converter.dpi == 300
        assert converter.output_format == "PNG"

    def test_init_custom_values(self):
        """Test l'initialisation avec des valeurs personnalisées."""
        converter = PDFConverter(dpi=150, output_format="JPEG")
        assert converter.dpi == 150
        assert converter.output_format == "JPEG"

    def test_convert_nonexistent_file(self):
        """Test la conversion d'un fichier inexistant."""
        converter = PDFConverter()
        with pytest.raises(FileNotFoundError):
            converter.convert_file("/nonexistent/path/file.pdf")

    def test_convert_non_pdf_file(self, tmp_path):
        """Test la conversion d'un fichier non-PDF."""
        # Créer un fichier texte avec extension .txt
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")

        converter = PDFConverter()
        from document_extraction.preprocessing import PDFConversionError

        with pytest.raises(PDFConversionError):
            converter.convert_file(txt_file)

    def test_convert_empty_bytes(self):
        """Test la conversion de bytes vides."""
        converter = PDFConverter()
        from document_extraction.preprocessing import PDFConversionError

        with pytest.raises(PDFConversionError):
            converter.convert_bytes(b"")


class TestImageEnhancer:
    """Tests pour ImageEnhancer."""

    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """Crée une image de test."""
        # Image grise avec du bruit
        np.random.seed(42)
        data = np.random.randint(100, 200, (100, 100), dtype=np.uint8)
        return Image.fromarray(data)

    @pytest.fixture
    def color_image(self) -> Image.Image:
        """Crée une image couleur de test."""
        np.random.seed(42)
        data = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(data)

    def test_init_default_values(self):
        """Test l'initialisation avec les valeurs par défaut."""
        enhancer = ImageEnhancer()
        assert enhancer.denoise is True
        assert enhancer.binarize is True
        assert enhancer.deskew is True
        assert enhancer.enhance_contrast is True

    def test_init_custom_values(self):
        """Test l'initialisation avec des valeurs personnalisées."""
        enhancer = ImageEnhancer(
            denoise=False,
            binarize=False,
            deskew=False,
            enhance_contrast=False,
        )
        assert enhancer.denoise is False
        assert enhancer.binarize is False
        assert enhancer.deskew is False
        assert enhancer.enhance_contrast is False

    def test_enhance_grayscale_image(self, sample_image):
        """Test l'amélioration d'une image en niveaux de gris."""
        enhancer = ImageEnhancer()
        result = enhancer.enhance(sample_image)

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_enhance_color_image(self, color_image):
        """Test l'amélioration d'une image couleur."""
        enhancer = ImageEnhancer()
        result = enhancer.enhance(color_image)

        assert isinstance(result, Image.Image)
        assert result.size == color_image.size

    def test_enhance_numpy_array(self):
        """Test l'amélioration depuis un array numpy."""
        np.random.seed(42)
        array = np.random.randint(100, 200, (100, 100), dtype=np.uint8)

        enhancer = ImageEnhancer()
        result = enhancer.enhance(array)

        assert isinstance(result, Image.Image)

    def test_enhance_batch(self, sample_image):
        """Test l'amélioration d'un batch d'images."""
        enhancer = ImageEnhancer()
        images = [sample_image, sample_image.copy()]
        results = enhancer.enhance_batch(images)

        assert len(results) == 2
        assert all(isinstance(img, Image.Image) for img in results)

    def test_no_processing(self, sample_image):
        """Test sans aucun traitement."""
        enhancer = ImageEnhancer(
            denoise=False,
            binarize=False,
            deskew=False,
            enhance_contrast=False,
        )
        result = enhancer.enhance(sample_image)

        assert isinstance(result, Image.Image)


class TestHelperFunctions:
    """Tests pour les fonctions utilitaires."""

    def test_is_pdf(self):
        """Test la détection de fichiers PDF."""
        assert is_pdf("document.pdf") is True
        assert is_pdf("document.PDF") is True
        assert is_pdf("/path/to/file.pdf") is True
        assert is_pdf("document.png") is False
        assert is_pdf("document.txt") is False

    def test_is_image(self):
        """Test la détection de fichiers image."""
        assert is_image("photo.png") is True
        assert is_image("photo.PNG") is True
        assert is_image("photo.jpg") is True
        assert is_image("photo.jpeg") is True
        assert is_image("photo.tiff") is True
        assert is_image("photo.bmp") is True
        assert is_image("document.pdf") is False
        assert is_image("document.txt") is False
