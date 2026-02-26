"""Tests pour le pipeline d'extraction."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from document_extraction.pipeline import ExtractionPipeline, ExtractionResult
from document_extraction.ocr import OCRResult
from document_extraction.models import Invoice


class TestExtractionResult:
    """Tests pour ExtractionResult."""

    def test_to_dict(self):
        """Test conversion en dictionnaire."""
        invoice = Invoice(
            invoice_number="FA-001",
            total_ttc=1200.00,
        )
        ocr_result = OCRResult(
            text="Test",
            confidence=0.95,
            language="fra",
            word_count=10,
            processing_time_ms=150.0,
        )

        result = ExtractionResult(
            document_type="invoice",
            data=invoice,
            ocr_result=ocr_result,
            success=True,
        )

        data = result.to_dict()

        assert data["document_type"] == "invoice"
        assert data["success"] is True
        assert data["data"]["invoice_number"] == "FA-001"
        assert data["ocr"]["word_count"] == 10

    def test_to_json(self):
        """Test conversion en JSON."""
        invoice = Invoice(invoice_number="FA-001")
        ocr_result = OCRResult(
            text="Test",
            confidence=0.9,
            language="fra",
            word_count=5,
        )

        result = ExtractionResult(
            document_type="invoice",
            data=invoice,
            ocr_result=ocr_result,
            success=True,
        )

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "FA-001" in json_str

    def test_error_result(self):
        """Test résultat avec erreur."""
        result = ExtractionResult(
            document_type="unknown",
            data=None,
            ocr_result=None,
            success=False,
            error_message="Test error",
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["error_message"] == "Test error"
        assert data["data"] is None


class TestExtractionPipeline:
    """Tests pour ExtractionPipeline."""

    @pytest.fixture
    def mock_config(self):
        """Configuration mockée."""
        config = MagicMock()
        config.preprocessing.pdf.dpi = 300
        config.preprocessing.pdf.output_format = "PNG"
        config.preprocessing.image.denoise = True
        config.preprocessing.image.binarize = True
        config.preprocessing.image.deskew = True
        config.preprocessing.image.enhance_contrast = True
        config.ocr.tesseract.lang = "fra"
        config.ocr.tesseract.psm = 6
        config.ocr.tesseract.oem = 3
        config.vertex_ai.use_mock = True
        return config

    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """Crée une image de test."""
        data = np.random.randint(200, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(data)

    def test_init_with_mock(self, mock_config):
        """Test initialisation avec config mock."""
        with patch("document_extraction.pipeline.get_config", return_value=mock_config):
            with patch("document_extraction.pipeline.is_tesseract_available", return_value=False):
                pipeline = ExtractionPipeline(mock_config)

        assert pipeline.pdf_converter is not None
        assert pipeline.image_enhancer is not None
        assert pipeline.llm_client is not None

    def test_process_file_not_found(self, mock_config):
        """Test avec fichier inexistant."""
        with patch("document_extraction.pipeline.is_tesseract_available", return_value=False):
            pipeline = ExtractionPipeline(mock_config)

        result = pipeline.process_file("/nonexistent/file.pdf")

        assert result.success is False
        assert "non trouvé" in result.error_message or "not found" in result.error_message.lower()

    def test_process_bytes_pdf(self, mock_config, sample_image):
        """Test traitement de bytes PDF."""
        with patch("document_extraction.pipeline.is_tesseract_available", return_value=True):
            with patch("document_extraction.pipeline.TesseractOCR") as mock_ocr_class:
                # Mock OCR
                mock_ocr = MagicMock()
                mock_ocr.extract_from_multiple.return_value = OCRResult(
                    text="FACTURE N° FA-001 Total TTC: 1200€",
                    confidence=0.9,
                    language="fra",
                    word_count=10,
                )
                mock_ocr_class.return_value = mock_ocr

                pipeline = ExtractionPipeline(mock_config)

                # Mock PDF converter
                pipeline.pdf_converter.convert_bytes = MagicMock(return_value=[sample_image])

                result = pipeline.process_bytes(
                    b"fake pdf content",
                    "test.pdf",
                    document_type="invoice",
                )

        assert result.document_type == "invoice"
        # Le résultat dépend du mock LLM

    def test_classifier_auto_detect(self, mock_config):
        """Test détection automatique du type."""
        with patch("document_extraction.pipeline.is_tesseract_available", return_value=False):
            pipeline = ExtractionPipeline(mock_config)

        # Le classifier mock détecte basé sur les mots-clés
        doc_type = pipeline.classifier.classify("FACTURE Total TTC TVA montant à payer")
        assert doc_type == "invoice"

        doc_type = pipeline.classifier.classify("CONTRAT parties signataires")
        assert doc_type == "contract"


class TestPipelineIntegration:
    """Tests d'intégration du pipeline."""

    @pytest.fixture
    def pipeline_with_mocks(self):
        """Pipeline avec tous les composants mockés."""
        config = MagicMock()
        config.preprocessing.pdf.dpi = 300
        config.preprocessing.pdf.output_format = "PNG"
        config.preprocessing.image.denoise = False
        config.preprocessing.image.binarize = False
        config.preprocessing.image.deskew = False
        config.preprocessing.image.enhance_contrast = False
        config.ocr.tesseract.lang = "fra"
        config.ocr.tesseract.psm = 6
        config.ocr.tesseract.oem = 3
        config.vertex_ai.use_mock = True

        with patch("document_extraction.pipeline.is_tesseract_available", return_value=False):
            pipeline = ExtractionPipeline(config)

        return pipeline

    def test_mock_llm_extraction(self, pipeline_with_mocks):
        """Test extraction avec LLM mock."""
        text = """
        FACTURE N° FA-2024-001
        Date: 15/01/2024
        Total TTC: 1200,00 €
        """

        invoice = pipeline_with_mocks.invoice_extractor.extract(text)

        assert invoice is not None
        assert invoice.extraction_method == "llm"
        assert invoice.confidence_score is not None
