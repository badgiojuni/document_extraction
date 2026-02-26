"""Pipeline d'extraction de documents."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .config import get_config, Config
from .models import Invoice, Contract
from .preprocessing import PDFConverter, ImageEnhancer, is_pdf, is_image, load_image
from .ocr import TesseractOCR, OCRResult, is_tesseract_available
from .llm import (
    VertexAIClient,
    MockLLMClient,
    BaseLLMClient,
    InvoiceExtractor,
    ContractExtractor,
    DocumentClassifier,
    ExtractionError,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Résultat complet d'une extraction."""

    document_type: str
    data: Union[Invoice, Contract]
    ocr_result: OCRResult
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convertit le résultat en dictionnaire."""
        return {
            "document_type": self.document_type,
            "success": self.success,
            "error_message": self.error_message,
            "data": self.data.to_dict_display() if self.data else None,
            "ocr": {
                "word_count": self.ocr_result.word_count if self.ocr_result else 0,
                "confidence": self.ocr_result.confidence if self.ocr_result else 0,
                "processing_time_ms": self.ocr_result.processing_time_ms if self.ocr_result else 0,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convertit le résultat en JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


class ExtractionPipeline:
    """Pipeline complet d'extraction de documents."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialise le pipeline.

        Args:
            config: Configuration (charge la config par défaut si None)
        """
        self.config = config or get_config()

        # Initialisation des composants
        self._init_preprocessing()
        self._init_ocr()
        self._init_llm()

        logger.info("Pipeline d'extraction initialisé")

    def _init_preprocessing(self) -> None:
        """Initialise les composants de preprocessing."""
        pdf_config = self.config.preprocessing.pdf
        img_config = self.config.preprocessing.image

        self.pdf_converter = PDFConverter(
            dpi=pdf_config.dpi,
            output_format=pdf_config.output_format,
        )

        self.image_enhancer = ImageEnhancer(
            denoise=img_config.denoise,
            binarize=img_config.binarize,
            deskew=img_config.deskew,
            enhance_contrast=img_config.enhance_contrast,
        )

    def _init_ocr(self) -> None:
        """Initialise le composant OCR."""
        if not is_tesseract_available():
            logger.warning("Tesseract non disponible - OCR désactivé")
            self.ocr = None
            return

        ocr_config = self.config.ocr.tesseract
        self.ocr = TesseractOCR(
            lang=ocr_config.lang,
            psm=ocr_config.psm,
            oem=ocr_config.oem,
        )

    def _init_llm(self) -> None:
        """Initialise le client LLM."""
        vertex_config = self.config.vertex_ai

        if vertex_config.use_mock:
            logger.info("Utilisation du client LLM mock")
            self.llm_client: BaseLLMClient = MockLLMClient()
        else:
            try:
                self.llm_client = VertexAIClient(
                    project_id=vertex_config.project_id,
                    location=vertex_config.location,
                    model_name=vertex_config.model_name,
                    credentials_path=vertex_config.credentials_path,
                )
                if not self.llm_client.is_available():
                    logger.warning("Vertex AI non disponible, fallback sur mock")
                    self.llm_client = MockLLMClient()
            except Exception as e:
                logger.warning(f"Erreur Vertex AI ({e}), fallback sur mock")
                self.llm_client = MockLLMClient()

        # Initialisation des extracteurs
        self.invoice_extractor = InvoiceExtractor(self.llm_client)
        self.contract_extractor = ContractExtractor(self.llm_client)
        self.classifier = DocumentClassifier(self.llm_client)

    def process_file(
        self,
        file_path: Union[str, Path],
        document_type: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Traite un fichier et extrait les données.

        Args:
            file_path: Chemin vers le document (PDF ou image)
            document_type: Type de document ("invoice" ou "contract")
                          Si None, détection automatique

        Returns:
            ExtractionResult avec les données extraites
        """
        file_path = Path(file_path)
        logger.info(f"Traitement du fichier: {file_path}")

        try:
            # Vérification du fichier
            if not file_path.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

            # 1. Preprocessing
            images = self._preprocess(file_path)

            # 2. OCR
            ocr_result = self._extract_text(images)

            # 3. Classification (si type non spécifié)
            if document_type is None:
                document_type = self.classifier.classify(ocr_result.text)
                logger.info(f"Type de document détecté: {document_type}")

            # 4. Extraction structurée
            data = self._extract_structured_data(ocr_result.text, document_type)

            return ExtractionResult(
                document_type=document_type,
                data=data,
                ocr_result=ocr_result,
                success=True,
            )

        except Exception as e:
            logger.error(f"Erreur lors du traitement: {e}")
            return ExtractionResult(
                document_type=document_type or "unknown",
                data=None,
                ocr_result=None,
                success=False,
                error_message=str(e),
            )

    def process_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        document_type: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Traite des bytes et extrait les données.

        Args:
            file_bytes: Contenu du fichier en bytes
            filename: Nom du fichier (pour déterminer le type)
            document_type: Type de document si connu

        Returns:
            ExtractionResult avec les données extraites
        """
        logger.info(f"Traitement de {len(file_bytes)} bytes ({filename})")

        try:
            # 1. Preprocessing
            if filename.lower().endswith(".pdf"):
                images = self.pdf_converter.convert_bytes(file_bytes)
            else:
                from PIL import Image
                import io
                images = [Image.open(io.BytesIO(file_bytes))]

            # Amélioration des images
            images = self.image_enhancer.enhance_batch(images)

            # 2. OCR
            ocr_result = self._extract_text(images)

            # 3. Classification
            if document_type is None:
                document_type = self.classifier.classify(ocr_result.text)
                logger.info(f"Type de document détecté: {document_type}")

            # 4. Extraction structurée
            data = self._extract_structured_data(ocr_result.text, document_type)

            return ExtractionResult(
                document_type=document_type,
                data=data,
                ocr_result=ocr_result,
                success=True,
            )

        except Exception as e:
            logger.error(f"Erreur lors du traitement: {e}")
            return ExtractionResult(
                document_type=document_type or "unknown",
                data=None,
                ocr_result=None,
                success=False,
                error_message=str(e),
            )

    def _preprocess(self, file_path: Path) -> list:
        """Prétraite le document."""
        logger.debug("Preprocessing du document...")

        if is_pdf(file_path):
            images = self.pdf_converter.convert_file(file_path)
        elif is_image(file_path):
            images = [load_image(file_path)]
        else:
            raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")

        # Amélioration des images
        enhanced = self.image_enhancer.enhance_batch(images)

        logger.debug(f"Preprocessing terminé: {len(enhanced)} image(s)")
        return enhanced

    def _extract_text(self, images: list) -> OCRResult:
        """Extrait le texte des images."""
        if self.ocr is None:
            raise RuntimeError("OCR non disponible (Tesseract non installé)")

        logger.debug("Extraction OCR...")
        result = self.ocr.extract_from_multiple(images, separator="\n\n")

        logger.debug(f"OCR terminé: {result.word_count} mots, confiance={result.confidence:.2%}")
        return result

    def _extract_structured_data(
        self,
        text: str,
        document_type: str,
    ) -> Union[Invoice, Contract]:
        """Extrait les données structurées."""
        logger.debug(f"Extraction structurée ({document_type})...")

        if document_type == "invoice":
            return self.invoice_extractor.extract(text)
        elif document_type == "contract":
            return self.contract_extractor.extract(text)
        else:
            raise ValueError(f"Type de document non supporté: {document_type}")


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Extraction de données depuis des documents"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Chemin vers le document à traiter",
    )
    parser.add_argument(
        "--type", "-t",
        choices=["invoice", "contract"],
        help="Type de document (détection auto si non spécifié)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Fichier de sortie JSON (stdout si non spécifié)",
    )
    parser.add_argument(
        "--config", "-c",
        help="Fichier de configuration YAML",
    )

    args = parser.parse_args()

    # Chargement de la configuration
    if args.config:
        from .config import load_config, setup_logging
        config = load_config(args.config)
        setup_logging(config)
    else:
        config = None

    # Traitement
    pipeline = ExtractionPipeline(config)
    result = pipeline.process_file(args.input, args.type)

    # Sortie
    output_json = result.to_json()

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Résultat sauvegardé dans {args.output}")
    else:
        print(output_json)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
