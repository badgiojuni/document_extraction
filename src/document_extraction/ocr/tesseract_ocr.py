"""Extraction de texte avec Tesseract OCR."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class TesseractNotFoundError(Exception):
    """Tesseract n'est pas installé ou introuvable."""

    pass


class OCRError(Exception):
    """Erreur lors de l'extraction OCR."""

    pass


@dataclass
class OCRResult:
    """Résultat d'une extraction OCR."""

    text: str
    confidence: float
    language: str
    word_count: int
    processing_time_ms: Optional[float] = None

    def is_empty(self) -> bool:
        """Vérifie si le résultat est vide."""
        return len(self.text.strip()) == 0

    def get_lines(self) -> list[str]:
        """Retourne le texte découpé en lignes non vides."""
        return [line.strip() for line in self.text.split("\n") if line.strip()]


@dataclass
class WordBox:
    """Boîte englobante d'un mot détecté."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class TesseractOCR:
    """Extracteur de texte utilisant Tesseract OCR."""

    def __init__(
        self,
        lang: str = "fra",
        psm: int = 6,
        oem: int = 3,
        tesseract_cmd: Optional[str] = None,
    ):
        """
        Initialise l'extracteur Tesseract.

        Args:
            lang: Langue(s) pour la reconnaissance (fra, eng, fra+eng)
            psm: Page Segmentation Mode (0-13)
                 - 3: Fully automatic page segmentation
                 - 6: Assume a single uniform block of text
                 - 11: Sparse text
            oem: OCR Engine Mode (0-3)
                 - 0: Legacy engine only
                 - 1: Neural nets LSTM engine only
                 - 2: Legacy + LSTM engines
                 - 3: Default (based on available)
            tesseract_cmd: Chemin vers l'exécutable Tesseract
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem

        # Configuration du chemin Tesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Vérification de l'installation
        self._verify_installation()

        logger.info(f"TesseractOCR initialisé (lang={lang}, psm={psm}, oem={oem})")

    def _verify_installation(self) -> None:
        """Vérifie que Tesseract est installé."""
        tesseract_path = shutil.which("tesseract")

        if tesseract_path is None:
            raise TesseractNotFoundError(
                "Tesseract n'est pas installé ou introuvable dans le PATH. "
                "Installez-le avec:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-fra\n"
                "  macOS: brew install tesseract tesseract-lang\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            )

        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version {version} trouvé à {tesseract_path}")
        except Exception as e:
            raise TesseractNotFoundError(f"Erreur lors de la vérification de Tesseract: {e}")

    def _get_config(self, custom_config: Optional[str] = None) -> str:
        """Génère la configuration Tesseract."""
        config_parts = [
            f"--psm {self.psm}",
            f"--oem {self.oem}",
        ]

        if custom_config:
            config_parts.append(custom_config)

        return " ".join(config_parts)

    def extract_text(
        self,
        image: Image.Image,
        custom_config: Optional[str] = None,
    ) -> OCRResult:
        """
        Extrait le texte d'une image.

        Args:
            image: Image PIL à traiter
            custom_config: Configuration Tesseract additionnelle

        Returns:
            OCRResult avec le texte extrait et les métadonnées
        """
        import time

        start_time = time.perf_counter()

        try:
            config = self._get_config(custom_config)

            # Extraction du texte
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=config,
            )

            # Extraction des données détaillées pour calculer la confiance
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            # Calcul de la confiance moyenne (ignorer les valeurs -1)
            confidences = [
                int(c) for c in data["conf"] if isinstance(c, (int, float)) and int(c) >= 0
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Comptage des mots
            words = [w for w in data["text"] if w.strip()]

            processing_time = (time.perf_counter() - start_time) * 1000

            result = OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,  # Normaliser entre 0 et 1
                language=self.lang,
                word_count=len(words),
                processing_time_ms=processing_time,
            )

            logger.debug(
                f"OCR terminé: {result.word_count} mots, "
                f"confiance={result.confidence:.2%}, "
                f"temps={processing_time:.1f}ms"
            )

            return result

        except pytesseract.TesseractError as e:
            raise OCRError(f"Erreur Tesseract: {e}")
        except Exception as e:
            raise OCRError(f"Erreur lors de l'extraction OCR: {e}")

    def extract_text_from_file(self, image_path: str | Path) -> OCRResult:
        """
        Extrait le texte d'un fichier image.

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            OCRResult avec le texte extrait
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {path}")

        image = Image.open(path)
        return self.extract_text(image)

    def extract_words_with_boxes(self, image: Image.Image) -> list[WordBox]:
        """
        Extrait les mots avec leurs positions (bounding boxes).

        Args:
            image: Image PIL à traiter

        Returns:
            Liste de WordBox avec positions et confiances
        """
        try:
            config = self._get_config()

            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            words = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                text = data["text"][i]
                conf = data["conf"][i]

                # Ignorer les entrées vides ou avec confiance négative
                if not text.strip() or int(conf) < 0:
                    continue

                words.append(
                    WordBox(
                        text=text,
                        confidence=float(conf) / 100.0,
                        left=data["left"][i],
                        top=data["top"][i],
                        width=data["width"][i],
                        height=data["height"][i],
                    )
                )

            logger.debug(f"Extraction avec boxes: {len(words)} mots détectés")
            return words

        except Exception as e:
            raise OCRError(f"Erreur lors de l'extraction avec boxes: {e}")

    def extract_from_multiple(
        self,
        images: list[Image.Image],
        separator: str = "\n\n--- Page {n} ---\n\n",
    ) -> OCRResult:
        """
        Extrait le texte de plusieurs images (pages).

        Args:
            images: Liste d'images PIL
            separator: Séparateur entre les pages ({n} = numéro de page)

        Returns:
            OCRResult combiné
        """
        if not images:
            return OCRResult(
                text="",
                confidence=0.0,
                language=self.lang,
                word_count=0,
            )

        texts = []
        total_confidence = 0.0
        total_words = 0
        total_time = 0.0

        for i, image in enumerate(images, start=1):
            logger.info(f"OCR page {i}/{len(images)}")
            result = self.extract_text(image)

            if separator and i > 1:
                texts.append(separator.format(n=i))

            texts.append(result.text)
            total_confidence += result.confidence * result.word_count
            total_words += result.word_count
            total_time += result.processing_time_ms or 0

        combined_text = "".join(texts)
        avg_confidence = total_confidence / total_words if total_words > 0 else 0.0

        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            language=self.lang,
            word_count=total_words,
            processing_time_ms=total_time,
        )


def get_available_languages() -> list[str]:
    """Retourne la liste des langues disponibles pour Tesseract."""
    try:
        return pytesseract.get_languages()
    except Exception:
        return []


def is_tesseract_available() -> bool:
    """Vérifie si Tesseract est disponible."""
    return shutil.which("tesseract") is not None
