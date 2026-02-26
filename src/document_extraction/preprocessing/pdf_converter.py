"""Conversion de documents PDF en images."""

import logging
import tempfile
from pathlib import Path
from typing import BinaryIO

from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
from PIL import Image

logger = logging.getLogger(__name__)


class PDFConversionError(Exception):
    """Erreur lors de la conversion PDF."""

    pass


class PDFConverter:
    """Convertisseur de documents PDF en images."""

    def __init__(self, dpi: int = 300, output_format: str = "PNG"):
        """
        Initialise le convertisseur PDF.

        Args:
            dpi: Résolution de sortie en DPI (défaut: 300)
            output_format: Format de sortie des images (PNG, JPEG)
        """
        self.dpi = dpi
        self.output_format = output_format.upper()
        logger.info(f"PDFConverter initialisé (DPI={dpi}, format={output_format})")

    def convert_file(self, pdf_path: str | Path) -> list[Image.Image]:
        """
        Convertit un fichier PDF en liste d'images.

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            Liste d'images PIL (une par page)

        Raises:
            PDFConversionError: En cas d'erreur de conversion
            FileNotFoundError: Si le fichier n'existe pas
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise PDFConversionError(f"Le fichier n'est pas un PDF: {pdf_path}")

        logger.info(f"Conversion du PDF: {pdf_path}")

        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.output_format.lower(),
            )
            logger.info(f"Conversion réussie: {len(images)} page(s)")
            return images

        except PDFInfoNotInstalledError:
            raise PDFConversionError(
                "poppler-utils n'est pas installé. "
                "Installez-le avec: sudo apt-get install poppler-utils"
            )
        except PDFPageCountError as e:
            raise PDFConversionError(f"Impossible de lire le PDF: {e}")
        except Exception as e:
            raise PDFConversionError(f"Erreur lors de la conversion: {e}")

    def convert_bytes(self, pdf_bytes: bytes) -> list[Image.Image]:
        """
        Convertit des bytes PDF en liste d'images.

        Args:
            pdf_bytes: Contenu du fichier PDF en bytes

        Returns:
            Liste d'images PIL (une par page)

        Raises:
            PDFConversionError: En cas d'erreur de conversion
        """
        if not pdf_bytes:
            raise PDFConversionError("Données PDF vides")

        logger.info(f"Conversion de {len(pdf_bytes)} bytes de PDF")

        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.dpi,
                fmt=self.output_format.lower(),
            )
            logger.info(f"Conversion réussie: {len(images)} page(s)")
            return images

        except PDFInfoNotInstalledError:
            raise PDFConversionError(
                "poppler-utils n'est pas installé. "
                "Installez-le avec: sudo apt-get install poppler-utils"
            )
        except Exception as e:
            raise PDFConversionError(f"Erreur lors de la conversion: {e}")

    def convert_stream(self, pdf_stream: BinaryIO) -> list[Image.Image]:
        """
        Convertit un flux PDF en liste d'images.

        Args:
            pdf_stream: Flux binaire du fichier PDF

        Returns:
            Liste d'images PIL (une par page)
        """
        pdf_bytes = pdf_stream.read()
        return self.convert_bytes(pdf_bytes)

    def save_images(
        self,
        images: list[Image.Image],
        output_dir: str | Path,
        prefix: str = "page",
    ) -> list[Path]:
        """
        Sauvegarde les images dans un répertoire.

        Args:
            images: Liste d'images PIL à sauvegarder
            output_dir: Répertoire de sortie
            prefix: Préfixe pour les noms de fichiers

        Returns:
            Liste des chemins des fichiers créés
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extension = "png" if self.output_format == "PNG" else "jpg"
        saved_paths = []

        for i, image in enumerate(images, start=1):
            filename = f"{prefix}_{i:03d}.{extension}"
            filepath = output_dir / filename
            image.save(filepath, self.output_format)
            saved_paths.append(filepath)
            logger.debug(f"Image sauvegardée: {filepath}")

        logger.info(f"{len(saved_paths)} image(s) sauvegardée(s) dans {output_dir}")
        return saved_paths


def is_pdf(file_path: str | Path) -> bool:
    """Vérifie si un fichier est un PDF."""
    path = Path(file_path)
    return path.suffix.lower() == ".pdf"


def is_image(file_path: str | Path) -> bool:
    """Vérifie si un fichier est une image supportée."""
    path = Path(file_path)
    supported_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    return path.suffix.lower() in supported_extensions
