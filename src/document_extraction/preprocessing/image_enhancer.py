"""Amélioration de la qualité des images pour l'OCR."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Améliore la qualité des images pour optimiser l'OCR."""

    def __init__(
        self,
        denoise: bool = True,
        binarize: bool = True,
        deskew: bool = True,
        enhance_contrast: bool = True,
    ):
        """
        Initialise l'enhancer d'images.

        Args:
            denoise: Appliquer le denoising
            binarize: Appliquer la binarisation adaptative
            deskew: Redresser automatiquement l'image
            enhance_contrast: Améliorer le contraste
        """
        self.denoise = denoise
        self.binarize = binarize
        self.deskew = deskew
        self.enhance_contrast = enhance_contrast

        logger.info(
            f"ImageEnhancer initialisé (denoise={denoise}, binarize={binarize}, "
            f"deskew={deskew}, enhance_contrast={enhance_contrast})"
        )

    def enhance(self, image: Image.Image | np.ndarray) -> Image.Image:
        """
        Applique les améliorations à une image.

        Args:
            image: Image PIL ou array numpy

        Returns:
            Image PIL améliorée
        """
        # Conversion en array numpy si nécessaire
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()

        # Conversion en niveaux de gris si nécessaire
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        logger.debug(f"Image originale: {gray.shape}")

        # Pipeline d'amélioration
        if self.deskew:
            gray = self._deskew_image(gray)

        if self.denoise:
            gray = self._denoise_image(gray)

        if self.enhance_contrast:
            gray = self._enhance_contrast(gray)

        if self.binarize:
            gray = self._binarize_image(gray)

        return Image.fromarray(gray)

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redresse une image inclinée.

        Args:
            image: Image en niveaux de gris

        Returns:
            Image redressée
        """
        try:
            # Détection des bords
            edges = cv2.Canny(image, 50, 150, apertureSize=3)

            # Détection des lignes avec Hough
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10,
            )

            if lines is None or len(lines) == 0:
                logger.debug("Aucune ligne détectée pour le deskew")
                return image

            # Calcul de l'angle médian des lignes
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Filtrer les angles proches de l'horizontal
                    if abs(angle) < 45:
                        angles.append(angle)

            if not angles:
                return image

            median_angle = np.median(angles)

            # Ignorer les très petits angles
            if abs(median_angle) < 0.5:
                logger.debug(f"Angle négligeable: {median_angle:.2f}°")
                return image

            logger.debug(f"Deskew: rotation de {median_angle:.2f}°")

            # Rotation de l'image
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

            return rotated

        except Exception as e:
            logger.warning(f"Erreur lors du deskew: {e}")
            return image

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un filtre de débruitage.

        Args:
            image: Image en niveaux de gris

        Returns:
            Image débruitée
        """
        try:
            # Débruitage non-local means
            denoised = cv2.fastNlMeansDenoising(
                image,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            logger.debug("Denoising appliqué")
            return denoised

        except Exception as e:
            logger.warning(f"Erreur lors du denoising: {e}")
            return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore le contraste de l'image.

        Args:
            image: Image en niveaux de gris

        Returns:
            Image avec contraste amélioré
        """
        try:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            logger.debug("Amélioration du contraste appliquée")
            return enhanced

        except Exception as e:
            logger.warning(f"Erreur lors de l'amélioration du contraste: {e}")
            return image

    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applique une binarisation adaptative.

        Args:
            image: Image en niveaux de gris

        Returns:
            Image binarisée
        """
        try:
            # Binarisation adaptative (Gaussian)
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2,
            )
            logger.debug("Binarisation appliquée")
            return binary

        except Exception as e:
            logger.warning(f"Erreur lors de la binarisation: {e}")
            return image

    def enhance_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        """
        Améliore une liste d'images.

        Args:
            images: Liste d'images PIL

        Returns:
            Liste d'images améliorées
        """
        enhanced_images = []
        for i, img in enumerate(images):
            logger.debug(f"Amélioration de l'image {i + 1}/{len(images)}")
            enhanced_images.append(self.enhance(img))
        return enhanced_images


def load_image(file_path: str | Path) -> Image.Image:
    """
    Charge une image depuis un fichier.

    Args:
        file_path: Chemin vers l'image

    Returns:
        Image PIL

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le fichier n'est pas une image valide
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")

    try:
        image = Image.open(path)
        # Convertir en RGB si nécessaire (pour les images RGBA ou palettisées)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Impossible de charger l'image: {e}")


def preprocess_document(
    file_path: str | Path,
    pdf_converter: Optional["PDFConverter"] = None,
    image_enhancer: Optional[ImageEnhancer] = None,
) -> list[Image.Image]:
    """
    Prétraite un document (PDF ou image) pour l'OCR.

    Args:
        file_path: Chemin vers le document
        pdf_converter: Instance de PDFConverter (créé si None)
        image_enhancer: Instance d'ImageEnhancer (créé si None)

    Returns:
        Liste d'images prétraitées
    """
    from .pdf_converter import PDFConverter, is_pdf, is_image

    path = Path(file_path)

    if pdf_converter is None:
        pdf_converter = PDFConverter()

    if image_enhancer is None:
        image_enhancer = ImageEnhancer()

    # Conversion si PDF
    if is_pdf(path):
        images = pdf_converter.convert_file(path)
    elif is_image(path):
        images = [load_image(path)]
    else:
        raise ValueError(f"Format de fichier non supporté: {path.suffix}")

    # Amélioration des images
    enhanced_images = image_enhancer.enhance_batch(images)

    return enhanced_images
