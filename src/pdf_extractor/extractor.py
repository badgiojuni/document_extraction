import json
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from .client import VertexAIClient


class PDFExtractor:
    """Extracteur d'informations depuis des PDF via VLLM."""

    def __init__(self, client: VertexAIClient, dpi: int = 150):
        self.client = client
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: Path) -> list[bytes]:
        """Convertit un PDF en liste d'images PNG."""
        doc = fitz.open(pdf_path)
        images = []

        for page in doc:
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))

        doc.close()
        return images

    def extract(
        self,
        pdf_path: str | Path,
        prompt: str,
        pages: list[int] | None = None,
    ) -> str:
        """Extrait des informations d'un PDF selon le prompt fourni.

        Args:
            pdf_path: Chemin vers le fichier PDF
            prompt: Instructions d'extraction pour le modèle
            pages: Liste des numéros de pages à traiter (0-indexé), None pour toutes

        Returns:
            Texte extrait par le modèle
        """
        pdf_path = Path(pdf_path)
        images = self.pdf_to_images(pdf_path)

        if pages is not None:
            images = [images[i] for i in pages if i < len(images)]

        return self.client.extract_from_images(images, prompt)

    def extract_structured(
        self,
        pdf_path: str | Path,
        schema: dict[str, Any],
        pages: list[int] | None = None,
    ) -> dict[str, Any]:
        """Extrait des informations structurées selon un schéma JSON.

        Args:
            pdf_path: Chemin vers le fichier PDF
            schema: Schéma JSON décrivant les champs à extraire
            pages: Liste des numéros de pages à traiter

        Returns:
            Dictionnaire avec les données extraites
        """
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        prompt = f"""Analyse ce document et extrait les informations selon ce schéma JSON.
Réponds UNIQUEMENT avec un objet JSON valide, sans texte additionnel.

Schéma attendu:
{schema_str}

Si une information n'est pas trouvée, utilise null.
"""

        result = self.extract(pdf_path, prompt, pages)

        # Nettoie la réponse si elle contient des marqueurs markdown
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]

        return json.loads(result.strip())
