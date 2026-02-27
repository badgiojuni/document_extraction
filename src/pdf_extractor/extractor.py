import json
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from .client import VertexAIClient

RESULTS_DIR = Path("data/results")


class PDFExtractor:
    """Extracteur d'informations depuis des PDF via VLLM."""

    def __init__(self, client: VertexAIClient, dpi: int = 150, output_dir: Path | None = None):
        self.client = client
        self.dpi = dpi
        self.output_dir = output_dir or RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _save_result(self, pdf_path: Path, result: dict[str, Any]) -> Path:
        """Sauvegarde le résultat dans un fichier JSON."""
        output_file = self.output_dir / f"{pdf_path.stem}.json"
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return output_file

    def extract(
        self,
        pdf_path: str | Path,
        prompt: str,
        pages: list[int] | None = None,
    ) -> dict[str, Any]:
        """Extrait des informations d'un PDF selon le prompt fourni.

        Args:
            pdf_path: Chemin vers le fichier PDF
            prompt: Instructions d'extraction pour le modèle
            pages: Liste des numéros de pages à traiter (0-indexé), None pour toutes

        Returns:
            Dictionnaire avec le résultat et le chemin du fichier JSON
        """
        pdf_path = Path(pdf_path)
        images = self.pdf_to_images(pdf_path)

        if pages is not None:
            images = [images[i] for i in pages if i < len(images)]

        text = self.client.extract_from_images(images, prompt)

        result = {
            "source_file": str(pdf_path),
            "prompt": prompt,
            "pages_processed": pages if pages else list(range(len(images))),
            "extraction": text,
        }

        output_file = self._save_result(pdf_path, result)
        result["output_file"] = str(output_file)

        return result

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
            Dictionnaire avec les données extraites et métadonnées
        """
        pdf_path = Path(pdf_path)
        images = self.pdf_to_images(pdf_path)

        if pages is not None:
            images = [images[i] for i in pages if i < len(images)]

        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        prompt = f"""Analyse ce document et extrait les informations selon ce schéma JSON.
Réponds UNIQUEMENT avec un objet JSON valide, sans texte additionnel.

Schéma attendu:
{schema_str}

Si une information n'est pas trouvée, utilise null.
"""

        raw_response = self.client.extract_from_images(images, prompt)

        # Nettoie la réponse si elle contient des marqueurs markdown
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        extracted_data = json.loads(cleaned.strip())

        result = {
            "source_file": str(pdf_path),
            "schema": schema,
            "pages_processed": pages if pages else list(range(len(images))),
            "extraction": extracted_data,
        }

        output_file = self._save_result(pdf_path, result)
        result["output_file"] = str(output_file)

        return result
