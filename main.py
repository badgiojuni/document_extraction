#!/usr/bin/env python3
"""Point d'entrée pour l'extraction de documents PDF."""

import argparse
import json
import os
from pathlib import Path

from src.pdf_extractor import PDFExtractor, VertexAIClient


def main():
    parser = argparse.ArgumentParser(description="Extraction d'informations depuis PDF")
    parser.add_argument("pdf", type=Path, help="Chemin vers le fichier PDF")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Extrait toutes les informations importantes de ce document.",
        help="Prompt d'extraction",
    )
    parser.add_argument(
        "--schema", "-s",
        type=Path,
        help="Fichier JSON contenant le schéma d'extraction",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="ID du projet Google Cloud",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=os.getenv("GOOGLE_CLOUD_LOCATION"),
        help="Région Vertex AI",
    )
    parser.add_argument(
        "--pages",
        type=str,
        help="Pages à traiter (ex: '0,1,2' ou '0-5')",
    )

    args = parser.parse_args()

    if not args.project:
        print("Erreur: Spécifiez --project ou définissez GOOGLE_CLOUD_PROJECT")
        return 1

    # Parse pages
    pages = None
    if args.pages:
        pages = []
        for part in args.pages.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                pages.extend(range(start, end + 1))
            else:
                pages.append(int(part))

    # Initialise le client et l'extracteur
    client = VertexAIClient(project_id=args.project, location=args.location)
    extractor = PDFExtractor(client)

    # Extraction
    if args.schema:
        schema = json.loads(args.schema.read_text())
        result = extractor.extract_structured(args.pdf, schema, pages)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = extractor.extract(args.pdf, args.prompt, pages)
        print(result)

    return 0


if __name__ == "__main__":
    exit(main())
