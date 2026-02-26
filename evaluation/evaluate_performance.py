"""Script principal d'évaluation des performances."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from document_extraction.config import load_config, setup_logging, Config
from document_extraction.pipeline import ExtractionPipeline
from document_extraction.llm import MockLLMClient, InvoiceExtractor, ContractExtractor

try:
    from .metrics import (
        EvaluationResults,
        FieldMetrics,
        calculate_field_metrics,
        build_confusion_matrix,
    )
    from .report_generator import generate_html_report, generate_comparison_report
except ImportError:
    from metrics import (
        EvaluationResults,
        FieldMetrics,
        calculate_field_metrics,
        build_confusion_matrix,
    )
    from report_generator import generate_html_report, generate_comparison_report

logger = logging.getLogger(__name__)


# Champs à évaluer par type de document
INVOICE_FIELDS = [
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "client_name",
    "total_ht",
    "total_tva",
    "total_ttc",
]

CONTRACT_FIELDS = [
    "contract_type",
    "signature_date",
    "effective_date",
    "total_amount",
    "duration",
]


def load_annotations(annotations_path: Path) -> list[dict]:
    """Charge les annotations ground truth."""
    if not annotations_path.exists():
        raise FileNotFoundError(f"Fichier d'annotations non trouvé: {annotations_path}")

    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("documents", [])


def evaluate_extraction(
    pipeline: ExtractionPipeline,
    annotations: list[dict],
    samples_dir: Path,
) -> EvaluationResults:
    """
    Évalue le pipeline sur un jeu de données annotées.

    Args:
        pipeline: Pipeline d'extraction
        annotations: Liste des annotations ground truth
        samples_dir: Répertoire contenant les documents

    Returns:
        Résultats de l'évaluation
    """
    results = EvaluationResults()

    predictions_invoice = []
    ground_truth_invoice = []
    predictions_contract = []
    ground_truth_contract = []

    for annotation in annotations:
        filename = annotation.get("filename")
        doc_type = annotation.get("type")
        expected = annotation.get("expected", {})

        file_path = samples_dir / filename

        results.total_documents += 1
        logger.info(f"Évaluation de {filename} ({doc_type})...")

        try:
            start_time = time.perf_counter()

            # Traiter le document
            if file_path.exists():
                result = pipeline.process_file(file_path, document_type=doc_type)
            else:
                # Mode simulation sans fichiers réels
                logger.warning(f"Fichier non trouvé: {file_path}, utilisation du texte simulé")
                text = expected.get("_raw_text", f"Document {doc_type} simulé")
                if doc_type == "invoice":
                    data = pipeline.invoice_extractor.extract(text)
                else:
                    data = pipeline.contract_extractor.extract(text)
                result = type('Result', (), {'success': True, 'data': data})()

            processing_time = (time.perf_counter() - start_time) * 1000
            results.processing_times.append(processing_time)

            if result.success:
                results.successful_extractions += 1

                # Extraire les prédictions
                predicted = result.data.to_dict_display()

                if doc_type == "invoice":
                    predictions_invoice.append(predicted)
                    ground_truth_invoice.append(expected)
                else:
                    predictions_contract.append(predicted)
                    ground_truth_contract.append(expected)
            else:
                results.failed_extractions += 1
                logger.warning(f"Échec de l'extraction: {filename}")

        except Exception as e:
            results.failed_extractions += 1
            logger.error(f"Erreur lors du traitement de {filename}: {e}")

    # Calculer les métriques par champ
    if predictions_invoice:
        invoice_metrics = calculate_field_metrics(
            predictions_invoice,
            ground_truth_invoice,
            INVOICE_FIELDS,
        )
        results.field_metrics.update(invoice_metrics)

    if predictions_contract:
        contract_metrics = calculate_field_metrics(
            predictions_contract,
            ground_truth_contract,
            CONTRACT_FIELDS,
        )
        results.field_metrics.update(contract_metrics)

    return results


def evaluate_ocr_only(
    annotations: list[dict],
) -> EvaluationResults:
    """
    Évalue l'extraction avec OCR seul (regex basique).

    Args:
        annotations: Liste des annotations

    Returns:
        Résultats de l'évaluation OCR seul
    """
    import re

    results = EvaluationResults()
    predictions = []

    for annotation in annotations:
        results.total_documents += 1
        expected = annotation.get("expected", {})
        raw_text = expected.get("_raw_text", "")

        # Extraction basique avec regex
        predicted = {}

        # Numéro de facture
        match = re.search(r"(?:facture|invoice)\s*n[°o]?\s*:?\s*(\S+)", raw_text, re.I)
        predicted["invoice_number"] = match.group(1) if match else None

        # Montants
        match = re.search(r"total\s*ttc\s*:?\s*([\d\s,\.]+)", raw_text, re.I)
        if match:
            try:
                predicted["total_ttc"] = float(match.group(1).replace(" ", "").replace(",", "."))
            except ValueError:
                predicted["total_ttc"] = None

        predictions.append(predicted)
        results.successful_extractions += 1

    # Calculer les métriques
    if predictions:
        metrics = calculate_field_metrics(
            predictions,
            [a.get("expected", {}) for a in annotations],
            ["invoice_number", "total_ttc"],
        )
        results.field_metrics.update(metrics)

    return results


def create_synthetic_dataset(output_dir: Path) -> None:
    """
    Crée un dataset de test synthétique.

    Args:
        output_dir: Répertoire de sortie
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Annotations
    annotations = {
        "documents": [
            {
                "filename": "invoice_001.txt",
                "type": "invoice",
                "expected": {
                    "invoice_number": "FA-2024-001",
                    "invoice_date": "2024-01-15",
                    "supplier_name": "ACME Corporation",
                    "client_name": "Client ABC",
                    "total_ht": 1000.00,
                    "total_tva": 200.00,
                    "total_ttc": 1200.00,
                    "_raw_text": "ACME Corporation\nFACTURE N° FA-2024-001\nDate: 15/01/2024\nClient: Client ABC\nTotal HT: 1000,00€\nTVA 20%: 200,00€\nTotal TTC: 1200,00€"
                }
            },
            {
                "filename": "invoice_002.txt",
                "type": "invoice",
                "expected": {
                    "invoice_number": "FA-2024-002",
                    "invoice_date": "2024-02-20",
                    "supplier_name": "Tech Solutions SARL",
                    "client_name": "Entreprise XYZ",
                    "total_ht": 5000.00,
                    "total_tva": 1000.00,
                    "total_ttc": 6000.00,
                    "_raw_text": "Tech Solutions SARL\nFACTURE N° FA-2024-002\nDate: 20/02/2024\nClient: Entreprise XYZ\nTotal HT: 5000,00€\nTVA 20%: 1000,00€\nTotal TTC: 6000,00€"
                }
            },
            {
                "filename": "invoice_003.txt",
                "type": "invoice",
                "expected": {
                    "invoice_number": "2024-INV-003",
                    "invoice_date": "2024-03-10",
                    "supplier_name": "Services Plus",
                    "client_name": "PME Dupont",
                    "total_ht": 2500.00,
                    "total_tva": 500.00,
                    "total_ttc": 3000.00,
                    "_raw_text": "Services Plus\nFacture 2024-INV-003\nLe 10/03/2024\nDestinataire: PME Dupont\nMontant HT: 2500€\nTVA: 500€\nNet à payer TTC: 3000€"
                }
            },
            {
                "filename": "contract_001.txt",
                "type": "contract",
                "expected": {
                    "contract_type": "service",
                    "signature_date": "2024-01-01",
                    "effective_date": "2024-01-01",
                    "total_amount": 50000.00,
                    "duration": "12 mois",
                    "_raw_text": "CONTRAT DE PRESTATION DE SERVICES\nEntre ACME Corp et Client SA\nSigné le 01/01/2024\nMontant: 50 000€\nDurée: 12 mois"
                }
            },
            {
                "filename": "contract_002.txt",
                "type": "contract",
                "expected": {
                    "contract_type": "employment",
                    "signature_date": "2024-06-15",
                    "effective_date": "2024-07-01",
                    "total_amount": 45000.00,
                    "duration": "CDI",
                    "_raw_text": "CONTRAT DE TRAVAIL\nEmployeur: Société ABC\nSalarié: Jean Martin\nSignature: 15/06/2024\nPrise de poste: 01/07/2024\nSalaire annuel: 45000€\nType: CDI"
                }
            },
        ]
    }

    # Sauvegarder les annotations
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Créer les fichiers texte simulés
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for doc in annotations["documents"]:
        file_path = samples_dir / doc["filename"]
        raw_text = doc["expected"].get("_raw_text", "")
        file_path.write_text(raw_text, encoding="utf-8")

    logger.info(f"Dataset synthétique créé dans {output_dir}")
    logger.info(f"  - {len(annotations['documents'])} documents")
    logger.info(f"  - Annotations: {annotations_path}")


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Évaluation des performances d'extraction"
    )
    parser.add_argument(
        "--data", "-d",
        default="evaluation/test_data",
        help="Répertoire des données de test",
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation/reports",
        help="Répertoire de sortie des rapports",
    )
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Créer un dataset de test synthétique",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparer OCR seul vs OCR + LLM",
    )
    parser.add_argument(
        "--config", "-c",
        help="Fichier de configuration",
    )

    args = parser.parse_args()

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # Créer le dataset si demandé
    if args.create_dataset:
        create_synthetic_dataset(data_dir)
        return

    # Charger les annotations
    annotations_path = data_dir / "annotations.json"
    if not annotations_path.exists():
        logger.error(f"Annotations non trouvées: {annotations_path}")
        logger.info("Utilisez --create-dataset pour créer un dataset de test")
        sys.exit(1)

    annotations = load_annotations(annotations_path)
    samples_dir = data_dir / "samples"

    logger.info(f"Évaluation sur {len(annotations)} documents...")

    # Charger la configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    # Forcer le mode mock pour l'évaluation
    config.vertex_ai.use_mock = True

    # Créer le pipeline
    pipeline = ExtractionPipeline(config)

    # Évaluation principale
    results = evaluate_extraction(pipeline, annotations, samples_dir)

    # Générer le rapport
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = generate_html_report(
        results,
        output_dir / "evaluation_report.html",
    )
    logger.info(f"Rapport généré: {report_path}")

    # Comparaison OCR seul vs OCR + LLM
    if args.compare:
        ocr_only_results = evaluate_ocr_only(annotations)
        comparison_path = generate_comparison_report(
            ocr_only_results,
            results,
            output_dir / "comparison_report.html",
        )
        logger.info(f"Rapport de comparaison: {comparison_path}")

    # Afficher le résumé
    print("\n" + "=" * 50)
    print("RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 50)
    print(f"Documents traités: {results.total_documents}")
    print(f"Taux de succès: {results.success_rate:.1%}")
    print(f"F1-Score macro: {results.macro_f1:.2%}")
    print(f"Temps moyen: {results.avg_processing_time:.0f}ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
