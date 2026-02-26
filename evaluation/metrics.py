"""Calcul des métriques d'évaluation."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FieldMetrics:
    """Métriques pour un champ spécifique."""

    field_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    total_samples: int = 0

    @property
    def precision(self) -> float:
        """Calcule la précision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calcule le recall."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calcule le F1-score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Calcule l'accuracy (correspondances exactes)."""
        if self.total_samples == 0:
            return 0.0
        return self.exact_matches / self.total_samples


@dataclass
class EvaluationResults:
    """Résultats complets de l'évaluation."""

    field_metrics: dict[str, FieldMetrics] = field(default_factory=dict)
    processing_times: list[float] = field(default_factory=list)
    total_documents: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0

    @property
    def success_rate(self) -> float:
        """Taux de succès des extractions."""
        if self.total_documents == 0:
            return 0.0
        return self.successful_extractions / self.total_documents

    @property
    def avg_processing_time(self) -> float:
        """Temps de traitement moyen en ms."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    @property
    def macro_precision(self) -> float:
        """Précision macro-moyennée."""
        if not self.field_metrics:
            return 0.0
        precisions = [m.precision for m in self.field_metrics.values()]
        return sum(precisions) / len(precisions)

    @property
    def macro_recall(self) -> float:
        """Recall macro-moyenné."""
        if not self.field_metrics:
            return 0.0
        recalls = [m.recall for m in self.field_metrics.values()]
        return sum(recalls) / len(recalls)

    @property
    def macro_f1(self) -> float:
        """F1-score macro-moyenné."""
        if not self.field_metrics:
            return 0.0
        f1s = [m.f1_score for m in self.field_metrics.values()]
        return sum(f1s) / len(f1s)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "summary": {
                "total_documents": self.total_documents,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
                "success_rate": self.success_rate,
                "avg_processing_time_ms": self.avg_processing_time,
                "macro_precision": self.macro_precision,
                "macro_recall": self.macro_recall,
                "macro_f1": self.macro_f1,
            },
            "fields": {
                name: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "accuracy": m.accuracy,
                    "true_positives": m.true_positives,
                    "false_positives": m.false_positives,
                    "false_negatives": m.false_negatives,
                }
                for name, m in self.field_metrics.items()
            },
        }


def normalize_value(value: Any) -> str:
    """Normalise une valeur pour la comparaison."""
    if value is None:
        return ""

    value_str = str(value).lower().strip()

    # Normaliser les espaces
    value_str = " ".join(value_str.split())

    # Supprimer la ponctuation courante
    for char in [".", ",", ";", ":", "-", "_", "'"]:
        value_str = value_str.replace(char, " ")

    return " ".join(value_str.split())


def compare_values(
    predicted: Any,
    expected: Any,
    field_type: str = "string",
) -> tuple[bool, bool]:
    """
    Compare une valeur prédite avec la valeur attendue.

    Args:
        predicted: Valeur prédite
        expected: Valeur attendue (ground truth)
        field_type: Type de champ (string, number, date)

    Returns:
        Tuple (exact_match, partial_match)
    """
    if expected is None:
        # Si pas de valeur attendue, ne pas évaluer
        return False, False

    if predicted is None:
        # Valeur attendue mais non prédite = faux négatif
        return False, False

    # Normalisation
    pred_norm = normalize_value(predicted)
    exp_norm = normalize_value(expected)

    # Correspondance exacte
    exact = pred_norm == exp_norm

    # Correspondance partielle
    partial = False
    if not exact:
        # Vérifier si l'un contient l'autre
        if pred_norm in exp_norm or exp_norm in pred_norm:
            partial = True
        # Pour les nombres, vérifier si proche
        elif field_type == "number":
            try:
                pred_num = float(predicted)
                exp_num = float(expected)
                # Tolérance de 1%
                if abs(pred_num - exp_num) / max(abs(exp_num), 1) < 0.01:
                    partial = True
            except (ValueError, TypeError):
                pass

    return exact, partial


def calculate_field_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    fields: list[str],
) -> dict[str, FieldMetrics]:
    """
    Calcule les métriques par champ.

    Args:
        predictions: Liste des prédictions
        ground_truth: Liste des vérités terrain
        fields: Liste des champs à évaluer

    Returns:
        Dictionnaire des métriques par champ
    """
    metrics = {field: FieldMetrics(field_name=field) for field in fields}

    for pred, truth in zip(predictions, ground_truth):
        for field in fields:
            pred_value = pred.get(field)
            truth_value = truth.get(field)

            metrics[field].total_samples += 1

            # Déterminer le type de champ
            field_type = "string"
            if any(x in field for x in ["total", "amount", "price", "tva"]):
                field_type = "number"
            elif "date" in field:
                field_type = "date"

            exact, partial = compare_values(pred_value, truth_value, field_type)

            if truth_value is not None:
                if pred_value is not None:
                    if exact:
                        metrics[field].true_positives += 1
                        metrics[field].exact_matches += 1
                    elif partial:
                        metrics[field].true_positives += 1
                        metrics[field].partial_matches += 1
                    else:
                        metrics[field].false_positives += 1
                else:
                    metrics[field].false_negatives += 1

    return metrics


def build_confusion_matrix(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    """
    Construit une matrice de confusion.

    Args:
        predictions: Prédictions
        ground_truth: Vérités terrain
        labels: Labels possibles

    Returns:
        Matrice de confusion sous forme de dictionnaire
    """
    matrix = {true: {pred: 0 for pred in labels} for true in labels}

    for pred, truth in zip(predictions, ground_truth):
        if truth in labels and pred in labels:
            matrix[truth][pred] += 1

    return matrix
