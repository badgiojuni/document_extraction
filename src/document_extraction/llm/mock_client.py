"""Client LLM mock pour les tests sans credentials."""

import json
import logging
import random
import re
from datetime import date, timedelta
from typing import Any

from .vertex_client import BaseLLMClient

logger = logging.getLogger(__name__)


class MockLLMClient(BaseLLMClient):
    """
    Client LLM simulé pour les tests et démos.

    Génère des réponses plausibles basées sur l'analyse du texte OCR
    sans faire appel à une API externe.
    """

    def __init__(self, simulate_errors: bool = False, error_rate: float = 0.1):
        """
        Initialise le client mock.

        Args:
            simulate_errors: Activer la simulation d'erreurs aléatoires
            error_rate: Taux d'erreur (0-1) si simulation activée
        """
        self.simulate_errors = simulate_errors
        self.error_rate = error_rate
        logger.info("MockLLMClient initialisé (mode simulation)")

    def is_available(self) -> bool:
        """Le mock est toujours disponible."""
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse simulée.

        Args:
            prompt: Le prompt (utilisé pour déterminer le type de réponse)

        Returns:
            Réponse simulée
        """
        if self.simulate_errors and random.random() < self.error_rate:
            raise Exception("Erreur simulée du LLM")

        prompt_lower = prompt.lower()

        # Détection du type de requête - classification en premier
        if "type de document" in prompt_lower or "type:" in prompt_lower or "réponds uniquement" in prompt_lower:
            return self._classify_document(prompt)
        elif "facture" in prompt_lower and "extrais" in prompt_lower:
            return self._generate_invoice_response(prompt)
        elif "contrat" in prompt_lower and "extrais" in prompt_lower:
            return self._generate_contract_response(prompt)
        elif "invoice" in prompt_lower or "facture" in prompt_lower:
            return self._generate_invoice_response(prompt)
        elif "contract" in prompt_lower or "contrat" in prompt_lower:
            return self._generate_contract_response(prompt)
        else:
            return "{}"

    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Génère une réponse JSON simulée.

        Args:
            prompt: Le prompt

        Returns:
            Dictionnaire simulé
        """
        response = self.generate(prompt, **kwargs)
        return json.loads(response)

    def _extract_text_from_prompt(self, prompt: str) -> str:
        """Extrait le texte OCR du prompt."""
        # Chercher le texte entre les balises ```
        match = re.search(r"```\s*(.*?)\s*```", prompt, re.DOTALL)
        if match:
            return match.group(1)
        return prompt

    def _find_pattern(self, text: str, patterns: list[str]) -> str | None:
        """Cherche un pattern dans le texte."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None

    def _find_amount(self, text: str, keywords: list[str]) -> float | None:
        """Cherche un montant associé à des mots-clés."""
        for keyword in keywords:
            # Pattern: keyword suivi d'un montant
            pattern = rf"{keyword}\s*:?\s*([\d\s]+[,.]?\d*)\s*€?"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(" ", "").replace(",", ".")
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return None

    def _find_date(self, text: str) -> str | None:
        """Cherche une date dans le texte."""
        # Format DD/MM/YYYY ou DD-MM-YYYY
        patterns = [
            r"(\d{2})[/\-](\d{2})[/\-](\d{4})",
            r"(\d{2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})",
        ]

        match = re.search(patterns[0], text)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month}-{day}"

        return None

    def _generate_invoice_response(self, prompt: str) -> str:
        """Génère une réponse simulée pour une facture."""
        text = self._extract_text_from_prompt(prompt)

        # Extraction basique des informations
        invoice_number = self._find_pattern(
            text,
            [r"facture\s*n[°o]?\s*:?\s*(\w+)", r"n[°o]\s*:?\s*(\w+)", r"FA[-_]?(\d+)"],
        )

        invoice_date = self._find_date(text)

        supplier_name = self._find_pattern(
            text, [r"^([A-Z][A-Za-z\s&]+(?:SARL|SAS|SA|EURL)?)", r"Société\s+([A-Za-z\s]+)"]
        )

        total_ttc = self._find_amount(text, ["total ttc", "ttc", "net à payer", "total"])
        total_ht = self._find_amount(text, ["total ht", "ht", "hors taxes"])
        total_tva = self._find_amount(text, ["tva", "total tva"])

        # Si on a TTC et HT, calculer TVA
        if total_ttc and total_ht and not total_tva:
            total_tva = round(total_ttc - total_ht, 2)

        # Générer une confiance basée sur ce qu'on a trouvé
        found_fields = sum(
            [
                invoice_number is not None,
                invoice_date is not None,
                total_ttc is not None,
                supplier_name is not None,
            ]
        )
        confidence = min(0.5 + (found_fields * 0.1), 0.95)

        response = {
            "invoice_number": invoice_number or f"MOCK-{random.randint(1000, 9999)}",
            "invoice_date": invoice_date or date.today().isoformat(),
            "due_date": (date.today() + timedelta(days=30)).isoformat() if invoice_date else None,
            "supplier_name": supplier_name or "Fournisseur Simulé SARL",
            "supplier_address": "123 Rue de la Simulation, 75001 Paris",
            "supplier_siret": "12345678901234",
            "supplier_vat_number": "FR12345678901",
            "client_name": "Client Exemple",
            "client_address": "456 Avenue du Test, 69001 Lyon",
            "client_siret": None,
            "total_ht": total_ht or round(random.uniform(100, 5000), 2),
            "total_tva": total_tva,
            "total_ttc": total_ttc or round(random.uniform(120, 6000), 2),
            "line_items": [
                {
                    "description": "Service de démonstration",
                    "quantity": 1,
                    "unit_price": total_ht or 1000.0,
                    "total_ht": total_ht or 1000.0,
                    "tva_rate": 20.0,
                }
            ],
            "confidence_score": confidence,
        }

        return json.dumps(response, ensure_ascii=False)

    def _generate_contract_response(self, prompt: str) -> str:
        """Génère une réponse simulée pour un contrat."""
        text = self._extract_text_from_prompt(prompt)

        # Détection du type de contrat
        contract_type = "other"
        if any(w in text.lower() for w in ["emploi", "travail", "salarié", "employeur"]):
            contract_type = "employment"
        elif any(w in text.lower() for w in ["prestation", "service", "mission"]):
            contract_type = "service"
        elif any(w in text.lower() for w in ["bail", "location", "loyer", "locataire"]):
            contract_type = "lease"
        elif any(w in text.lower() for w in ["vente", "achat", "acquéreur"]):
            contract_type = "sale"
        elif any(w in text.lower() for w in ["confidentialité", "nda", "non-disclosure"]):
            contract_type = "nda"

        signature_date = self._find_date(text)
        total_amount = self._find_amount(text, ["montant", "prix", "rémunération", "loyer"])

        confidence = 0.75 if signature_date or total_amount else 0.5

        response = {
            "contract_type": contract_type,
            "contract_number": f"CTR-{random.randint(2024, 2025)}-{random.randint(100, 999)}",
            "title": f"Contrat de {contract_type}",
            "parties": [
                {
                    "name": "Société ABC",
                    "role": "Prestataire",
                    "address": "10 Rue du Commerce, 75001 Paris",
                    "siret": "98765432109876",
                    "representative": "Jean Dupont",
                },
                {
                    "name": "Entreprise XYZ",
                    "role": "Client",
                    "address": "20 Avenue des Affaires, 69002 Lyon",
                    "siret": "12345678901234",
                    "representative": "Marie Martin",
                },
            ],
            "signature_date": signature_date or date.today().isoformat(),
            "effective_date": signature_date or date.today().isoformat(),
            "end_date": (date.today() + timedelta(days=365)).isoformat(),
            "duration": "12 mois",
            "total_amount": total_amount or round(random.uniform(10000, 100000), 2),
            "payment_terms": "30 jours fin de mois",
            "currency": "EUR",
            "key_clauses": [
                {
                    "title": "Confidentialité",
                    "content": "Les parties s'engagent à maintenir la confidentialité des informations échangées",
                    "importance": "high",
                },
                {
                    "title": "Résiliation",
                    "content": "Le contrat peut être résilié avec un préavis de 3 mois",
                    "importance": "high",
                },
            ],
            "termination_conditions": "Préavis de 3 mois",
            "renewal_terms": "Reconduction tacite",
            "signatures": ["Jean Dupont", "Marie Martin"],
            "confidence_score": confidence,
        }

        return json.dumps(response, ensure_ascii=False)

    def _classify_document(self, prompt: str) -> str:
        """Classifie le type de document."""
        text = self._extract_text_from_prompt(prompt).lower()

        if any(w in text for w in ["facture", "invoice", "ttc", "tva", "montant à payer"]):
            return "invoice"
        elif any(w in text for w in ["contrat", "contract", "parties", "signataires"]):
            return "contract"
        else:
            return "unknown"
