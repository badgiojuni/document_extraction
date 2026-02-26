"""Extracteurs de données utilisant les LLM."""

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Optional, Union

from ..models.invoice import Invoice, LineItem
from ..models.contract import Contract, ContractType, Party, Clause
from .vertex_client import BaseLLMClient, LLMClientError
from .prompts import get_invoice_prompt, get_contract_prompt, get_classification_prompt

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Erreur lors de l'extraction de données."""

    pass


class BaseExtractor:
    """Classe de base pour les extracteurs."""

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialise l'extracteur.

        Args:
            llm_client: Client LLM à utiliser
        """
        self.llm_client = llm_client

    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse une date depuis différents formats."""
        if value is None:
            return None

        if isinstance(value, date):
            return value

        if isinstance(value, str):
            try:
                # Format ISO
                return date.fromisoformat(value)
            except ValueError:
                pass

            # Format DD/MM/YYYY
            try:
                parts = value.replace("-", "/").split("/")
                if len(parts) == 3:
                    if len(parts[0]) == 4:  # YYYY/MM/DD
                        return date(int(parts[0]), int(parts[1]), int(parts[2]))
                    else:  # DD/MM/YYYY
                        return date(int(parts[2]), int(parts[1]), int(parts[0]))
            except (ValueError, IndexError):
                pass

        logger.warning(f"Impossible de parser la date: {value}")
        return None

    def _parse_decimal(self, value: Any) -> Optional[Decimal]:
        """Parse un montant en Decimal."""
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                return Decimal(str(value))
            if isinstance(value, str):
                # Nettoyer la chaîne
                cleaned = value.replace(" ", "").replace(",", ".").replace("€", "")
                return Decimal(cleaned)
        except Exception:
            logger.warning(f"Impossible de parser le montant: {value}")

        return None


class InvoiceExtractor(BaseExtractor):
    """Extracteur de données de factures."""

    def extract(self, ocr_text: str) -> Invoice:
        """
        Extrait les données d'une facture à partir du texte OCR.

        Args:
            ocr_text: Texte extrait par OCR

        Returns:
            Invoice avec les données extraites

        Raises:
            ExtractionError: En cas d'erreur d'extraction
        """
        if not ocr_text or not ocr_text.strip():
            raise ExtractionError("Texte OCR vide")

        logger.info("Extraction des données de facture...")

        try:
            prompt = get_invoice_prompt(ocr_text)
            data = self.llm_client.generate_json(prompt)

            return self._parse_invoice_data(data, ocr_text)

        except LLMClientError as e:
            raise ExtractionError(f"Erreur LLM: {e}")
        except Exception as e:
            logger.error(f"Erreur d'extraction: {e}")
            raise ExtractionError(f"Erreur lors de l'extraction: {e}")

    def _parse_invoice_data(self, data: dict[str, Any], ocr_text: str) -> Invoice:
        """Parse les données JSON en objet Invoice."""
        # Parsing des lignes de facture
        line_items = []
        for item_data in data.get("line_items", []):
            line_items.append(
                LineItem(
                    description=item_data.get("description", ""),
                    quantity=item_data.get("quantity"),
                    unit_price=self._parse_decimal(item_data.get("unit_price")),
                    total_ht=self._parse_decimal(item_data.get("total_ht")),
                    tva_rate=item_data.get("tva_rate"),
                )
            )

        return Invoice(
            invoice_number=data.get("invoice_number"),
            invoice_date=self._parse_date(data.get("invoice_date")),
            due_date=self._parse_date(data.get("due_date")),
            supplier_name=data.get("supplier_name"),
            supplier_address=data.get("supplier_address"),
            supplier_siret=data.get("supplier_siret"),
            supplier_vat_number=data.get("supplier_vat_number"),
            client_name=data.get("client_name"),
            client_address=data.get("client_address"),
            client_siret=data.get("client_siret"),
            total_ht=self._parse_decimal(data.get("total_ht")),
            total_tva=self._parse_decimal(data.get("total_tva")),
            total_ttc=self._parse_decimal(data.get("total_ttc")),
            line_items=line_items,
            confidence_score=data.get("confidence_score"),
            raw_text=ocr_text,
            extraction_method="llm",
        )


class ContractExtractor(BaseExtractor):
    """Extracteur de données de contrats."""

    def extract(self, ocr_text: str) -> Contract:
        """
        Extrait les données d'un contrat à partir du texte OCR.

        Args:
            ocr_text: Texte extrait par OCR

        Returns:
            Contract avec les données extraites

        Raises:
            ExtractionError: En cas d'erreur d'extraction
        """
        if not ocr_text or not ocr_text.strip():
            raise ExtractionError("Texte OCR vide")

        logger.info("Extraction des données de contrat...")

        try:
            prompt = get_contract_prompt(ocr_text)
            data = self.llm_client.generate_json(prompt)

            return self._parse_contract_data(data, ocr_text)

        except LLMClientError as e:
            raise ExtractionError(f"Erreur LLM: {e}")
        except Exception as e:
            logger.error(f"Erreur d'extraction: {e}")
            raise ExtractionError(f"Erreur lors de l'extraction: {e}")

    def _parse_contract_data(self, data: dict[str, Any], ocr_text: str) -> Contract:
        """Parse les données JSON en objet Contract."""
        # Parsing du type de contrat
        contract_type = None
        type_str = data.get("contract_type")
        if type_str:
            try:
                contract_type = ContractType(type_str.lower())
            except ValueError:
                contract_type = ContractType.OTHER

        # Parsing des parties
        parties = []
        for party_data in data.get("parties", []):
            parties.append(
                Party(
                    name=party_data.get("name", ""),
                    role=party_data.get("role"),
                    address=party_data.get("address"),
                    siret=party_data.get("siret"),
                    representative=party_data.get("representative"),
                )
            )

        # Parsing des clauses
        clauses = []
        for clause_data in data.get("key_clauses", []):
            clauses.append(
                Clause(
                    title=clause_data.get("title", ""),
                    content=clause_data.get("content", ""),
                    importance=clause_data.get("importance"),
                )
            )

        return Contract(
            contract_type=contract_type,
            contract_number=data.get("contract_number"),
            title=data.get("title"),
            parties=parties,
            signature_date=self._parse_date(data.get("signature_date")),
            effective_date=self._parse_date(data.get("effective_date")),
            end_date=self._parse_date(data.get("end_date")),
            duration=data.get("duration"),
            total_amount=self._parse_decimal(data.get("total_amount")),
            payment_terms=data.get("payment_terms"),
            currency=data.get("currency", "EUR"),
            key_clauses=clauses,
            termination_conditions=data.get("termination_conditions"),
            renewal_terms=data.get("renewal_terms"),
            signatures=data.get("signatures", []),
            confidence_score=data.get("confidence_score"),
            raw_text=ocr_text,
            extraction_method="llm",
        )


class DocumentClassifier:
    """Classifie le type de document."""

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialise le classificateur.

        Args:
            llm_client: Client LLM à utiliser
        """
        self.llm_client = llm_client

    def classify(self, ocr_text: str) -> str:
        """
        Détermine le type de document.

        Args:
            ocr_text: Texte extrait par OCR

        Returns:
            Type de document: "invoice", "contract", ou "unknown"
        """
        if not ocr_text or not ocr_text.strip():
            return "unknown"

        try:
            prompt = get_classification_prompt(ocr_text)
            result = self.llm_client.generate(prompt).strip().lower()

            if result in ["invoice", "contract"]:
                return result
            return "unknown"

        except Exception as e:
            logger.warning(f"Erreur de classification: {e}")
            return "unknown"


def create_extractor(
    document_type: str,
    llm_client: BaseLLMClient,
) -> Union[InvoiceExtractor, ContractExtractor]:
    """
    Factory pour créer l'extracteur approprié.

    Args:
        document_type: Type de document ("invoice" ou "contract")
        llm_client: Client LLM à utiliser

    Returns:
        Instance de l'extracteur approprié

    Raises:
        ValueError: Si le type de document n'est pas supporté
    """
    if document_type == "invoice":
        return InvoiceExtractor(llm_client)
    elif document_type == "contract":
        return ContractExtractor(llm_client)
    else:
        raise ValueError(f"Type de document non supporté: {document_type}")
