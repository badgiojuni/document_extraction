"""Tests pour le module LLM."""

import pytest
from datetime import date
from decimal import Decimal

from document_extraction.llm import (
    MockLLMClient,
    InvoiceExtractor,
    ContractExtractor,
    DocumentClassifier,
    ExtractionError,
    create_extractor,
)
from document_extraction.models import Invoice, Contract


class TestMockLLMClient:
    """Tests pour MockLLMClient."""

    def test_is_available(self):
        """Le mock est toujours disponible."""
        client = MockLLMClient()
        assert client.is_available() is True

    def test_generate_invoice_response(self):
        """Test génération de réponse facture."""
        client = MockLLMClient()
        prompt = "Extrais les données de cette facture: ```Facture N° FA-2024-001```"
        response = client.generate(prompt)

        assert isinstance(response, str)
        data = client.generate_json(prompt)
        assert "invoice_number" in data
        assert "total_ttc" in data

    def test_generate_contract_response(self):
        """Test génération de réponse contrat."""
        client = MockLLMClient()
        prompt = "Extrais les données de ce contrat: ```Contrat de prestation```"
        response = client.generate(prompt)

        assert isinstance(response, str)
        data = client.generate_json(prompt)
        assert "contract_type" in data
        assert "parties" in data

    def test_classify_invoice(self):
        """Test classification d'une facture."""
        client = MockLLMClient()
        prompt = "Analyse le texte et détermine le type de document. Réponds uniquement avec: ```Facture N° 123 Total TTC: 1000€``` Type:"
        result = client.generate(prompt)
        assert result == "invoice"

    def test_classify_contract(self):
        """Test classification d'un contrat."""
        client = MockLLMClient()
        prompt = "Analyse le texte et détermine le type de document. Réponds uniquement avec: ```Contrat entre les parties signataires``` Type:"
        result = client.generate(prompt)
        assert result == "contract"

    def test_simulate_errors(self):
        """Test simulation d'erreurs."""
        client = MockLLMClient(simulate_errors=True, error_rate=1.0)
        with pytest.raises(Exception):
            client.generate("test")


class TestInvoiceExtractor:
    """Tests pour InvoiceExtractor."""

    @pytest.fixture
    def extractor(self) -> InvoiceExtractor:
        """Crée un extracteur avec client mock."""
        return InvoiceExtractor(MockLLMClient())

    @pytest.fixture
    def sample_invoice_text(self) -> str:
        """Texte OCR d'exemple de facture."""
        return """
        ENTREPRISE ABC SARL
        123 Rue du Commerce
        75001 Paris

        FACTURE N° FA-2024-0042
        Date: 15/01/2024

        Client: Société XYZ

        Description                 Qté    Prix HT    Total HT
        Service consulting          10     100,00     1000,00

        Total HT:    1000,00 €
        TVA 20%:      200,00 €
        Total TTC:   1200,00 €
        """

    def test_extract_invoice(self, extractor, sample_invoice_text):
        """Test extraction d'une facture."""
        result = extractor.extract(sample_invoice_text)

        assert isinstance(result, Invoice)
        assert result.extraction_method == "llm"
        assert result.raw_text == sample_invoice_text
        assert result.confidence_score is not None

    def test_extract_empty_text(self, extractor):
        """Test avec texte vide."""
        with pytest.raises(ExtractionError):
            extractor.extract("")

    def test_extract_whitespace_only(self, extractor):
        """Test avec espaces uniquement."""
        with pytest.raises(ExtractionError):
            extractor.extract("   \n\t  ")

    def test_parse_date(self, extractor):
        """Test parsing de dates."""
        assert extractor._parse_date("2024-01-15") == date(2024, 1, 15)
        assert extractor._parse_date("15/01/2024") == date(2024, 1, 15)
        assert extractor._parse_date(None) is None
        assert extractor._parse_date("invalid") is None

    def test_parse_decimal(self, extractor):
        """Test parsing de montants."""
        assert extractor._parse_decimal(100.50) == Decimal("100.50")
        assert extractor._parse_decimal("1 234,56") == Decimal("1234.56")
        assert extractor._parse_decimal("500€") == Decimal("500")
        assert extractor._parse_decimal(None) is None


class TestContractExtractor:
    """Tests pour ContractExtractor."""

    @pytest.fixture
    def extractor(self) -> ContractExtractor:
        """Crée un extracteur avec client mock."""
        return ContractExtractor(MockLLMClient())

    @pytest.fixture
    def sample_contract_text(self) -> str:
        """Texte OCR d'exemple de contrat."""
        return """
        CONTRAT DE PRESTATION DE SERVICES

        Entre les soussignés:

        La société ABC, représentée par M. Jean Dupont
        ci-après dénommée "Le Prestataire"

        Et

        La société XYZ, représentée par Mme Marie Martin
        ci-après dénommée "Le Client"

        Article 1 - Objet
        Le présent contrat a pour objet...

        Fait à Paris, le 01/02/2024

        Montant: 50 000 €
        Durée: 12 mois
        """

    def test_extract_contract(self, extractor, sample_contract_text):
        """Test extraction d'un contrat."""
        result = extractor.extract(sample_contract_text)

        assert isinstance(result, Contract)
        assert result.extraction_method == "llm"
        assert result.raw_text == sample_contract_text
        assert result.confidence_score is not None
        assert len(result.parties) > 0

    def test_extract_empty_text(self, extractor):
        """Test avec texte vide."""
        with pytest.raises(ExtractionError):
            extractor.extract("")


class TestDocumentClassifier:
    """Tests pour DocumentClassifier."""

    @pytest.fixture
    def classifier(self) -> DocumentClassifier:
        """Crée un classificateur avec client mock."""
        return DocumentClassifier(MockLLMClient())

    def test_classify_invoice(self, classifier):
        """Test classification facture."""
        text = "FACTURE N° 123 Total TTC: 1500€ TVA: 250€ montant à payer"
        result = classifier.classify(text)
        assert result == "invoice"

    def test_classify_contract(self, classifier):
        """Test classification contrat."""
        text = "CONTRAT entre les parties soussignées signataires"
        result = classifier.classify(text)
        assert result == "contract"

    def test_classify_empty(self, classifier):
        """Test avec texte vide."""
        result = classifier.classify("")
        assert result == "unknown"


class TestCreateExtractor:
    """Tests pour la factory create_extractor."""

    def test_create_invoice_extractor(self):
        """Test création extracteur facture."""
        client = MockLLMClient()
        extractor = create_extractor("invoice", client)
        assert isinstance(extractor, InvoiceExtractor)

    def test_create_contract_extractor(self):
        """Test création extracteur contrat."""
        client = MockLLMClient()
        extractor = create_extractor("contract", client)
        assert isinstance(extractor, ContractExtractor)

    def test_create_invalid_type(self):
        """Test avec type invalide."""
        client = MockLLMClient()
        with pytest.raises(ValueError):
            create_extractor("invalid", client)
