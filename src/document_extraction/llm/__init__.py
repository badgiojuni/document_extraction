"""Module LLM pour l'extraction structurée."""

from .vertex_client import VertexAIClient, BaseLLMClient, LLMClientError
from .mock_client import MockLLMClient
from .extractors import (
    InvoiceExtractor,
    ContractExtractor,
    DocumentClassifier,
    ExtractionError,
    create_extractor,
)
from .prompts import (
    get_invoice_prompt,
    get_contract_prompt,
    get_classification_prompt,
)

__all__ = [
    # Clients
    "VertexAIClient",
    "BaseLLMClient",
    "MockLLMClient",
    "LLMClientError",
    # Extracteurs
    "InvoiceExtractor",
    "ContractExtractor",
    "DocumentClassifier",
    "ExtractionError",
    "create_extractor",
    # Prompts
    "get_invoice_prompt",
    "get_contract_prompt",
    "get_classification_prompt",
]
