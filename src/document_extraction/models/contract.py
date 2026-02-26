"""Modèle de données pour les contrats."""

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContractType(str, Enum):
    """Types de contrats supportés."""

    SERVICE = "service"
    EMPLOYMENT = "employment"
    LEASE = "lease"
    SALE = "sale"
    NDA = "nda"
    PARTNERSHIP = "partnership"
    OTHER = "other"


class Party(BaseModel):
    """Partie prenante d'un contrat."""

    name: str = Field(..., description="Nom de la partie")
    role: Optional[str] = Field(None, description="Rôle (vendeur, acheteur, employeur, etc.)")
    address: Optional[str] = Field(None, description="Adresse")
    siret: Optional[str] = Field(None, description="Numéro SIRET")
    representative: Optional[str] = Field(None, description="Représentant légal")


class Clause(BaseModel):
    """Clause importante d'un contrat."""

    title: str = Field(..., description="Titre ou type de clause")
    content: str = Field(..., description="Contenu résumé de la clause")
    importance: Optional[str] = Field(
        None, description="Niveau d'importance (high, medium, low)"
    )


class Contract(BaseModel):
    """Modèle de données pour un contrat extrait."""

    # Type et identification
    contract_type: Optional[ContractType] = Field(None, description="Type de contrat")
    contract_number: Optional[str] = Field(None, description="Numéro/référence du contrat")
    title: Optional[str] = Field(None, description="Titre du contrat")

    # Parties
    parties: list[Party] = Field(default_factory=list, description="Parties prenantes")

    # Dates
    signature_date: Optional[date] = Field(None, description="Date de signature")
    effective_date: Optional[date] = Field(None, description="Date d'entrée en vigueur")
    end_date: Optional[date] = Field(None, description="Date de fin")
    duration: Optional[str] = Field(None, description="Durée du contrat")

    # Aspects financiers
    total_amount: Optional[Decimal] = Field(None, description="Montant total")
    payment_terms: Optional[str] = Field(None, description="Conditions de paiement")
    currency: Optional[str] = Field(default="EUR", description="Devise")

    # Clauses
    key_clauses: list[Clause] = Field(
        default_factory=list, description="Clauses importantes identifiées"
    )

    # Conditions
    termination_conditions: Optional[str] = Field(
        None, description="Conditions de résiliation"
    )
    renewal_terms: Optional[str] = Field(None, description="Conditions de renouvellement")

    # Signatures
    signatures: list[str] = Field(
        default_factory=list, description="Signataires identifiés"
    )

    # Métadonnées d'extraction
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Score de confiance de l'extraction"
    )
    raw_text: Optional[str] = Field(None, description="Texte brut extrait par OCR")
    extraction_method: Optional[str] = Field(None, description="Méthode d'extraction utilisée")

    class Config:
        """Configuration Pydantic."""

        json_encoders = {
            Decimal: lambda v: float(v) if v else None,
            date: lambda v: v.isoformat() if v else None,
        }

    def to_dict_display(self) -> dict:
        """Convertit en dictionnaire pour affichage (sans raw_text)."""
        data = self.model_dump(exclude={"raw_text"})
        if data.get("total_amount"):
            data["total_amount"] = float(data["total_amount"])
        if data.get("contract_type"):
            data["contract_type"] = data["contract_type"].value
        return data

    def get_extracted_fields(self) -> dict[str, bool]:
        """Retourne les champs extraits avec succès."""
        fields = {}
        exclude = {"parties", "key_clauses", "signatures", "raw_text",
                   "confidence_score", "extraction_method"}
        for field_name in self.model_fields:
            if field_name in exclude:
                continue
            value = getattr(self, field_name)
            fields[field_name] = value is not None
        # Ajouter les champs liste
        fields["parties"] = len(self.parties) > 0
        fields["key_clauses"] = len(self.key_clauses) > 0
        fields["signatures"] = len(self.signatures) > 0
        return fields
