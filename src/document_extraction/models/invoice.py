"""Modèle de données pour les factures."""

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """Ligne de détail d'une facture."""

    description: str = Field(..., description="Description du produit/service")
    quantity: Optional[float] = Field(None, description="Quantité")
    unit_price: Optional[Decimal] = Field(None, description="Prix unitaire HT")
    total_ht: Optional[Decimal] = Field(None, description="Total HT de la ligne")
    tva_rate: Optional[float] = Field(None, description="Taux de TVA en %")


class Invoice(BaseModel):
    """Modèle de données pour une facture extraite."""

    # Identifiants
    invoice_number: Optional[str] = Field(None, description="Numéro de facture")
    invoice_date: Optional[date] = Field(None, description="Date de facturation")
    due_date: Optional[date] = Field(None, description="Date d'échéance")

    # Fournisseur
    supplier_name: Optional[str] = Field(None, description="Nom du fournisseur")
    supplier_address: Optional[str] = Field(None, description="Adresse du fournisseur")
    supplier_siret: Optional[str] = Field(None, description="SIRET du fournisseur")
    supplier_vat_number: Optional[str] = Field(None, description="Numéro TVA fournisseur")

    # Client
    client_name: Optional[str] = Field(None, description="Nom du client")
    client_address: Optional[str] = Field(None, description="Adresse du client")
    client_siret: Optional[str] = Field(None, description="SIRET du client")

    # Montants
    total_ht: Optional[Decimal] = Field(None, description="Total HT")
    total_tva: Optional[Decimal] = Field(None, description="Total TVA")
    total_ttc: Optional[Decimal] = Field(None, description="Total TTC")

    # Lignes de détail
    line_items: list[LineItem] = Field(default_factory=list, description="Lignes de facture")

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
        # Convertir les Decimal en float pour l'affichage
        for key in ["total_ht", "total_tva", "total_ttc"]:
            if data.get(key):
                data[key] = float(data[key])
        for item in data.get("line_items", []):
            for k in ["unit_price", "total_ht"]:
                if item.get(k):
                    item[k] = float(item[k])
        return data

    def get_extracted_fields(self) -> dict[str, bool]:
        """Retourne les champs extraits avec succès."""
        fields = {}
        for field_name in self.model_fields:
            if field_name in ["line_items", "raw_text", "confidence_score", "extraction_method"]:
                continue
            value = getattr(self, field_name)
            fields[field_name] = value is not None
        return fields
