"""Composant d'affichage des résultats."""

import io
import json
import streamlit as st
import pandas as pd
from typing import Any, Union

from document_extraction.models import Invoice, Contract


def render_results_display(
    data: Union[Invoice, Contract],
    document_type: str,
) -> None:
    """
    Affiche les résultats de l'extraction.

    Args:
        data: Données extraites (Invoice ou Contract)
        document_type: Type de document
    """
    st.markdown("### 📊 Données extraites")

    # Score de confiance global
    if data.confidence_score:
        _render_confidence_badge(data.confidence_score)

    # Affichage selon le type
    if document_type == "invoice":
        _render_invoice_data(data)
    else:
        _render_contract_data(data)

    # Export
    st.markdown("---")
    _render_export_buttons(data, document_type)


def _render_confidence_badge(score: float) -> None:
    """Affiche le badge de confiance."""
    if score >= 0.8:
        color = "green"
        label = "Haute confiance"
    elif score >= 0.5:
        color = "orange"
        label = "Confiance moyenne"
    else:
        color = "red"
        label = "Faible confiance"

    st.markdown(
        f"**Confiance d'extraction:** "
        f":{color}[{label} ({score:.0%})]"
    )


def _render_invoice_data(invoice: Invoice) -> None:
    """Affiche les données d'une facture."""
    # Informations principales
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏢 Informations générales")
        _display_field("Numéro de facture", invoice.invoice_number)
        _display_field("Date de facture", invoice.invoice_date)
        _display_field("Date d'échéance", invoice.due_date)

    with col2:
        st.markdown("#### 💰 Montants")
        _display_field("Total HT", _format_amount(invoice.total_ht))
        _display_field("TVA", _format_amount(invoice.total_tva))
        _display_field("Total TTC", _format_amount(invoice.total_ttc), highlight=True)

    # Fournisseur et client
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📤 Fournisseur")
        _display_field("Nom", invoice.supplier_name)
        _display_field("Adresse", invoice.supplier_address)
        _display_field("SIRET", invoice.supplier_siret)
        _display_field("TVA Intra.", invoice.supplier_vat_number)

    with col2:
        st.markdown("#### 📥 Client")
        _display_field("Nom", invoice.client_name)
        _display_field("Adresse", invoice.client_address)
        _display_field("SIRET", invoice.client_siret)

    # Lignes de facture
    if invoice.line_items:
        st.markdown("#### 📋 Lignes de facture")
        df = pd.DataFrame([
            {
                "Description": item.description,
                "Quantité": item.quantity,
                "Prix unitaire": _format_amount(item.unit_price),
                "Total HT": _format_amount(item.total_ht),
                "TVA %": item.tva_rate,
            }
            for item in invoice.line_items
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_contract_data(contract: Contract) -> None:
    """Affiche les données d'un contrat."""
    # Informations générales
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Informations générales")
        _display_field("Type", contract.contract_type.value if contract.contract_type else None)
        _display_field("Référence", contract.contract_number)
        _display_field("Titre", contract.title)

    with col2:
        st.markdown("#### 📅 Dates")
        _display_field("Date de signature", contract.signature_date)
        _display_field("Date d'effet", contract.effective_date)
        _display_field("Date de fin", contract.end_date)
        _display_field("Durée", contract.duration)

    # Aspects financiers
    st.markdown("#### 💰 Aspects financiers")
    col1, col2 = st.columns(2)
    with col1:
        _display_field("Montant total", _format_amount(contract.total_amount, contract.currency))
    with col2:
        _display_field("Conditions de paiement", contract.payment_terms)

    # Parties
    if contract.parties:
        st.markdown("#### 👥 Parties")
        for i, party in enumerate(contract.parties, 1):
            with st.expander(f"Partie {i}: {party.name}", expanded=True):
                _display_field("Rôle", party.role)
                _display_field("Adresse", party.address)
                _display_field("SIRET", party.siret)
                _display_field("Représentant", party.representative)

    # Clauses importantes
    if contract.key_clauses:
        st.markdown("#### ⚖️ Clauses importantes")
        for clause in contract.key_clauses:
            importance_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                clause.importance, "⚪"
            )
            with st.expander(f"{importance_icon} {clause.title}"):
                st.write(clause.content)

    # Autres informations
    col1, col2 = st.columns(2)
    with col1:
        _display_field("Conditions de résiliation", contract.termination_conditions)
    with col2:
        _display_field("Conditions de renouvellement", contract.renewal_terms)

    # Signataires
    if contract.signatures:
        st.markdown("#### ✍️ Signataires")
        st.write(", ".join(contract.signatures))


def _display_field(label: str, value: Any, highlight: bool = False) -> None:
    """Affiche un champ avec son label."""
    if value is not None:
        if highlight:
            st.markdown(f"**{label}:** :green[**{value}**]")
        else:
            st.markdown(f"**{label}:** {value}")
    else:
        st.markdown(f"**{label}:** :gray[Non détecté]")


def _format_amount(amount: Any, currency: str = "EUR") -> str | None:
    """Formate un montant."""
    if amount is None:
        return None
    symbol = {"EUR": "€", "USD": "$", "GBP": "£"}.get(currency, currency)
    return f"{float(amount):,.2f} {symbol}".replace(",", " ")


def _render_export_buttons(
    data: Union[Invoice, Contract],
    document_type: str,
) -> None:
    """Affiche les boutons d'export."""
    st.markdown("### 💾 Export des données")

    col1, col2 = st.columns(2)

    # Export JSON
    with col1:
        json_data = json.dumps(
            data.to_dict_display(),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        st.download_button(
            label="📥 Télécharger JSON",
            data=json_data,
            file_name=f"{document_type}_extracted.json",
            mime="application/json",
        )

    # Export CSV
    with col2:
        csv_data = _convert_to_csv(data, document_type)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv_data,
            file_name=f"{document_type}_extracted.csv",
            mime="text/csv",
        )


def _convert_to_csv(data: Union[Invoice, Contract], document_type: str) -> str:
    """Convertit les données en CSV."""
    flat_data = {}
    data_dict = data.to_dict_display()

    for key, value in data_dict.items():
        if isinstance(value, list):
            # Ignorer les listes complexes pour le CSV simple
            flat_data[key] = len(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_data[f"{key}_{sub_key}"] = sub_value
        else:
            flat_data[key] = value

    df = pd.DataFrame([flat_data])
    return df.to_csv(index=False)


def render_error_message(error: str) -> None:
    """Affiche un message d'erreur."""
    st.error(f"❌ Erreur lors de l'extraction: {error}")

    st.markdown("""
    **Causes possibles:**
    - Document illisible ou de mauvaise qualité
    - Format de document non supporté
    - Problème de connexion au service LLM

    **Solutions:**
    - Vérifiez la qualité du document
    - Essayez avec un autre fichier
    - Contactez le support si le problème persiste
    """)
