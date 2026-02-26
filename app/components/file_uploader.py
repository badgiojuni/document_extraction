"""Composant d'upload de fichiers."""

import streamlit as st
from typing import Optional, Tuple
from pathlib import Path


def render_file_uploader(
    allowed_extensions: list[str],
    max_size_mb: int = 10,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Affiche le composant d'upload de fichiers.

    Args:
        allowed_extensions: Extensions autorisées
        max_size_mb: Taille maximale en MB

    Returns:
        Tuple (contenu en bytes, nom du fichier) ou (None, None)
    """
    st.markdown("### 📄 Upload de document")

    # Format des extensions pour Streamlit
    extensions_display = ", ".join([f".{ext}" for ext in allowed_extensions])

    uploaded_file = st.file_uploader(
        f"Déposez votre document ({extensions_display})",
        type=allowed_extensions,
        help=f"Taille maximale: {max_size_mb} MB",
    )

    if uploaded_file is not None:
        # Vérification de la taille
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        if file_size_mb > max_size_mb:
            st.error(f"❌ Fichier trop volumineux ({file_size_mb:.1f} MB > {max_size_mb} MB)")
            return None, None

        # Afficher les infos du fichier
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ **{uploaded_file.name}**")
        with col2:
            st.info(f"📊 {file_size_mb:.2f} MB")

        return uploaded_file.getvalue(), uploaded_file.name

    return None, None


def render_document_type_selector() -> Optional[str]:
    """
    Affiche le sélecteur de type de document.

    Returns:
        Type de document sélectionné ou None pour auto-détection
    """
    st.markdown("### 📋 Type de document")

    doc_type = st.radio(
        "Sélectionnez le type de document",
        options=["auto", "invoice", "contract"],
        format_func=lambda x: {
            "auto": "🔍 Détection automatique",
            "invoice": "🧾 Facture",
            "contract": "📝 Contrat",
        }[x],
        horizontal=True,
    )

    return None if doc_type == "auto" else doc_type
