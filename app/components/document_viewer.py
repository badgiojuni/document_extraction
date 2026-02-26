"""Composant de visualisation de documents."""

import io
import streamlit as st
from PIL import Image
from typing import Optional


def render_document_viewer(
    file_bytes: bytes,
    filename: str,
) -> None:
    """
    Affiche le document uploadé.

    Args:
        file_bytes: Contenu du fichier en bytes
        filename: Nom du fichier
    """
    st.markdown("### 👁️ Aperçu du document")

    if filename.lower().endswith(".pdf"):
        _render_pdf_preview(file_bytes)
    else:
        _render_image_preview(file_bytes)


def _render_pdf_preview(file_bytes: bytes) -> None:
    """Affiche un aperçu de PDF."""
    try:
        from pdf2image import convert_from_bytes

        # Convertir la première page seulement pour l'aperçu
        with st.spinner("Génération de l'aperçu..."):
            images = convert_from_bytes(
                file_bytes,
                dpi=100,  # DPI réduit pour l'aperçu
                first_page=1,
                last_page=1,
            )

        if images:
            st.image(
                images[0],
                caption="Page 1",
                use_container_width=True,
            )

            # Compter le nombre total de pages
            all_images = convert_from_bytes(file_bytes, dpi=50)
            if len(all_images) > 1:
                st.caption(f"📄 Document de {len(all_images)} pages")

    except Exception as e:
        st.warning(f"⚠️ Impossible d'afficher l'aperçu PDF: {e}")
        st.info("Le document sera tout de même traité.")


def _render_image_preview(file_bytes: bytes) -> None:
    """Affiche un aperçu d'image."""
    try:
        image = Image.open(io.BytesIO(file_bytes))

        # Redimensionner si trop grand
        max_width = 800
        if image.width > max_width:
            ratio = max_width / image.width
            new_size = (max_width, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        st.image(
            image,
            caption=f"Dimensions: {image.width}x{image.height}",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"❌ Erreur lors de l'affichage: {e}")


def render_ocr_text(text: str, word_count: int, confidence: float) -> None:
    """
    Affiche le texte OCR extrait.

    Args:
        text: Texte extrait
        word_count: Nombre de mots
        confidence: Score de confiance
    """
    st.markdown("### 📝 Texte OCR extrait")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mots détectés", word_count)
    with col2:
        st.metric("Confiance OCR", f"{confidence:.1%}")

    with st.expander("Voir le texte brut", expanded=False):
        st.text_area(
            "Texte OCR",
            text,
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )
