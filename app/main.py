"""Application Streamlit pour l'extraction de documents."""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

from document_extraction.config import get_config, Config
from document_extraction.pipeline import ExtractionPipeline
from document_extraction.ocr import is_tesseract_available

from components.file_uploader import render_file_uploader, render_document_type_selector
from components.document_viewer import render_document_viewer, render_ocr_text
from components.results_display import render_results_display, render_error_message


def init_session_state() -> None:
    """Initialise le state de la session."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "result" not in st.session_state:
        st.session_state.result = None


def get_pipeline(config: Config) -> ExtractionPipeline:
    """Récupère ou crée le pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("Initialisation du pipeline..."):
            st.session_state.pipeline = ExtractionPipeline(config)
    return st.session_state.pipeline


def render_header() -> None:
    """Affiche l'en-tête de l'application."""
    st.set_page_config(
        page_title="Document Extraction POC",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📄 Document Extraction POC")
    st.markdown(
        "Extraction automatique de données depuis des factures et contrats "
        "utilisant OCR et LLM."
    )


def render_sidebar(config: Config) -> None:
    """Affiche la sidebar avec les informations système."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Statut des composants
        st.subheader("Statut des composants")

        # OCR
        ocr_available = is_tesseract_available()
        if ocr_available:
            st.success("✅ Tesseract OCR")
        else:
            st.error("❌ Tesseract non installé")
            st.caption("Installez Tesseract pour activer l'OCR")

        # LLM
        if config.vertex_ai.use_mock:
            st.warning("⚠️ Mode simulation LLM")
            st.caption("Les extractions sont simulées")
        else:
            st.success("✅ Vertex AI configuré")

        # Configuration OCR
        st.subheader("Paramètres OCR")
        st.text(f"Langue: {config.ocr.tesseract.lang}")
        st.text(f"PSM: {config.ocr.tesseract.psm}")

        # Informations
        st.markdown("---")
        st.subheader("📚 À propos")
        st.markdown("""
        **Document Extraction POC**

        Ce projet démontre l'extraction automatique
        de données depuis des documents numérisés.

        **Technologies:**
        - OCR: Tesseract
        - LLM: Vertex AI (Gemini)
        - Interface: Streamlit
        """)

        # Liens
        st.markdown("---")
        st.markdown(
            "💻 [Code source](https://github.com/example/document-extraction)"
        )


def render_main_content(config: Config) -> None:
    """Affiche le contenu principal."""
    # Upload et type de document
    col1, col2 = st.columns([2, 1])

    with col1:
        file_bytes, filename = render_file_uploader(
            allowed_extensions=config.app.allowed_extensions,
            max_size_mb=config.app.max_file_size_mb,
        )

    with col2:
        document_type = render_document_type_selector()

    if file_bytes is None:
        st.info("👆 Uploadez un document pour commencer l'extraction.")
        _render_demo_section()
        return

    # Visualisation du document
    st.markdown("---")
    render_document_viewer(file_bytes, filename)

    # Bouton d'extraction
    st.markdown("---")
    if st.button("🚀 Lancer l'extraction", type="primary", use_container_width=True):
        _run_extraction(config, file_bytes, filename, document_type)

    # Affichage des résultats
    if st.session_state.result is not None:
        st.markdown("---")
        _display_results()


def _run_extraction(
    config: Config,
    file_bytes: bytes,
    filename: str,
    document_type: str | None,
) -> None:
    """Lance l'extraction."""
    pipeline = get_pipeline(config)

    with st.spinner("🔄 Extraction en cours..."):
        # Progress bar simulée
        progress_bar = st.progress(0)

        progress_bar.progress(20, "Preprocessing du document...")
        progress_bar.progress(50, "Extraction OCR...")
        progress_bar.progress(80, "Analyse LLM...")

        result = pipeline.process_bytes(
            file_bytes,
            filename,
            document_type=document_type,
        )

        progress_bar.progress(100, "Terminé!")
        st.session_state.result = result


def _display_results() -> None:
    """Affiche les résultats de l'extraction."""
    result = st.session_state.result

    if result.success:
        # Infos OCR
        if result.ocr_result:
            render_ocr_text(
                result.ocr_result.text,
                result.ocr_result.word_count,
                result.ocr_result.confidence,
            )

        # Données extraites
        st.markdown("---")
        render_results_display(result.data, result.document_type)

        # Métriques de performance
        st.markdown("---")
        _render_performance_metrics(result)

    else:
        render_error_message(result.error_message)


def _render_performance_metrics(result) -> None:
    """Affiche les métriques de performance."""
    st.markdown("### ⏱️ Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        ocr_time = result.ocr_result.processing_time_ms if result.ocr_result else 0
        st.metric("Temps OCR", f"{ocr_time:.0f} ms")

    with col2:
        st.metric("Type détecté", result.document_type.upper())

    with col3:
        fields = result.data.get_extracted_fields()
        extracted = sum(fields.values())
        total = len(fields)
        st.metric("Champs extraits", f"{extracted}/{total}")


def _render_demo_section() -> None:
    """Affiche une section de démonstration."""
    st.markdown("---")
    st.markdown("### 🎯 Fonctionnalités")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🧾 Extraction de factures**
        - Numéro et dates de facture
        - Informations fournisseur/client
        - Montants HT, TVA, TTC
        - Lignes de détail
        """)

    with col2:
        st.markdown("""
        **📝 Extraction de contrats**
        - Type et référence
        - Parties prenantes
        - Dates et durée
        - Clauses importantes
        """)

    st.markdown("---")
    st.markdown("### 🔄 Pipeline de traitement")
    st.markdown("""
    1. **Preprocessing** - Conversion PDF, amélioration de la qualité d'image
    2. **OCR** - Extraction du texte avec Tesseract
    3. **LLM** - Analyse structurée avec Vertex AI (Gemini)
    4. **Export** - Résultats en JSON ou CSV
    """)


def main():
    """Point d'entrée de l'application."""
    init_session_state()
    render_header()

    try:
        config = get_config()
    except Exception as e:
        st.error(f"Erreur de configuration: {e}")
        st.stop()

    render_sidebar(config)
    render_main_content(config)


if __name__ == "__main__":
    main()
