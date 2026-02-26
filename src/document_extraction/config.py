"""Module de configuration pour l'extraction de documents."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VertexAIConfig(BaseModel):
    """Configuration Vertex AI."""

    project_id: str = Field(default="")
    location: str = Field(default="europe-west1")
    model_name: str = Field(default="gemini-1.5-flash")
    credentials_path: Optional[str] = None
    use_mock: bool = Field(default=True)


class TesseractConfig(BaseModel):
    """Configuration Tesseract OCR."""

    lang: str = Field(default="fra")
    psm: int = Field(default=6)
    oem: int = Field(default=3)


class OCRConfig(BaseModel):
    """Configuration OCR."""

    tesseract: TesseractConfig = Field(default_factory=TesseractConfig)


class PDFConfig(BaseModel):
    """Configuration conversion PDF."""

    dpi: int = Field(default=300)
    output_format: str = Field(default="PNG")


class ImageConfig(BaseModel):
    """Configuration traitement image."""

    denoise: bool = Field(default=True)
    binarize: bool = Field(default=True)
    deskew: bool = Field(default=True)
    enhance_contrast: bool = Field(default=True)


class PreprocessingConfig(BaseModel):
    """Configuration preprocessing."""

    pdf: PDFConfig = Field(default_factory=PDFConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)


class LoggingConfig(BaseModel):
    """Configuration logging."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[str] = Field(default=None)


class AppConfig(BaseModel):
    """Configuration application Streamlit."""

    title: str = Field(default="Document Extraction POC")
    max_file_size_mb: int = Field(default=10)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: ["pdf", "png", "jpg", "jpeg", "tiff"]
    )


class Config(BaseModel):
    """Configuration principale de l'application."""

    vertex_ai: VertexAIConfig = Field(default_factory=VertexAIConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    document_types: list[str] = Field(default_factory=lambda: ["invoice", "contract"])
    extraction: dict[str, Any] = Field(default_factory=dict)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    app: AppConfig = Field(default_factory=AppConfig)


def find_config_file() -> Optional[Path]:
    """Recherche le fichier de configuration dans les emplacements standards."""
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path(__file__).parent.parent.parent.parent / "config.yaml",
        Path.home() / ".config" / "document_extraction" / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Charge la configuration depuis un fichier YAML.

    Args:
        config_path: Chemin vers le fichier de configuration.
                    Si None, recherche automatiquement.

    Returns:
        Config: Instance de configuration.

    Raises:
        FileNotFoundError: Si aucun fichier de configuration n'est trouvé.
    """
    if config_path is None:
        config_path = find_config_file()

    if config_path is None:
        logger.warning("Aucun fichier de configuration trouvé, utilisation des valeurs par défaut")
        return Config()

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")

    logger.info(f"Chargement de la configuration depuis {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    # Substitution des variables d'environnement
    config_data = _substitute_env_vars(config_data)

    return Config(**config_data)


def _substitute_env_vars(config: dict) -> dict:
    """
    Substitue les variables d'environnement dans la configuration.

    Supporte la syntaxe ${VAR_NAME} ou ${VAR_NAME:default_value}
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _substitute_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_expr = value[2:-1]
            if ":" in var_expr:
                var_name, default = var_expr.split(":", 1)
            else:
                var_name, default = var_expr, None
            result[key] = os.environ.get(var_name, default)
        elif isinstance(value, list):
            result[key] = [
                _substitute_env_vars(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def setup_logging(config: Config) -> None:
    """Configure le logging selon la configuration."""
    log_config = config.logging

    # Créer le répertoire de logs si nécessaire
    if log_config.file:
        log_dir = Path(log_config.file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configuration du logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_config.file:
        handlers.append(logging.FileHandler(log_config.file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_config.level.upper()),
        format=log_config.format,
        handlers=handlers,
        force=True,
    )

    logger.info("Logging configuré avec succès")


# Instance globale de configuration (lazy loading)
_config: Optional[Config] = None


def get_config() -> Config:
    """Retourne l'instance globale de configuration."""
    global _config
    if _config is None:
        _config = load_config()
        setup_logging(_config)
    return _config


def reset_config() -> None:
    """Réinitialise la configuration globale (utile pour les tests)."""
    global _config
    _config = None
