# Document Extraction POC

POC d'extraction de données depuis des documents (factures, contrats) utilisant OCR et LLM.

## Fonctionnalités

- **Extraction de factures** : numéro, date, montants (HT/TTC), fournisseur, client, lignes de détail
- **Extraction de contrats** : parties, dates, montants, clauses importantes
- **Interface Streamlit** : upload, visualisation, export JSON/CSV
- **Pipeline modulaire** : preprocessing, OCR (Tesseract), LLM (Vertex AI)
- **Évaluation** : métriques de performance, rapport HTML

## Prérequis

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (gestionnaire de paquets)
- Tesseract OCR installé sur le système
- (Optionnel) Compte Google Cloud avec Vertex AI activé

### Installation de Tesseract

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang

# Windows
# Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
```

## Installation

```bash
# Cloner le projet
git clone <repo-url>
cd document_extraction

# Créer l'environnement et installer les dépendances
uv sync

# Copier et configurer le fichier de configuration
cp config.example.yaml config.yaml
# Éditer config.yaml avec vos paramètres
```

## Configuration

Éditer `config.yaml` :

```yaml
vertex_ai:
  project_id: "your-gcp-project-id"
  location: "europe-west1"
  use_mock: false  # true pour tester sans credentials GCP

ocr:
  tesseract:
    lang: "fra"  # ou "eng" ou "fra+eng"
```

## Utilisation

### Interface Streamlit

```bash
uv run streamlit run app/main.py
```

### Ligne de commande

```bash
# Extraire un document
uv run extract --input document.pdf --type invoice --output result.json

# Évaluer les performances
uv run evaluate --data evaluation/test_data
```

### Notebook

```bash
uv run jupyter notebook notebooks/
```

## Structure du projet

```
document_extraction/
├── src/document_extraction/    # Code source principal
│   ├── preprocessing/          # Conversion PDF, amélioration images
│   ├── ocr/                    # Extraction texte (Tesseract)
│   ├── llm/                    # Client Vertex AI, extracteurs
│   └── models/                 # Modèles de données Pydantic
├── app/                        # Application Streamlit
├── evaluation/                 # Scripts d'évaluation
├── notebooks/                  # Notebooks pédagogiques
└── tests/                      # Tests unitaires
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Document  │────▶│ Preprocessing│────▶│     OCR     │────▶│     LLM      │
│  (PDF/IMG)  │     │              │     │ (Tesseract) │     │ (Vertex AI)  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                           │                    │                    │
                           ▼                    ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
                    │   Images    │     │  Texte brut │     │   Données    │
                    │ optimisées  │     │             │     │  structurées │
                    └─────────────┘     └─────────────┘     └──────────────┘
```

## Tests

```bash
# Lancer les tests
uv run pytest

# Avec couverture
uv run pytest --cov=src/document_extraction --cov-report=html
```

## Licence

MIT
