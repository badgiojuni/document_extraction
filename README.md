# PDF Extractor

Extraction d'informations depuis des documents PDF à structure variée via un Vision LLM (Gemini) sur Vertex AI.

## Architecture

```
PDF → [PyMuPDF] → Images PNG → [Vertex AI Gemini] → Texte/JSON
```

Les VLLM comme Gemini ne lisent pas directement les PDF. Le processus convertit chaque page en image, puis envoie ces images au modèle avec un prompt d'extraction.

## Structure du projet

```
pdf-extractor/
├── main.py                      # CLI pour extraire depuis un PDF
├── pyproject.toml               # Dépendances
└── src/pdf_extractor/
    ├── __init__.py
    ├── client.py                # Client Vertex AI (Gemini Vision)
    └── extractor.py             # Conversion PDF → images + extraction
```

## Installation

```bash
# Cloner et installer
uv sync

# Authentification GCP
gcloud auth application-default login
```

## Composants

### 1. VertexAIClient (`client.py`)

Encapsule la communication avec Vertex AI.

```python
class VertexAIClient:
    def __init__(self, project_id, location="europe-west1", model_name="gemini-2.0-flash-001"):
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
```

- `vertexai.init()` initialise le SDK avec les credentials GCP
- `GenerativeModel` charge le modèle Gemini (vision-capable)
- `europe-west1` : région par défaut pour la latence en Europe

#### Méthode `extract_from_images`

```python
def extract_from_images(self, images: list[bytes], prompt: str) -> str:
    parts = []
    for img_bytes in images:
        parts.append(Part.from_data(img_bytes, mime_type="image/png"))
    parts.append(prompt)
    response = self.model.generate_content(parts)
    return response.text
```

- Construit une requête **multimodale** : images + texte
- `Part.from_data()` encode chaque image comme partie de la requête
- Le prompt est ajouté **après** les images (le modèle "voit" d'abord les images, puis lit l'instruction)

### 2. PDFExtractor (`extractor.py`)

Convertit les PDF en images et orchestre l'extraction.

```python
class PDFExtractor:
    def __init__(self, client: VertexAIClient, dpi: int = 150):
```

- `dpi=150` : résolution de rendu. Bon compromis qualité/taille. Augmenter à 200-300 pour des documents avec petit texte.

#### Méthode `pdf_to_images`

```python
def pdf_to_images(self, pdf_path: Path) -> list[bytes]:
    doc = fitz.open(pdf_path)
    for page in doc:
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
```

- `fitz` = PyMuPDF (nom historique de la bibliothèque)
- `Matrix(dpi/72, dpi/72)` : facteur de zoom. PDF standard = 72 DPI, donc `150/72 ≈ 2x`
- `get_pixmap()` : rasterise la page en image
- `tobytes("png")` : encode en PNG (sans perte, adapté au texte)

#### Méthode `extract`

```python
def extract(self, pdf_path, prompt, pages=None) -> str:
```

- Extraction "libre" : prompt personnalisé, réponse en texte
- `pages` : limite aux pages spécifiées (économise tokens et coûts)

#### Méthode `extract_structured`

```python
def extract_structured(self, pdf_path, schema, pages=None) -> dict:
```

- Extraction structurée : schéma JSON en entrée, dictionnaire en sortie
- Le prompt force le modèle à répondre uniquement en JSON valide
- Nettoyage automatique des marqueurs markdown (` ```json ``` `)

### 3. CLI (`main.py`)

| Argument | Description |
|----------|-------------|
| `pdf` | Fichier PDF à traiter (obligatoire) |
| `--prompt, -p` | Instructions d'extraction libre |
| `--schema, -s` | Fichier JSON pour extraction structurée |
| `--project` | ID projet GCP (ou variable `GOOGLE_CLOUD_PROJECT`) |
| `--location` | Région Vertex AI (défaut: `europe-west1`) |
| `--pages` | Pages à traiter : `0,1,2` ou `0-5` |

## Utilisation

### Prérequis

```bash
# Définir le projet GCP
export GOOGLE_CLOUD_PROJECT="mon-projet-gcp"

# Ou passer via --project à chaque appel
```

### Extraction libre

```bash
# Extraction par défaut (toutes infos importantes)
python main.py facture.pdf

# Extraction avec prompt personnalisé
python main.py facture.pdf -p "Extrait le nom du client, la date et le montant total TTC"

# Limiter aux 2 premières pages
python main.py rapport.pdf -p "Résume ce document" --pages 0,1
```

### Extraction structurée

Créer un fichier `schema.json` :

```json
{
  "numero_facture": "string",
  "date": "string (format JJ/MM/AAAA)",
  "client": {
    "nom": "string",
    "adresse": "string"
  },
  "lignes": [
    {
      "description": "string",
      "quantite": "number",
      "prix_unitaire": "number"
    }
  ],
  "total_ht": "number",
  "tva": "number",
  "total_ttc": "number"
}
```

```bash
python main.py facture.pdf -s schema.json
```

Sortie :

```json
{
  "numero_facture": "FAC-2024-001234",
  "date": "15/03/2024",
  "client": {
    "nom": "Entreprise ABC",
    "adresse": "123 rue Example, 75001 Paris"
  },
  "lignes": [
    {
      "description": "Prestation de conseil",
      "quantite": 5,
      "prix_unitaire": 500.00
    }
  ],
  "total_ht": 2500.00,
  "tva": 500.00,
  "total_ttc": 3000.00
}
```

### Utilisation programmatique

```python
from src.pdf_extractor import PDFExtractor, VertexAIClient

# Initialisation
client = VertexAIClient(project_id="mon-projet-gcp")
extractor = PDFExtractor(client, dpi=200)

# Extraction libre
texte = extractor.extract("document.pdf", "Liste tous les noms de personnes mentionnées")

# Extraction structurée
schema = {
    "titre": "string",
    "auteur": "string",
    "date_publication": "string"
}
data = extractor.extract_structured("article.pdf", schema, pages=[0])
```

## Flux de données

```
1. main.py reçoit : document.pdf + prompt
2. PDFExtractor.extract() appelé
3. pdf_to_images() convertit chaque page en PNG (bytes)
4. VertexAIClient.extract_from_images() envoie à Gemini :
   [Image1, Image2, ..., "Extrait le nom et l'adresse"]
5. Gemini analyse visuellement et répond
6. Réponse affichée (texte ou JSON parsé)
```

## Configuration avancée

### Changer de modèle

```python
client = VertexAIClient(
    project_id="mon-projet",
    model_name="gemini-1.5-pro-001"  # Plus puissant, plus lent
)
```

### Ajuster la résolution

```python
extractor = PDFExtractor(client, dpi=300)  # Meilleure qualité pour petit texte
```

## Limites

- Coût API proportionnel au nombre de pages et à la résolution
- Documents très longs (>50 pages) peuvent dépasser les limites de contexte
- La qualité d'extraction dépend de la lisibilité du PDF (scans de mauvaise qualité = résultats dégradés)
