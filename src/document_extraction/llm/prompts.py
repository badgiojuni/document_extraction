"""Templates de prompts pour l'extraction de données."""

INVOICE_EXTRACTION_PROMPT = """Tu es un expert en extraction de données de factures. Analyse le texte OCR suivant et extrais les informations structurées.

TEXTE DU DOCUMENT:
```
{ocr_text}
```

Extrais les informations suivantes au format JSON strict:

{{
    "invoice_number": "numéro de facture ou null",
    "invoice_date": "date au format YYYY-MM-DD ou null",
    "due_date": "date d'échéance au format YYYY-MM-DD ou null",
    "supplier_name": "nom du fournisseur ou null",
    "supplier_address": "adresse complète du fournisseur ou null",
    "supplier_siret": "numéro SIRET (14 chiffres) ou null",
    "supplier_vat_number": "numéro TVA intracommunautaire ou null",
    "client_name": "nom du client ou null",
    "client_address": "adresse complète du client ou null",
    "client_siret": "numéro SIRET du client ou null",
    "total_ht": montant HT en nombre décimal ou null,
    "total_tva": montant TVA en nombre décimal ou null,
    "total_ttc": montant TTC en nombre décimal ou null,
    "line_items": [
        {{
            "description": "description du produit/service",
            "quantity": quantité en nombre ou null,
            "unit_price": prix unitaire HT en nombre ou null,
            "total_ht": total ligne HT en nombre ou null,
            "tva_rate": taux TVA en pourcentage ou null
        }}
    ],
    "confidence_score": score de confiance entre 0 et 1
}}

RÈGLES IMPORTANTES:
1. Retourne UNIQUEMENT le JSON, sans texte avant ou après
2. Utilise null pour les champs non trouvés
3. Les montants doivent être des nombres (pas de chaînes)
4. Les dates doivent être au format YYYY-MM-DD
5. Le score de confiance reflète la qualité de l'extraction (1.0 = très confiant)
6. Pour les lignes de facture, extrais autant d'informations que possible

JSON:"""


CONTRACT_EXTRACTION_PROMPT = """Tu es un expert en analyse juridique de contrats. Analyse le texte OCR suivant et extrais les informations structurées.

TEXTE DU DOCUMENT:
```
{ocr_text}
```

Extrais les informations suivantes au format JSON strict:

{{
    "contract_type": "service|employment|lease|sale|nda|partnership|other",
    "contract_number": "référence du contrat ou null",
    "title": "titre du contrat ou null",
    "parties": [
        {{
            "name": "nom de la partie",
            "role": "rôle (vendeur, acheteur, employeur, employé, bailleur, locataire, etc.)",
            "address": "adresse ou null",
            "siret": "numéro SIRET ou null",
            "representative": "représentant légal ou null"
        }}
    ],
    "signature_date": "date de signature au format YYYY-MM-DD ou null",
    "effective_date": "date d'entrée en vigueur au format YYYY-MM-DD ou null",
    "end_date": "date de fin au format YYYY-MM-DD ou null",
    "duration": "durée du contrat (ex: '12 mois', '3 ans') ou null",
    "total_amount": montant total en nombre décimal ou null,
    "payment_terms": "conditions de paiement ou null",
    "currency": "devise (EUR, USD, etc.) ou EUR par défaut",
    "key_clauses": [
        {{
            "title": "type de clause (confidentialité, non-concurrence, résiliation, etc.)",
            "content": "résumé de la clause",
            "importance": "high|medium|low"
        }}
    ],
    "termination_conditions": "conditions de résiliation ou null",
    "renewal_terms": "conditions de renouvellement ou null",
    "signatures": ["liste des signataires identifiés"],
    "confidence_score": score de confiance entre 0 et 1
}}

RÈGLES IMPORTANTES:
1. Retourne UNIQUEMENT le JSON, sans texte avant ou après
2. Utilise null pour les champs non trouvés
3. Les montants doivent être des nombres (pas de chaînes)
4. Les dates doivent être au format YYYY-MM-DD
5. Identifie les clauses importantes (confidentialité, pénalités, résiliation, etc.)
6. Le score de confiance reflète la qualité de l'extraction

JSON:"""


DOCUMENT_CLASSIFICATION_PROMPT = """Analyse le texte suivant et détermine le type de document.

TEXTE:
```
{ocr_text}
```

Réponds UNIQUEMENT avec un des types suivants:
- invoice (facture)
- contract (contrat)
- unknown (autre/inconnu)

Type:"""


def get_invoice_prompt(ocr_text: str) -> str:
    """Génère le prompt d'extraction de facture."""
    return INVOICE_EXTRACTION_PROMPT.format(ocr_text=ocr_text)


def get_contract_prompt(ocr_text: str) -> str:
    """Génère le prompt d'extraction de contrat."""
    return CONTRACT_EXTRACTION_PROMPT.format(ocr_text=ocr_text)


def get_classification_prompt(ocr_text: str) -> str:
    """Génère le prompt de classification de document."""
    return DOCUMENT_CLASSIFICATION_PROMPT.format(ocr_text=ocr_text)
