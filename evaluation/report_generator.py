"""Génération de rapports HTML."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from .metrics import EvaluationResults
except ImportError:
    from metrics import EvaluationResults


def generate_html_report(
    results: EvaluationResults,
    output_path: str | Path,
    title: str = "Rapport d'évaluation - Document Extraction",
    include_charts: bool = True,
) -> Path:
    """
    Génère un rapport HTML complet.

    Args:
        results: Résultats de l'évaluation
        output_path: Chemin de sortie du rapport
        title: Titre du rapport
        include_charts: Inclure les graphiques interactifs

    Returns:
        Chemin du rapport généré
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Générer les graphiques
    charts_html = ""
    if include_charts:
        charts_html = _generate_charts(results)

    # Générer le HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --primary: #4F46E5;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --gray-100: #F3F4F6;
            --gray-800: #1F2937;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-800);
            background: var(--gray-100);
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--primary); margin-bottom: 0.5rem; }}
        .subtitle {{ color: #6B7280; margin-bottom: 2rem; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: var(--primary);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--gray-100);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .metric {{
            text-align: center;
            padding: 1rem;
            background: var(--gray-100);
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        .metric-label {{ color: #6B7280; font-size: 0.875rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-100);
        }}
        th {{ background: var(--gray-100); font-weight: 600; }}
        tr:hover {{ background: #F9FAFB; }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-success {{ background: #D1FAE5; color: #065F46; }}
        .badge-warning {{ background: #FEF3C7; color: #92400E; }}
        .badge-danger {{ background: #FEE2E2; color: #991B1B; }}
        .chart-container {{ margin: 1.5rem 0; }}
        .footer {{
            text-align: center;
            color: #6B7280;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #E5E7EB;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <p class="subtitle">Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>

        <!-- Résumé -->
        <div class="card">
            <h2>📈 Résumé global</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{results.total_documents}</div>
                    <div class="metric-label">Documents traités</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.success_rate:.1%}</div>
                    <div class="metric-label">Taux de succès</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.macro_f1:.2f}</div>
                    <div class="metric-label">F1-Score macro</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.avg_processing_time:.0f}ms</div>
                    <div class="metric-label">Temps moyen</div>
                </div>
            </div>
        </div>

        <!-- Métriques par champ -->
        <div class="card">
            <h2>🎯 Métriques par champ</h2>
            <table>
                <thead>
                    <tr>
                        <th>Champ</th>
                        <th>Précision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Accuracy</th>
                        <th>Statut</th>
                    </tr>
                </thead>
                <tbody>
                    {_generate_field_rows(results)}
                </tbody>
            </table>
        </div>

        <!-- Graphiques -->
        {charts_html}

        <!-- Détails -->
        <div class="card">
            <h2>📋 Données brutes</h2>
            <pre style="background: var(--gray-100); padding: 1rem; border-radius: 8px; overflow-x: auto;">
{json.dumps(results.to_dict(), indent=2, ensure_ascii=False)}
            </pre>
        </div>

        <div class="footer">
            <p>Document Extraction POC - Rapport d'évaluation</p>
        </div>
    </div>
</body>
</html>
    """

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def _generate_field_rows(results: EvaluationResults) -> str:
    """Génère les lignes du tableau des métriques."""
    rows = []

    for name, metrics in sorted(results.field_metrics.items()):
        # Déterminer le badge de statut
        if metrics.f1_score >= 0.8:
            badge = '<span class="badge badge-success">Excellent</span>'
        elif metrics.f1_score >= 0.5:
            badge = '<span class="badge badge-warning">Moyen</span>'
        else:
            badge = '<span class="badge badge-danger">À améliorer</span>'

        rows.append(f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{metrics.precision:.2%}</td>
                <td>{metrics.recall:.2%}</td>
                <td>{metrics.f1_score:.2%}</td>
                <td>{metrics.accuracy:.2%}</td>
                <td>{badge}</td>
            </tr>
        """)

    return "\n".join(rows)


def _generate_charts(results: EvaluationResults) -> str:
    """Génère les graphiques Plotly."""
    charts = []

    # Graphique 1: F1-Score par champ
    field_names = list(results.field_metrics.keys())
    f1_scores = [m.f1_score for m in results.field_metrics.values()]

    fig1 = go.Figure(data=[
        go.Bar(
            x=field_names,
            y=f1_scores,
            marker_color=['#10B981' if s >= 0.8 else '#F59E0B' if s >= 0.5 else '#EF4444' for s in f1_scores],
        )
    ])
    fig1.update_layout(
        title="F1-Score par champ",
        xaxis_title="Champ",
        yaxis_title="F1-Score",
        yaxis_range=[0, 1],
        height=400,
    )

    # Graphique 2: Précision vs Recall
    precisions = [m.precision for m in results.field_metrics.values()]
    recalls = [m.recall for m in results.field_metrics.values()]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=precisions,
        y=recalls,
        mode='markers+text',
        text=field_names,
        textposition='top center',
        marker=dict(size=12, color='#4F46E5'),
    ))
    fig2.add_shape(
        type='line',
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color='gray', dash='dash'),
    )
    fig2.update_layout(
        title="Précision vs Recall",
        xaxis_title="Précision",
        yaxis_title="Recall",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        height=400,
    )

    # Convertir en HTML
    charts_html = f"""
    <div class="card">
        <h2>📊 Visualisations</h2>
        <div class="chart-container">
            {fig1.to_html(full_html=False, include_plotlyjs=False)}
        </div>
        <div class="chart-container">
            {fig2.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
    """

    return charts_html


def generate_comparison_report(
    ocr_only_results: EvaluationResults,
    ocr_llm_results: EvaluationResults,
    output_path: str | Path,
) -> Path:
    """
    Génère un rapport de comparaison OCR seul vs OCR + LLM.

    Args:
        ocr_only_results: Résultats avec OCR uniquement
        ocr_llm_results: Résultats avec OCR + LLM
        output_path: Chemin de sortie

    Returns:
        Chemin du rapport généré
    """
    output_path = Path(output_path)

    # Créer un graphique de comparaison
    fields = list(set(ocr_only_results.field_metrics.keys()) |
                  set(ocr_llm_results.field_metrics.keys()))

    ocr_f1 = [ocr_only_results.field_metrics.get(f, None) for f in fields]
    ocr_f1 = [m.f1_score if m else 0 for m in ocr_f1]

    llm_f1 = [ocr_llm_results.field_metrics.get(f, None) for f in fields]
    llm_f1 = [m.f1_score if m else 0 for m in llm_f1]

    fig = go.Figure(data=[
        go.Bar(name='OCR seul', x=fields, y=ocr_f1, marker_color='#6B7280'),
        go.Bar(name='OCR + LLM', x=fields, y=llm_f1, marker_color='#4F46E5'),
    ])
    fig.update_layout(
        title="Comparaison OCR seul vs OCR + LLM",
        xaxis_title="Champ",
        yaxis_title="F1-Score",
        barmode='group',
        yaxis_range=[0, 1],
    )

    # Générer le rapport
    improvement = ocr_llm_results.macro_f1 - ocr_only_results.macro_f1

    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Comparaison OCR vs OCR + LLM</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: sans-serif; padding: 2rem; background: #F3F4F6; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }}
        h1 {{ color: #4F46E5; }}
        .improvement {{ font-size: 2rem; color: {'#10B981' if improvement > 0 else '#EF4444'}; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Comparaison OCR vs OCR + LLM</h1>

        <div class="card">
            <h2>Amélioration globale</h2>
            <p class="improvement">
                {'+' if improvement > 0 else ''}{improvement:.1%} F1-Score
            </p>
            <p>OCR seul: {ocr_only_results.macro_f1:.2%} → OCR + LLM: {ocr_llm_results.macro_f1:.2%}</p>
        </div>

        <div class="card">
            {fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
</body>
</html>
    """

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
