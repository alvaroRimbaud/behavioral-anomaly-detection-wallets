from __future__ import annotations

from pathlib import Path
import html
import json
import pandas as pd


def _table(df: pd.DataFrame, decimals: int = 4) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    display = df.copy()
    for c in display.select_dtypes(include="number").columns:
        display[c] = display[c].map(lambda x: f"{x:.{decimals}f}" if isinstance(x, float) else x)
    return display.to_html(index=False, classes="tbl", border=0)


def build_report(output_dir: Path, summary: dict, metrics: pd.DataFrame,
                 intensity: pd.DataFrame, dimensions: pd.DataFrame,
                 alerts: pd.DataFrame) -> Path:
    metrics_show = metrics[["model", "feature_set", "precision", "recall", "f1", "pr_auc", "fp", "fn", "threshold"]].copy()
    alerts_show = alerts.head(20).copy()
    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f4f6f8;color:#17202a}
    header{background:#163a5f;color:white;padding:24px 36px}
    main{max-width:1200px;margin:24px auto;padding:0 20px}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .card{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px #0002}
    .kpi{font-size:26px;font-weight:700;margin-top:6px}
    h2{margin-top:28px;color:#163a5f}.tbl{border-collapse:collapse;width:100%;background:white}
    .tbl th,.tbl td{padding:9px 10px;border-bottom:1px solid #dde3e8;text-align:left;font-size:13px}
    .tbl th{background:#eaf0f5;color:#163a5f;position:sticky;top:0}
    .note{background:#fff7db;border-left:4px solid #d5a000;padding:12px 14px;margin:16px 0}
    .good{color:#08783e;font-weight:700}.bad{color:#a11;font-weight:700}
    footer{margin:40px 0 20px;color:#667;font-size:12px}
    @media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}}
    """
    total = summary["n_transactions"]
    test_anom = summary["test_anomalies"]
    best = metrics.sort_values("pr_auc", ascending=False).iloc[0]
    html_doc = f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><title>T12 Demo - Detección de anomalías</title><style>{css}</style></head>
    <body><header><h1>T12 — Detección de anomalías conductuales</h1><p>Demo reproducible S8 · billeteras digitales sintéticas · entrenamiento no supervisado</p></header>
    <main>
      <div class='note'><b>Alcance:</b> esta es una corrida base de construcción inicial. Las métricas son reales para el dataset sintético generado con la configuración registrada, pero no constituyen todavía el experimento final de tesis.</div>
      <div class='grid'>
        <div class='card'><div>Transacciones</div><div class='kpi'>{total:,}</div></div>
        <div class='card'><div>Usuarios</div><div class='kpi'>{summary['n_users']:,}</div></div>
        <div class='card'><div>Anomalías en test</div><div class='kpi'>{test_anom:,}</div></div>
        <div class='card'><div>Mejor PR-AUC</div><div class='kpi'>{best['pr_auc']:.3f}</div><div>{html.escape(str(best['model']))} / {html.escape(str(best['feature_set']))}</div></div>
      </div>
      <h2>1. Desempeño global en test</h2>{_table(metrics_show)}
      <h2>2. Recall por intensidad de anomalía</h2>{_table(intensity)}
      <h2>3. Recall por dimensión alterada</h2>{_table(dimensions)}
      <h2>4. Alertas de mayor puntaje (modelo principal)</h2>{_table(alerts_show)}
      <h2>5. Trazabilidad de la corrida</h2>
      <pre class='card'>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
      <footer>Generado automáticamente por el pipeline T12. Ground truth excluido del entrenamiento; usado en validación y evaluación.</footer>
    </main></body></html>"""
    path = output_dir / "reporte_demo.html"
    path.write_text(html_doc, encoding="utf-8")
    return path
