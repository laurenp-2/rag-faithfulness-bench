"""
visualization.py — RAG robustness evaluation dashboard
Usage:
    python visualization.py results.json
    python visualization.py results.json --output report.html
    python visualization.py results.json --output report.html --title "My Eval Run"

Expects a JSON file that is a list of records with this shape:
{
  "perturbation_type": "original" | "paraphrase" | "entity_swap" | "negation",
  "scores": {
    "faithfulness_score": float,
    "correctness_em": bool,
    "correctness_f1": float,
    "hallucination_flag": bool,
    "context_contradiction": bool
  },
  "generation": {
    "abstained": bool
  }
}
Any extra fields are ignored, so your full results file works as-is.
"""

# python eval/visualization.py results/outputs.json --output report.html --title "My Eval Run"
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict


PERTURBATION_ORDER = ["original", "paraphrase", "entity_swap", "negation"]
PERTURBATION_LABELS = {
    "original":    "Original",
    "paraphrase":  "Paraphrase",
    "entity_swap": "Entity swap",
    "negation":    "Negation",
}
COLORS = {
    "original":    "#378ADD",
    "paraphrase":  "#1D9E75",
    "entity_swap": "#EF9F27",
    "negation":    "#D85A30",
}


IDEAL_DIRECTION = {
    ("faith",     "original"):    "high",
    ("faith",     "paraphrase"):  "high",
    ("faith",     "entity_swap"): "low",
    ("faith",     "negation"):    "low",
    ("em",        "original"):    "high",
    ("em",        "paraphrase"):  "high",
    ("em",        "entity_swap"): "low",
    ("em",        "negation"):    "low",
    ("f1",        "original"):    "high",
    ("f1",        "paraphrase"):  "high",
    ("f1",        "entity_swap"): "low",
    ("f1",        "negation"):    "low",
    ("abstain",   "original"):    "low",
    ("abstain",   "paraphrase"):  "low",
    ("abstain",   "entity_swap"): "high",
    ("abstain",   "negation"):    "high",
    ("halluc",    "original"):    "low",
    ("halluc",    "paraphrase"):  "low",
    ("halluc",    "entity_swap"): "low",
    ("halluc",    "negation"):    "low",
    ("contradict","original"):    "low",
    ("contradict","paraphrase"):  "low",
    ("contradict","entity_swap"): "high",
    ("contradict","negation"):    "high",
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must be a list of result records.")
    return data


def compute_metrics(records: list[dict]) -> dict[str, dict]:
    """
    Groups records by perturbation_type and computes per-group metrics.
    Returns: { perturbation_type: { metric_key: float, "n": int } }
    """
    groups = defaultdict(list)
    for r in records:
        pt = r.get("perturbation_type", "unknown")
        groups[pt].append(r)

    metrics = {}
    for pt, recs in groups.items():
        def mean_score(key):
            vals = [r["scores"][key] for r in recs if key in r.get("scores", {})]
            return statistics.mean(vals) if vals else 0.0

        def mean_gen(key):
            vals = [float(r["generation"][key]) for r in recs if key in r.get("generation", {})]
            return statistics.mean(vals) if vals else 0.0

        metrics[pt] = {
            "n":         len(recs),
            "faith":     mean_score("faithfulness_score"),
            "em":        mean_score("correctness_em"),
            "f1":        mean_score("correctness_f1"),
            "halluc":    mean_score("hallucination_flag"),
            "contradict":mean_score("context_contradiction"),
            "abstain":   mean_gen("abstained"),
        }
    return metrics


def cell_color(value: float, metric: str, condition: str) -> str:
    """Returns a CSS color string for a table cell based on whether the value is desirable."""
    direction = IDEAL_DIRECTION.get((metric, condition), "high")
    pct = value * 100
    if direction == "high":
        if pct >= 60: return "#1D9E75"
        if pct >= 30: return "#BA7517"
        return "#D85A30"
    else:
        if pct <= 20: return "#1D9E75"
        if pct <= 40: return "#BA7517"
        return "#D85A30"


def fmt(v: float) -> str:
    return f"{v*100:.0f}%"


def build_html(metrics: dict, title: str) -> str:
    # Ordered conditions present in data
    conditions = [c for c in PERTURBATION_ORDER if c in metrics]
    # Also include any unexpected conditions at the end
    for c in metrics:
        if c not in conditions:
            conditions.append(c)

    # Build chart dataset JSON for Chart.js
    metric_keys = ["faith", "em", "f1", "abstain", "halluc", "contradict"]
    metric_labels = ["Faithfulness", "Exact match", "Token F1", "Abstention", "Hallucination", "Contradiction"]

    datasets_json = json.dumps([
        {
            "label": PERTURBATION_LABELS.get(c, c),
            "data":  [round(metrics[c][k], 3) for k in metric_keys],
            "backgroundColor": COLORS.get(c, "#888"),
        }
        for c in conditions
    ])
    chart_labels_json = json.dumps(metric_labels)

    # Build table rows
    table_rows = ""
    col_headers = ["faith","em","f1","abstain","halluc","contradict"]
    col_display  = ["Faith↑","EM↑","F1↑","Abstain↑/↓","Halluc↓","Contradict↓"]

    for c in conditions:
        m = metrics[c]
        label = PERTURBATION_LABELS.get(c, c)
        cells = f'<td class="label">{label}</td><td>{m["n"]}</td>'
        for k in col_headers:
            color = cell_color(m[k], k, c)
            cells += f'<td style="color:{color};font-weight:500">{fmt(m[k])}</td>'
        table_rows += f"<tr>{cells}</tr>\n"

    # Build legend HTML
    legend_html = ""
    for c in conditions:
        col = COLORS.get(c, "#888")
        lbl = PERTURBATION_LABELS.get(c, c)
        legend_html += f'<span><span class="dot" style="background:{col}"></span>{lbl}</span>\n'

    # Column headers for table
    th_cells = '<th class="left">Condition</th><th>N</th>'
    for d in col_display:
        th_cells += f"<th>{d}</th>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; background: #fff; }}
  h1 {{ font-size: 20px; font-weight: 500; margin-bottom: 0.25rem; }}
  .subtitle {{ font-size: 14px; color: #666; margin-bottom: 2rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #555; margin-bottom: 1.5rem; }}
  .legend span {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .chart-wrap {{ position: relative; width: 100%; height: 340px; margin-bottom: 2.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ font-size: 12px; font-weight: 500; color: #666; text-align: right; padding: 8px 12px; border-bottom: 1px solid #e5e5e5; }}
  th.left {{ text-align: left; }}
  td {{ text-align: right; padding: 8px 12px; border-bottom: 1px solid #f0f0f0; font-variant-numeric: tabular-nums; }}
  td.label {{ text-align: left; font-weight: 500; color: #1a1a1a; }}
  tr:last-child td {{ border-bottom: none; }}
  .note {{ font-size: 12px; color: #888; margin-top: 1rem; }}
  .section {{ font-size: 11px; font-weight: 500; color: #999; text-transform: uppercase; letter-spacing: 0.05em; margin: 2rem 0 0.75rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">llama3.2:3b · {sum(m["n"] for m in metrics.values())} total records · {len(conditions)} perturbation type(s)</p>

<div class="section">Metrics by perturbation type</div>
<div class="legend">{legend_html}</div>

<div class="chart-wrap">
  <canvas id="chart" role="img" aria-label="Grouped bar chart of RAG robustness metrics across perturbation types"></canvas>
</div>

<div class="section">Summary table</div>
<table>
  <thead><tr>{th_cells}</tr></thead>
  <tbody>{table_rows}</tbody>
</table>
<p class="note">Color coding: <span style="color:#1D9E75">green</span> = desirable for that metric/condition · <span style="color:#BA7517">orange</span> = borderline · <span style="color:#D85A30">red</span> = concerning. Thresholds: high-is-good metrics green ≥60%, red &lt;30%; low-is-good metrics green ≤20%, red &gt;40%.</p>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
new Chart(document.getElementById('chart'), {{
  type: 'bar',
  data: {{
    labels: {chart_labels_json},
    datasets: {datasets_json}
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y * 100).toFixed(0) + '%'
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ font: {{ size: 12 }}, maxRotation: 0 }} }},
      y: {{
        min: 0, max: 1,
        ticks: {{ callback: v => (v*100) + '%', font: {{ size: 12 }} }},
        grid: {{ color: 'rgba(0,0,0,0.06)' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>
"""


def print_summary(metrics: dict) -> None:
    conditions = [c for c in PERTURBATION_ORDER if c in metrics]
    col_w = 12
    header = f"{'Condition':<16} {'N':>4} {'Faith':>{col_w}} {'EM':>{col_w}} {'F1':>{col_w}} {'Abstain':>{col_w}} {'Halluc':>{col_w}} {'Contradict':>{col_w}}"
    print("\n" + header)
    print("-" * len(header))
    for c in conditions:
        m = metrics[c]
        label = PERTURBATION_LABELS.get(c, c)
        print(
            f"{label:<16} {m['n']:>4} "
            f"{fmt(m['faith']):>{col_w}} {fmt(m['em']):>{col_w}} {fmt(m['f1']):>{col_w}} "
            f"{fmt(m['abstain']):>{col_w}} {fmt(m['halluc']):>{col_w}} {fmt(m['contradict']):>{col_w}}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize RAG robustness evaluation results.")
    parser.add_argument("results", help="Path to results JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output HTML file (default: <results stem>_report.html)")
    parser.add_argument("--title", "-t", default="RAG robustness evaluation", help="Report title")
    args = parser.parse_args()

    records = load_results(args.results)
    print(f"Loaded {len(records)} records.")

    metrics = compute_metrics(records)
    print_summary(metrics)

    out_path = args.output or str(Path(args.results).with_suffix("")) + "_report.html"
    html = build_html(metrics, args.title)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
