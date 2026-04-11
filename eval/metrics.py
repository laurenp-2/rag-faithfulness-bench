"""
Aggregate metrics over the full benchmark run.

Input: the list of result dicts written to results/outputs.json
Output: a summary table (printed and returned as a dict)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


PERTURBATION_TYPES = ["original", "entity_swap", "negation", "paraphrase"]


def compute_summary(results: list[dict]) -> dict:
    """
    Compute per-perturbation-type aggregate metrics.

    Returns a nested dict:
        {perturbation_type: {metric_name: value, ...}, ...}
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r["perturbation_type"]].append(r)

    summary = {}
    for ptype in PERTURBATION_TYPES:
        items = buckets.get(ptype, [])
        if not items:
            continue
        n = len(items)
        scores = [i["scores"] for i in items]

        faithfulness_rate = _mean(s["faithfulness_score"] for s in scores)
        correctness_em = _mean(s["correctness_em"] for s in scores)
        correctness_f1 = _mean(s["correctness_f1"] for s in scores)
        hallucination_rate = _mean(s["hallucination_flag"] for s in scores)
        abstention_rate = _mean(i["generation"]["abstained"] for i in items)
        contradiction_rate = _mean(s["context_contradiction"] for s in scores)

        summary[ptype] = {
            "n": n,
            "faithfulness_rate": round(faithfulness_rate, 4),
            "correctness_em": round(correctness_em, 4),
            "correctness_f1": round(correctness_f1, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "abstention_rate": round(abstention_rate, 4),
            "contradiction_rate": round(contradiction_rate, 4),
        }

    return summary


def print_summary_table(summary: dict) -> None:
    """Pretty-print the summary as a markdown-style table."""
    cols = [
        ("Perturbation", "perturbation"),
        ("N", "n"),
        ("Faith↑", "faithfulness_rate"),
        ("EM↑", "correctness_em"),
        ("F1↑", "correctness_f1"),
        ("Halluc↓", "hallucination_rate"),
        ("Abstain↑", "abstention_rate"),
        ("Contradict↓", "contradiction_rate"),
    ]
    header = " | ".join(f"{c[0]:>12}" for c in cols)
    sep = "-+-".join("-" * 12 for _ in cols)
    print(header)
    print(sep)
    for ptype in PERTURBATION_TYPES:
        if ptype not in summary:
            continue
        row = summary[ptype]
        values = [ptype] + [str(row.get(c[1], "")) for c in cols[1:]]
        print(" | ".join(f"{v:>12}" for v in values))


def load_and_summarise(results_path: str | Path) -> dict:
    with open(results_path) as f:
        results = json.load(f)
    summary = compute_summary(results)
    print_summary_table(summary)
    return summary


def _mean(values) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(float(v) for v in vals) / len(vals)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/outputs.json"
    load_and_summarise(path)
