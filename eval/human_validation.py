"""
Human validation of perturbation quality.

Workflow:
  1. Sample: python eval/human_validation.py sample --results results/llama3.2_3b/outputs.json
             Writes data/annotation_sample.json (50–100 examples across all perturbation types)

  2. Annotate: open data/annotation_sample.json in any text editor / spreadsheet.
               Fill in annotator_1_fluency, annotator_1_correctness fields.
               A second annotator fills in annotator_2_fluency, annotator_2_correctness.
               Scale: 1 = poor, 2 = acceptable, 3 = good.

  3. Report: python eval/human_validation.py report --annotations data/annotation_sample.json
             Prints Cohen's kappa and percentage rated fluent / factually correct.
             Writes eval/human_validation_report.json.

Rubric (also printed by `rubric` subcommand):
  Fluency (1–3):
    1 — Grammatically broken or unnatural
    2 — Slightly awkward but readable
    3 — Fluent and natural
  Correctness of perturbation intent (1–3):
    1 — Perturbation failed (e.g. entity swap produced same entity, negation is ambiguous)
    2 — Partial success (e.g. negation inserted but semantics unclear)
    3 — Clear success (perturbation clearly achieves its intended distortion)
  Note: for "original" and "paraphrase" conditions, correctness means the context
        still faithfully expresses the original fact.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent.parent

RUBRIC = {
    "fluency": {
        1: "Grammatically broken or unnatural",
        2: "Slightly awkward but readable",
        3: "Fluent and natural",
    },
    "correctness": {
        1: "Perturbation failed",
        2: "Partial success",
        3: "Clear success (or faithful for original/paraphrase)",
    },
}

PERTURBATION_TYPES = ["original", "entity_swap", "negation", "paraphrase"]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_for_annotation(
    results_path: str | Path,
    output_path: str | Path = ROOT / "data" / "annotation_sample.json",
    n_per_type: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Draw n_per_type examples from each perturbation type for human annotation.
    Target: ~80 examples total (4 types × 20 each).
    """
    random.seed(seed)

    with open(results_path) as f:
        results = json.load(f)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["perturbation_type"]].append(r)

    sample = []
    for ptype in PERTURBATION_TYPES:
        pool = by_type.get(ptype, [])
        chosen = random.sample(pool, min(n_per_type, len(pool)))
        for r in chosen:
            sample.append({
                "id": r.get("id", ""),
                "perturbation_type": ptype,
                "original_context": r["gold_context"],
                "perturbed_context": r["perturbed_context"],
                "question": r["question"],
                # Fields annotators should fill in
                "annotator_1_fluency": None,
                "annotator_1_correctness": None,
                "annotator_2_fluency": None,
                "annotator_2_correctness": None,
                "annotator_1_notes": "",
                "annotator_2_notes": "",
            })

    random.shuffle(sample)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Sampled {len(sample)} examples to {out}")
    print("  Distribution:")
    by_t: dict[str, int] = defaultdict(int)
    for s in sample:
        by_t[s["perturbation_type"]] += 1
    for pt, cnt in by_t.items():
        print(f"    {pt}: {cnt}")
    return sample


# ---------------------------------------------------------------------------
# Cohen's kappa computation
# ---------------------------------------------------------------------------

def cohen_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Compute Cohen's kappa for two lists of ordinal ratings."""
    if len(ratings_a) != len(ratings_b):
        raise ValueError("Rating lists must have the same length")
    n = len(ratings_a)
    if n == 0:
        return float("nan")

    categories = sorted(set(ratings_a) | set(ratings_b))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Confusion matrix
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        conf[cat_idx[a]][cat_idx[b]] += 1

    po = sum(conf[i][i] for i in range(k)) / n

    row_totals = [sum(conf[i]) for i in range(k)]
    col_totals = [sum(conf[r][i] for r in range(k)) for i in range(k)]
    pe = sum(row_totals[i] * col_totals[i] for i in range(k)) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(
    annotations_path: str | Path,
    output_path: str | Path = ROOT / "eval" / "human_validation_report.json",
) -> dict:
    with open(annotations_path) as f:
        annotations = json.load(f)

    # Filter to fully annotated items
    complete = [
        a for a in annotations
        if a["annotator_1_fluency"] is not None
        and a["annotator_2_fluency"] is not None
        and a["annotator_1_correctness"] is not None
        and a["annotator_2_correctness"] is not None
    ]

    if not complete:
        print("No fully annotated examples found. Fill in annotator fields first.")
        return {}

    flu_a = [int(a["annotator_1_fluency"]) for a in complete]
    flu_b = [int(a["annotator_2_fluency"]) for a in complete]
    cor_a = [int(a["annotator_1_correctness"]) for a in complete]
    cor_b = [int(a["annotator_2_correctness"]) for a in complete]

    kappa_fluency = cohen_kappa(flu_a, flu_b)
    kappa_correctness = cohen_kappa(cor_a, cor_b)

    # Average ratings per perturbation type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for a in complete:
        by_type[a["perturbation_type"]].append(a)

    per_type_stats = {}
    for ptype in PERTURBATION_TYPES:
        items = by_type.get(ptype, [])
        if not items:
            continue
        avg_flu = sum(
            (i["annotator_1_fluency"] + i["annotator_2_fluency"]) / 2 for i in items
        ) / len(items)
        avg_cor = sum(
            (i["annotator_1_correctness"] + i["annotator_2_correctness"]) / 2 for i in items
        ) / len(items)
        # Fluent = mean rating >= 2
        pct_fluent = sum(
            1 for i in items
            if (i["annotator_1_fluency"] + i["annotator_2_fluency"]) / 2 >= 2
        ) / len(items)
        pct_correct = sum(
            1 for i in items
            if (i["annotator_1_correctness"] + i["annotator_2_correctness"]) / 2 >= 2
        ) / len(items)
        per_type_stats[ptype] = {
            "n": len(items),
            "avg_fluency": round(avg_flu, 3),
            "avg_correctness": round(avg_cor, 3),
            "pct_fluent": round(pct_fluent, 3),
            "pct_correct_intent": round(pct_correct, 3),
        }

    report_data = {
        "n_annotated": len(complete),
        "cohen_kappa_fluency": round(kappa_fluency, 4),
        "cohen_kappa_correctness": round(kappa_correctness, 4),
        "per_type": per_type_stats,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nHuman Validation Report ({len(complete)} examples)")
    print(f"  Cohen's κ (fluency):     {kappa_fluency:.4f}")
    print(f"  Cohen's κ (correctness): {kappa_correctness:.4f}")
    print(f"\n  Per-type stats:")
    for ptype, stats in per_type_stats.items():
        print(
            f"    {ptype:12s}: n={stats['n']}  "
            f"fluent={stats['pct_fluent']:.0%}  "
            f"correct_intent={stats['pct_correct_intent']:.0%}"
        )
    print(f"\nReport written to {out}")
    return report_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human validation of perturbation quality")
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("sample", help="Sample examples for annotation")
    sp.add_argument("--results", required=True, help="Path to outputs.json")
    sp.add_argument("--output", default=str(ROOT / "data" / "annotation_sample.json"))
    sp.add_argument("--n", type=int, default=20, help="Examples per perturbation type")
    sp.add_argument("--seed", type=int, default=42)

    rp = sub.add_parser("report", help="Compute kappa and print stats")
    rp.add_argument("--annotations", default=str(ROOT / "data" / "annotation_sample.json"))
    rp.add_argument("--output", default=str(ROOT / "eval" / "human_validation_report.json"))

    sub.add_parser("rubric", help="Print annotation rubric")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "sample":
        sample_for_annotation(args.results, args.output, n_per_type=args.n, seed=args.seed)
    elif args.command == "report":
        report(args.annotations, args.output)
    elif args.command == "rubric":
        print("\nAnnotation Rubric")
        print("=" * 40)
        for dim, levels in RUBRIC.items():
            print(f"\n{dim.capitalize()} (1–3):")
            for score, desc in levels.items():
                print(f"  {score} — {desc}")
        print()
    else:
        parse_args().print_help()
