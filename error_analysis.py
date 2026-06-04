"""
Error analysis for the RAG faithfulness benchmark.

Produces two analyses:
  1. Failure-mode examples (context-following failures and abstention failures)
  2. Cross-tabulation of failure mode by answer type (person, org, date/number, other)
     — answers entity-swap failure-rate questions like "are date answers more susceptible?"

Usage:
    python error_analysis.py --results results/qwen2.5_7b/outputs.json
    python error_analysis.py --results results/qwen2.5_7b/outputs.json --report error_report.json
    python error_analysis.py --results results/qwen2.5_7b/outputs.json --all-models
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import spacy

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Answer-type detection
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def classify_answer_type(answer: str) -> str:
    """
    Return one of: "person", "org", "date_number", "location", "other".

    Heuristic: run NER on the answer text; use the first entity label.
    Falls back to "date_number" for numeric/year-looking strings.
    """
    nlp = _get_nlp()
    doc = nlp(answer.strip())

    for ent in doc.ents:
        label = ent.label_
        if label == "PERSON":
            return "person"
        if label in ("ORG", "PRODUCT", "WORK_OF_ART", "LAW", "EVENT"):
            return "org"
        if label in ("GPE", "LOC", "FAC"):
            return "location"
        if label in ("DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"):
            return "date_number"

    # Fallback: purely numeric or year-like
    stripped = answer.strip()
    if stripped.replace(",", "").replace(".", "").replace("-", "").replace(" ", "").isdigit():
        return "date_number"

    return "other"


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

ANSWER_TYPES = ["person", "org", "location", "date_number", "other"]
PERTURBATION_TYPES = ["original", "entity_swap", "negation", "paraphrase"]


def load_results(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def annotate_answer_types(records: list[dict]) -> list[dict]:
    """Add 'answer_type' field to each record in place."""
    for r in records:
        if "answer_type" not in r:
            r["answer_type"] = classify_answer_type(r["gold_answer"])
    return records


def cross_tabulate_failures(records: list[dict]) -> dict:
    """
    Cross-tabulate failure modes by answer type and perturbation type.

    For each (perturbation_type, answer_type) cell, compute:
      - n: total examples
      - context_follower_rate: abstained=False AND faithfulness>0.7 AND EM=False
      - abstention_rate: fraction that abstained
      - hallucination_rate: fraction with hallucination_flag=True
      - correctness_f1_mean: mean token F1
    """
    results: dict[str, dict[str, dict]] = {}

    for ptype in PERTURBATION_TYPES:
        results[ptype] = {}
        ptype_records = [r for r in records if r["perturbation_type"] == ptype]

        for atype in ANSWER_TYPES:
            bucket = [r for r in ptype_records if r.get("answer_type") == atype]
            n = len(bucket)
            if n == 0:
                continue

            abstained = [r["generation"]["abstained"] for r in bucket]
            faithfulness = [r["scores"]["faithfulness_score"] for r in bucket]
            em = [r["scores"]["correctness_em"] for r in bucket]
            f1 = [r["scores"]["correctness_f1"] for r in bucket]
            halluc = [r["scores"]["hallucination_flag"] for r in bucket]

            # Context-follower: model believed the perturbed context (didn't abstain,
            # high faithfulness, but wrong answer)
            context_follower_rate = sum(
                1
                for i in range(n)
                if not abstained[i] and faithfulness[i] > 0.7 and not em[i]
            ) / n

            results[ptype][atype] = {
                "n": n,
                "abstention_rate": round(sum(abstained) / n, 4),
                "context_follower_rate": round(context_follower_rate, 4),
                "hallucination_rate": round(sum(halluc) / n, 4),
                "correctness_f1_mean": round(sum(f1) / n, 4),
                "faithfulness_mean": round(sum(faithfulness) / n, 4),
            }

    return results


def print_cross_tab(cross_tab: dict) -> None:
    """Print a formatted cross-tabulation table."""
    col_w = 14
    for ptype in PERTURBATION_TYPES:
        if ptype not in cross_tab:
            continue
        print(f"\n=== Perturbation: {ptype} ===")
        header = f"{'Answer type':<14}" + "".join(
            f"{'N':>{col_w}} {'AbsRate':>{col_w}} {'CtxFollow':>{col_w}} {'Halluc':>{col_w}} {'F1':>{col_w}}"
        )
        print(header)
        print("-" * len(header))
        for atype in ANSWER_TYPES:
            stats = cross_tab.get(ptype, {}).get(atype)
            if stats is None:
                continue
            print(
                f"{atype:<14}"
                f"{stats['n']:>{col_w}}"
                f"{stats['abstention_rate']:>{col_w}.3f}"
                f"{stats['context_follower_rate']:>{col_w}.3f}"
                f"{stats['hallucination_rate']:>{col_w}.3f}"
                f"{stats['correctness_f1_mean']:>{col_w}.3f}"
            )


def show_failure_examples(records: list[dict], n_samples: int = 5, seed: int = 42) -> None:
    """Print qualitative examples of context-follower and abstention failures."""
    random.seed(seed)

    for ptype in ["entity_swap", "negation"]:
        followers = [
            r for r in records
            if r["perturbation_type"] == ptype
            and not r["generation"]["abstained"]
            and not r["scores"]["correctness_em"]
            and r["scores"]["faithfulness_score"] > 0.7
        ]

        print(f"\n{'='*60}")
        print(f"Context-follower failures — {ptype} (n={len(followers)})")
        print(f"{'='*60}")
        for r in random.sample(followers, min(n_samples, len(followers))):
            print(f"  Q:            {r['question']}")
            print(f"  Context:      {r['perturbed_context']}")
            print(f"  Answer:       {r['generation']['answer']}")
            print(f"  Gold:         {r['gold_answer']}")
            print(f"  Faith:        {r['scores']['faithfulness_score']:.3f}")
            print(f"  Answer type:  {r.get('answer_type', 'unknown')}")
            print()


def analyze(
    results_path: str | Path,
    report_path: str | Path | None = None,
    n_samples: int = 5,
) -> dict:
    records = load_results(results_path)
    records = annotate_answer_types(records)

    cross_tab = cross_tabulate_failures(records)
    print_cross_tab(cross_tab)
    show_failure_examples(records, n_samples=n_samples)

    report = {
        "results_path": str(results_path),
        "n_records": len(records),
        "cross_tabulation": cross_tab,
    }

    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nError analysis written to {out}")

    return report


def analyze_all_models(results_dir: str | Path = ROOT / "results") -> None:
    """Run error analysis across all model result directories."""
    results_dir = Path(results_dir)
    for model_dir in sorted(results_dir.iterdir()):
        outputs = model_dir / "outputs.json"
        if not outputs.exists():
            continue
        print(f"\n{'#'*60}")
        print(f"# Model: {model_dir.name}")
        print(f"{'#'*60}")
        report_path = model_dir / "error_analysis.json"
        analyze(outputs, report_path=report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG faithfulness error analysis")
    parser.add_argument("--results", default="results/qwen2.5_7b/outputs.json")
    parser.add_argument("--report", default=None, help="Write JSON report to this path")
    parser.add_argument("--samples", type=int, default=5, help="Qualitative examples per type")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Analyze all model directories under results/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all_models:
        analyze_all_models()
    else:
        analyze(args.results, report_path=args.report, n_samples=args.samples)
