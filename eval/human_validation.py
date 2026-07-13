"""
Human validation of perturbation quality.

Workflow:
  1. Sample: python eval/human_validation.py sample --results results/llama3.2_3b/outputs.json
             Writes data/annotation_sample.json (50–100 examples across all perturbation types)

  2. Annotate: open data/annotation_sample.json in any text editor / spreadsheet,
               or use the terminal UI:
               python eval/human_validation.py annotate
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
import difflib
import json
import random
import re
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
# Terminal annotation UI
# ---------------------------------------------------------------------------

def annotate_terminal(
    annotations_path: str | Path = ROOT / "data" / "annotation_sample.json",
    annotator: int = 2,
    include_complete: bool = False,
    color: bool = True,
) -> None:
    """
    Prompt for fluency, correctness, and notes in the terminal.

    Saves the JSON file after every annotated example so the session can be resumed.
    """
    if annotator not in (1, 2):
        raise ValueError("annotator must be 1 or 2")

    path = Path(annotations_path)
    with open(path) as f:
        annotations = json.load(f)

    fluency_key = f"annotator_{annotator}_fluency"
    correctness_key = f"annotator_{annotator}_correctness"
    notes_key = f"annotator_{annotator}_notes"

    pending = [
        idx for idx, item in enumerate(annotations)
        if include_complete
        or item.get(fluency_key) is None
        or item.get(correctness_key) is None
    ]

    if not pending:
        print(f"All examples already have annotator {annotator} grades.")
        return

    print("\nHuman validation terminal annotator")
    print("=" * 44)
    print(f"File: {path}")
    print(f"Annotator: {annotator}")
    print(f"Items to review: {len(pending)} / {len(annotations)}")
    print("\nCommands at grade prompts: q = quit, s = skip, Enter = keep existing value")
    print("\nFluency: 1 = broken, 2 = awkward/readable, 3 = fluent")
    print("Correctness: 1 = failed, 2 = partial, 3 = clear success/faithful")

    reviewed = 0
    for position, idx in enumerate(pending, start=1):
        item = annotations[idx]
        _print_annotation_item(
            item,
            idx + 1,
            len(annotations),
            position,
            len(pending),
            color=color,
        )

        fluency = _prompt_grade("Fluency", current=item.get(fluency_key))
        if fluency == "quit":
            break
        if fluency == "skip":
            continue

        correctness = _prompt_grade("Correctness", current=item.get(correctness_key))
        if correctness == "quit":
            break
        if correctness == "skip":
            continue

        current_notes = item.get(notes_key, "") or ""
        note_prompt = "Notes"
        if current_notes:
            note_prompt += f" [{current_notes}]"
        note_prompt += ": "
        notes = input(note_prompt).strip()
        if not notes:
            notes = current_notes

        item[fluency_key] = fluency
        item[correctness_key] = correctness
        item[notes_key] = notes

        with open(path, "w") as f:
            json.dump(annotations, f, indent=2)

        reviewed += 1
        remaining = sum(
            1 for a in annotations
            if a.get(fluency_key) is None or a.get(correctness_key) is None
        )
        print(f"Saved. Remaining for annotator {annotator}: {remaining}")

    print(f"\nSession complete. Annotated {reviewed} item(s).")
    print(f"Saved to {path}")


def _print_annotation_item(
    item: dict,
    absolute_idx: int,
    total: int,
    pending_idx: int,
    pending_total: int,
    color: bool = True,
) -> None:
    original = item.get("original_context", "")
    perturbed = item.get("perturbed_context", "")
    original_display, perturbed_display, changed = _highlight_context_diff(
        original,
        perturbed,
        color=color,
    )

    print("\n" + "=" * 88)
    print(f"Item {absolute_idx}/{total}  |  Pending {pending_idx}/{pending_total}")
    print(f"Type: {item.get('perturbation_type', '')}")
    print(f"ID: {item.get('id', '')}")
    print("\nQuestion:")
    print(item.get("question", ""))
    if changed:
        print("\nOriginal context (changed text highlighted red):")
    else:
        print("\nOriginal context (unchanged):")
    print(original_display)
    if changed:
        print("\nPerturbed context (changed text highlighted green):")
    else:
        print("\nPerturbed context (unchanged):")
    print(perturbed_display)


def _highlight_context_diff(original: str, perturbed: str, color: bool = True) -> tuple[str, str, bool]:
    if original == perturbed:
        return original, perturbed, False

    original_tokens = _diff_tokens(original)
    perturbed_tokens = _diff_tokens(perturbed)
    matcher = difflib.SequenceMatcher(a=original_tokens, b=perturbed_tokens, autojunk=False)

    original_parts: list[str] = []
    perturbed_parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        original_text = "".join(original_tokens[i1:i2])
        perturbed_text = "".join(perturbed_tokens[j1:j2])

        if tag == "equal":
            original_parts.append(original_text)
            perturbed_parts.append(perturbed_text)
        elif tag == "delete":
            original_parts.append(_mark(original_text, "red", color))
        elif tag == "insert":
            perturbed_parts.append(_mark(perturbed_text, "green", color))
        elif tag == "replace":
            original_parts.append(_mark(original_text, "red", color))
            perturbed_parts.append(_mark(perturbed_text, "green", color))

    return "".join(original_parts), "".join(perturbed_parts), True


def _diff_tokens(text: str) -> list[str]:
    return re.findall(r"\s+|\S+", text)


def _mark(text: str, color_name: str, color: bool) -> str:
    if not text:
        return text
    if color:
        codes = {
            "red": "\033[1;31m",
            "green": "\033[1;32m",
        }
        return f"{codes[color_name]}{text}\033[0m"
    return f"[[{text}]]"


def _prompt_grade(label: str, current: int | None = None) -> int | str:
    while True:
        suffix = f" [{current}]" if current is not None else ""
        raw = input(f"{label} (1-3){suffix}: ").strip().lower()
        if raw == "":
            if current is not None:
                return int(current)
            print("Please enter 1, 2, 3, s, or q.")
            continue
        if raw in {"q", "quit", "exit"}:
            return "quit"
        if raw in {"s", "skip"}:
            return "skip"
        if raw in {"1", "2", "3"}:
            return int(raw)
        print("Please enter 1, 2, 3, s, or q.")


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

    ap = sub.add_parser("annotate", help="Grade annotation examples in the terminal")
    ap.add_argument("--annotations", default=str(ROOT / "data" / "annotation_sample.json"))
    ap.add_argument("--annotator", type=int, choices=[1, 2], default=2)
    ap.add_argument(
        "--include-complete",
        action="store_true",
        help="Review all examples, including ones already graded by this annotator",
    )
    ap.add_argument(
        "--no-color",
        action="store_true",
        help="Use bracket markers instead of terminal colors for changed spans",
    )

    sub.add_parser("rubric", help="Print annotation rubric")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "sample":
        sample_for_annotation(args.results, args.output, n_per_type=args.n, seed=args.seed)
    elif args.command == "report":
        report(args.annotations, args.output)
    elif args.command == "annotate":
        annotate_terminal(
            annotations_path=args.annotations,
            annotator=args.annotator,
            include_complete=args.include_complete,
            color=not args.no_color,
        )
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
