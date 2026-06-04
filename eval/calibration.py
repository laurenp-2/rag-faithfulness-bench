"""
Abstention calibration analysis.

Computes and plots calibration curves: for each bin of faithfulness score,
what fraction of examples did the model actually abstain?

A well-calibrated model should abstain more as faithfulness score decreases
(i.e. when the context is less supportive of a confident answer).

Also computes Expected Calibration Error (ECE) — borrowed from the
probability calibration literature — adapted for the abstention task:
  ECE = Σ_b (|b| / N) * |mean_faith(b) − abstain_rate(b)|

Usage:
    python eval/calibration.py --results results/llama3.2_3b/outputs.json
    python eval/calibration.py --results results/  --all-models
    python eval/calibration.py --results results/llama3.2_3b/outputs.json --plot fig_calibration.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent.parent
PERTURBATION_TYPES = ["original", "entity_swap", "negation", "paraphrase"]


# ---------------------------------------------------------------------------
# Core calibration computation
# ---------------------------------------------------------------------------

def compute_calibration(
    records: list[dict],
    n_bins: int = 10,
    perturbation_filter: list[str] | None = None,
) -> dict:
    """
    Compute calibration statistics for a set of benchmark records.

    Args:
        records: list of result dicts (from outputs.json)
        n_bins: number of equal-width bins over [0, 1] faithfulness range
        perturbation_filter: if set, only include these perturbation types

    Returns:
        {
          "bins": [{"lower": float, "upper": float, "n": int,
                    "mean_faithfulness": float, "abstention_rate": float}, ...],
          "ece": float,
          "n_total": int,
        }
    """
    if perturbation_filter:
        records = [r for r in records if r["perturbation_type"] in perturbation_filter]

    bins: list[dict] = []
    bin_width = 1.0 / n_bins
    bin_buckets: list[list[dict]] = [[] for _ in range(n_bins)]

    for r in records:
        faith = r["scores"]["faithfulness_score"]
        bin_idx = min(int(faith / bin_width), n_bins - 1)
        bin_buckets[bin_idx].append(r)

    n_total = len(records)
    ece = 0.0

    for i, bucket in enumerate(bin_buckets):
        lower = i * bin_width
        upper = lower + bin_width
        n = len(bucket)
        if n == 0:
            bins.append({"lower": round(lower, 3), "upper": round(upper, 3),
                         "n": 0, "mean_faithfulness": None, "abstention_rate": None})
            continue

        mean_faith = sum(r["scores"]["faithfulness_score"] for r in bucket) / n
        abstain_rate = sum(1 for r in bucket if r["generation"]["abstained"]) / n

        bins.append({
            "lower": round(lower, 3),
            "upper": round(upper, 3),
            "n": n,
            "mean_faithfulness": round(mean_faith, 4),
            "abstention_rate": round(abstain_rate, 4),
        })

        # ECE contribution: weight by fraction of total examples in this bin
        # ideal: abstain when faithfulness is low, answer when faithfulness is high
        # "ideal abstention" = 1 - mean_faithfulness
        ideal_abstain = 1.0 - mean_faith
        ece += (n / n_total) * abs(ideal_abstain - abstain_rate)

    return {
        "bins": bins,
        "ece": round(ece, 4),
        "n_total": n_total,
    }


def compute_calibration_by_perturbation(records: list[dict], n_bins: int = 10) -> dict:
    """Compute calibration separately per perturbation type."""
    return {
        ptype: compute_calibration(
            [r for r in records if r["perturbation_type"] == ptype],
            n_bins=n_bins,
        )
        for ptype in PERTURBATION_TYPES
        if any(r["perturbation_type"] == ptype for r in records)
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    calibration_by_model: dict[str, dict],
    output_path: str | Path,
    title: str = "Abstention Calibration Curves",
) -> None:
    """
    Plot calibration curves for one or more models.

    calibration_by_model: { model_name: compute_calibration(...) result }
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["#4e8098", "#c44d34", "#e8a838", "#6b4c9a", "#1D9E75"]

    for (model_name, cal_data), color in zip(calibration_by_model.items(), colors):
        bins = [b for b in cal_data["bins"] if b["n"] > 0]
        if not bins:
            continue
        x = [b["mean_faithfulness"] for b in bins]
        y = [b["abstention_rate"] for b in bins]
        sizes = [max(20, b["n"] * 3) for b in bins]

        ax.plot(x, y, "o-", color=color, label=f"{model_name} (ECE={cal_data['ece']:.3f})",
                linewidth=1.8, markersize=5, zorder=3)
        ax.scatter(x, y, s=sizes, color=color, alpha=0.3, zorder=2)

    # Ideal calibration line: abstain rate = 1 - faithfulness
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, 1 - xs, "--", color="#999", linewidth=1.2, label="Ideal (abstain = 1−faith)",
            zorder=1)

    ax.set_xlabel("Mean faithfulness score", fontsize=11)
    ax.set_ylabel("Abstention rate", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Calibration curve saved to {out}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_calibration_summary(model_name: str, cal_data: dict) -> None:
    print(f"\n{'='*55}")
    print(f"Calibration — {model_name}  (n={cal_data['n_total']}, ECE={cal_data['ece']:.4f})")
    print(f"{'='*55}")
    print(f"  {'Faith bin':>12}  {'N':>6}  {'Mean faith':>12}  {'Abstain rate':>13}")
    print(f"  {'-'*55}")
    for b in cal_data["bins"]:
        if b["n"] == 0:
            continue
        print(
            f"  [{b['lower']:.1f}–{b['upper']:.1f}]"
            f"  {b['n']:>6}"
            f"  {b['mean_faithfulness']:>12.3f}"
            f"  {b['abstention_rate']:>13.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_single(results_path: str | Path, plot_path: str | Path | None, n_bins: int = 10) -> dict:
    with open(results_path) as f:
        records = json.load(f)

    model_name = Path(results_path).parent.name
    cal = compute_calibration(records, n_bins=n_bins)
    print_calibration_summary(model_name, cal)

    report = {
        "model": model_name,
        "overall": cal,
        "by_perturbation": compute_calibration_by_perturbation(records, n_bins=n_bins),
    }

    report_path = Path(results_path).parent / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Calibration report written to {report_path}")

    if plot_path:
        plot_calibration_curves({model_name: cal}, output_path=plot_path)

    return report


def run_all_models(
    results_dir: str | Path = ROOT / "results",
    plot_path: str | Path | None = None,
    n_bins: int = 10,
) -> None:
    results_dir = Path(results_dir)
    all_cal: dict[str, dict] = {}

    for model_dir in sorted(results_dir.iterdir()):
        outputs = model_dir / "outputs.json"
        if not outputs.exists():
            continue
        with open(outputs) as f:
            records = json.load(f)
        model_name = model_dir.name
        cal = compute_calibration(records, n_bins=n_bins)
        all_cal[model_name] = cal
        print_calibration_summary(model_name, cal)

    if plot_path and all_cal:
        plot_calibration_curves(all_cal, output_path=plot_path, title="Abstention Calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Abstention calibration analysis")
    parser.add_argument("--results", required=True,
                        help="Path to outputs.json or results/ directory (with --all-models)")
    parser.add_argument("--all-models", action="store_true",
                        help="Analyze all model directories under results/")
    parser.add_argument("--plot", default=None, help="Save calibration curve plot to this path")
    parser.add_argument("--bins", type=int, default=10, help="Number of calibration bins")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all_models:
        run_all_models(results_dir=args.results, plot_path=args.plot, n_bins=args.bins)
    else:
        run_single(args.results, plot_path=args.plot, n_bins=args.bins)
