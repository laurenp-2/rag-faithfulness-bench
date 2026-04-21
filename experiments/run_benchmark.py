"""
Main benchmark runner.

For each QA pair in data/base_qa_pairs.json:
  - Apply all four perturbation types (original, entity_swap, negation, paraphrase)
  - Build a FAISS index over all four context variants
  - Retrieve top-k contexts for the question
  - Generate an answer via Claude
  - Score faithfulness + correctness
  - Write results to results/outputs.json

Usage:
    python experiments/run_benchmark.py \
        [--data data/base_qa_pairs.json] \
        [--output results/outputs.json] \
        [--model llama3.2:3b] \
        [--k 3] \
        [--limit 20]       # for quick smoke-tests
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from tqdm import tqdm

# Allow running from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from perturbations.entity_swap import swap_entities
from perturbations.negation import negate_context
from perturbations.paraphrase import paraphrase_context
from pipeline.retriever import Retriever
from pipeline.generator import generate_answer
from eval.faithfulness_scorer import FaithfulnessScorer
from eval.metrics import compute_summary, print_summary_table


def apply_perturbations(gold_context: str, model: str) -> dict[str, str]:
    """Return all four context variants for a single gold context."""
    return {
        "original": gold_context,
        "entity_swap": swap_entities(gold_context),
        "negation": negate_context(gold_context),
        "paraphrase": paraphrase_context(gold_context, model=model),
    }


def run_benchmark(
    data_path: str | Path,
    output_path: str | Path,
    model: str = "llama3.2:3b",
    k: int = 3,
    limit: int | None = None,
) -> list[dict]:
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        qa_pairs: list[dict] = json.load(f)

    if limit:
        qa_pairs = qa_pairs[:limit]

    scorer = FaithfulnessScorer()
    retriever = Retriever()
    all_results: list[dict] = []

    for item in tqdm(qa_pairs, desc="QA pairs"):
        question = item["question"]
        gold_answer = item["gold_answer"]
        gold_context = item["gold_context"]


        try:
            variants = apply_perturbations(gold_context, model=model)
        except Exception as e:
            print(f"[perturb] Skipping '{question[:60]}': {e}")
            continue

        for ptype, perturbed_ctx in variants.items():
            # Retrieve: build an index of just this variant (simulates a
            # real RAG pipeline that indexed one version of the document)
            retriever.build_index([perturbed_ctx])
            retrieved = retriever.retrieve(question, k=min(k, 1))

            # Generate
            try:
                gen = generate_answer(question, retrieved, model=model)
            except Exception as e:
                print(f"[generate] Error on '{question[:40]}' ({ptype}): {e}")
                continue

            # Score
            try:
                scores = scorer.score(
                    generated_answer=gen["answer"],
                    perturbed_context=perturbed_ctx,
                    gold_answer=gold_answer,
                    gold_context=gold_context,
                )
            except Exception as e:
                print(f"[score] Error on '{question[:40]}' ({ptype}): {e}")
                continue

            result: dict[str, Any] = {
                "id": item.get("id", ""),
                "question": question,
                "gold_answer": gold_answer,
                "gold_context": gold_context,
                "perturbation_type": ptype,
                "perturbed_context": perturbed_ctx,
                "retrieved_contexts": retrieved,
                "generation": gen,
                "scores": scores,
            }
            all_results.append(result)


    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {output_path}")


    summary = compute_summary(all_results)
    print_summary_table(summary)

    summary_path = output_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG faithfulness benchmark runner")
    parser.add_argument("--data", default="data/base_qa_pairs.json")
    parser.add_argument("--output", default=None,
                        help="Output path (default: results/<model>/outputs.json)")
    parser.add_argument("--model", default="llama3.2:3b",
                        help="Ollama model tag, e.g. llama3.1:8b, mistral:7b, qwen2.5:7b")
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Limit QA pairs (for testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Default output path namespaced by model so runs don't overwrite each other
    if args.output is None:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        output_path = ROOT / "results" / safe_model / "outputs.json"
    else:
        output_path = ROOT / args.output
    run_benchmark(
        data_path=ROOT / args.data,
        output_path=output_path,
        model=args.model,
        k=args.k,
        limit=args.limit,
    )
