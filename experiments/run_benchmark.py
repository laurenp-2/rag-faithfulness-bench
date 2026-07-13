"""
Main benchmark runner.

For each QA pair in data/base_qa_pairs.json:
  - Apply all four perturbation types (original, entity_swap, negation, paraphrase)
  - Build a FAISS index over the context variant
  - Retrieve top-k contexts (with similarity scores) for the question
  - Generate an answer via the configured model backend
  - Score faithfulness + correctness
  - Write results to results/<model>/outputs.json

Supported model backends (selected automatically by prefix):
  - Ollama (local):   llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b …
  - OpenAI API:       gpt-4o-mini, gpt-4o …   (OPENAI_API_KEY env var)
  - Anthropic API:    claude-haiku-4-5-20251001 …  (ANTHROPIC_API_KEY env var)

Usage:
    python experiments/run_benchmark.py \\
        [--data data/base_qa_pairs.json] \\
        [--output results/outputs.json] \\
        [--model llama3.2:3b] \\
        [--k 3] \\
        [--limit 20] \\
        [--include-similarity]   # supply FAISS score as prompt prefix
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from perturbations.entity_swap import swap_entities
from perturbations.negation import negate_context
from perturbations.paraphrase import paraphrase_context
from pipeline.retriever import Retriever
from pipeline.generator import generate_answer
from eval.faithfulness_scorer import FaithfulnessScorer
from eval.metrics import compute_summary, print_summary_table


def _is_api_model(model: str) -> bool:
    """Return True for models that call an external API (no Ollama paraphrase needed)."""
    return (
        model.startswith("gpt-")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("claude-")
    )


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _handle_error(stage: str, question: str, ptype: str | None, exc: Exception, skip_errors: bool) -> None:
    location = f" ({ptype})" if ptype else ""
    message = f"[{stage}] Error on '{question[:60]}'{location}: {exc}"
    if skip_errors:
        print(message)
        return
    raise RuntimeError(message) from exc


def apply_perturbations(gold_context: str, model: str) -> dict[str, str]:
    """Return all four context variants for a single gold context."""
    # Use llama3.2:3b for paraphrase when the benchmark model is a closed-source API
    paraphrase_model = "llama3.2:3b" if _is_api_model(model) else model
    return {
        "original": gold_context,
        "entity_swap": swap_entities(gold_context),
        "negation": negate_context(gold_context),
        "paraphrase": paraphrase_context(gold_context, model=paraphrase_model),
    }


def run_benchmark(
    data_path: str | Path,
    output_path: str | Path,
    model: str = "llama3.2:3b",
    k: int = 3,
    limit: int | None = None,
    include_similarity: bool = False,
    skip_errors: bool = False,
    save_every: int = 25,
) -> list[dict]:
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")

    with open(data_path) as f:
        qa_pairs: list[dict] = json.load(f)

    if limit is not None:
        qa_pairs = qa_pairs[:limit]

    if not qa_pairs:
        _write_json_atomic(output_path, [])
        summary = compute_summary([])
        summary_path = output_path.parent / "summary.json"
        _write_json_atomic(summary_path, summary)
        print(f"\nNo QA pairs to run. Empty results written to {output_path}")
        print(f"Summary written to {summary_path}")
        return []

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
            _handle_error("perturb", question, None, e, skip_errors)
            continue

        for ptype, perturbed_ctx in variants.items():
            retriever.build_index([perturbed_ctx])

            if include_similarity:
                retrieved_with_scores = retriever.retrieve_with_scores(question, k=min(k, 1))
                retrieved = [ctx for ctx, _ in retrieved_with_scores]
                top_score = retrieved_with_scores[0][1] if retrieved_with_scores else None
            else:
                retrieved = retriever.retrieve(question, k=min(k, 1))
                top_score = None

            try:
                gen = generate_answer(
                    question,
                    retrieved,
                    model=model,
                    similarity_score=top_score if include_similarity else None,
                )
            except Exception as e:
                _handle_error("generate", question, ptype, e, skip_errors)
                continue

            try:
                scores = scorer.score(
                    generated_answer=gen["answer"],
                    perturbed_context=perturbed_ctx,
                    gold_answer=gold_answer,
                    gold_context=gold_context,
                )
            except Exception as e:
                _handle_error("score", question, ptype, e, skip_errors)
                continue

            result: dict[str, Any] = {
                "id": item.get("id", ""),
                "question": question,
                "gold_answer": gold_answer,
                "gold_context": gold_context,
                "source": item.get("source", "squad_v1.1"),
                "perturbation_type": ptype,
                "perturbed_context": perturbed_ctx,
                "retrieved_contexts": retrieved,
                "retrieval_similarity_score": top_score,
                "generation": gen,
                "scores": scores,
            }
            all_results.append(result)
            if len(all_results) % save_every == 0:
                _write_json_atomic(output_path, all_results)

    _write_json_atomic(output_path, all_results)
    print(f"\nResults written to {output_path}")

    summary = compute_summary(all_results)
    print_summary_table(summary)

    summary_path = output_path.parent / "summary.json"
    _write_json_atomic(summary_path, summary)
    print(f"Summary written to {summary_path}")

    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG faithfulness benchmark runner")
    parser.add_argument("--data", default="data/base_qa_pairs.json")
    parser.add_argument("--output", default=None,
                        help="Output path (default: results/<model>/outputs.json)")
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help=(
            "Model to use. Ollama: llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b. "
            "OpenAI: gpt-4o-mini (needs OPENAI_API_KEY). "
            "Anthropic: claude-haiku-4-5-20251001 (needs ANTHROPIC_API_KEY)."
        ),
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Limit QA pairs (for testing)")
    parser.add_argument(
        "--include-similarity",
        action="store_true",
        help="Supply FAISS similarity score as a prompt prefix (Section 5.6 experiment)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue after per-example errors. By default, errors fail fast to avoid biased results.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Write an atomic partial outputs.json checkpoint every N completed records.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output is None:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        suffix = "_sim" if args.include_similarity else ""
        output_path = ROOT / "results" / f"{safe_model}{suffix}" / "outputs.json"
    else:
        output_path = ROOT / args.output
    run_benchmark(
        data_path=ROOT / args.data,
        output_path=output_path,
        model=args.model,
        k=args.k,
        limit=args.limit,
        include_similarity=args.include_similarity,
        skip_errors=args.skip_errors,
        save_every=args.save_every,
    )
