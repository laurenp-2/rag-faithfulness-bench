"""
Downloads TriviaQA (rc.wikipedia split) via HuggingFace datasets and converts
it into the same benchmark format used by download_data.py.

Usage:
    python data/download_triviaqa.py [--num 500] [--seed 42]

Output: data/triviaqa_qa_pairs.json

Requirements:
    pip install datasets

Notes:
    - Uses the 'rc' (reading comprehension) configuration with Wikipedia evidence.
    - Gold context is the shortest evidence sentence containing the answer.
    - Combined dataset (SQuAD + TriviaQA) written to data/base_qa_pairs.json if
      --combine is passed.
"""

import json
import random
import argparse
from pathlib import Path


def download_triviaqa(
    num_examples: int = 500,
    seed: int = 42,
    min_context_len: int = 30,
    split: str = "train",
) -> list[dict]:
    from datasets import load_dataset

    random.seed(seed)
    print(f"Loading TriviaQA (rc.wikipedia, {split})…")
    dataset = load_dataset("trivia_qa", "rc.wikipedia", split=split, trust_remote_code=True)

    candidates = []
    for item in dataset:
        question = item["question"].strip()
        # answer is a dict with 'value' and 'aliases'
        answer_obj = item.get("answer", {})
        gold_answer = answer_obj.get("value", "").strip()
        if not gold_answer:
            continue

        # evidence passages are in item["search_results"]["search_context"] (list of strings)
        # or item["entity_pages"]["wiki_context"] for Wikipedia-based examples
        evidence_texts = []
        ep = item.get("entity_pages", {})
        if ep:
            for ctx in ep.get("wiki_context", []):
                if ctx:
                    evidence_texts.append(ctx)

        if not evidence_texts:
            sr = item.get("search_results", {})
            if sr:
                for ctx in sr.get("search_context", []):
                    if ctx:
                        evidence_texts.append(ctx)

        gold_context = ""
        for passage in evidence_texts:
            sentences = [s.strip() for s in passage.split(".") if s.strip()]
            for sentence in sentences:
                if (
                    gold_answer.lower() in sentence.lower()
                    and len(sentence.strip()) > min_context_len
                ):
                    gold_context = sentence.strip() + "."
                    break
            if gold_context:
                break

        if not gold_context:
            continue

        # Build a stable ID from the question id field
        qid = item.get("question_id", f"tqa_{len(candidates)}")
        candidates.append({
            "id": str(qid),
            "question": question,
            "gold_answer": gold_answer,
            "gold_context": gold_context,
            "source": "triviaqa_rc_wikipedia",
            "article_title": "",
        })

        if len(candidates) >= num_examples * 3:
            # Collect 3× what we need before shuffling so diversity is preserved
            break

    random.shuffle(candidates)
    qa_pairs = candidates[:num_examples]

    out_path = Path(__file__).parent / "triviaqa_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved {len(qa_pairs)} TriviaQA pairs to {out_path}")
    return qa_pairs


def combine_datasets(
    squad_path: str = "data/base_qa_pairs.json",
    triviaqa_path: str = "data/triviaqa_qa_pairs.json",
    output_path: str = "data/base_qa_pairs_combined.json",
    seed: int = 42,
) -> list[dict]:
    """Merge SQuAD and TriviaQA QA pairs, shuffle, and write to output_path."""
    random.seed(seed)

    with open(squad_path) as f:
        squad_pairs = json.load(f)
    with open(triviaqa_path) as f:
        tqa_pairs = json.load(f)

    combined = squad_pairs + tqa_pairs
    random.shuffle(combined)

    out = Path(output_path)
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)

    squad_n = sum(1 for p in combined if p.get("source", "").startswith("squad"))
    tqa_n = sum(1 for p in combined if p.get("source", "").startswith("triviaqa"))
    print(f"Combined dataset: {len(combined)} pairs ({squad_n} SQuAD + {tqa_n} TriviaQA)")
    print(f"Written to {out}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert TriviaQA")
    parser.add_argument("--num", type=int, default=500, help="Number of TriviaQA pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Merge with SQuAD (data/base_qa_pairs.json) into data/base_qa_pairs_combined.json",
    )
    parser.add_argument(
        "--squad", default="data/base_qa_pairs.json", help="Path to existing SQuAD pairs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_triviaqa(num_examples=args.num, seed=args.seed)
    if args.combine:
        combine_datasets(squad_path=args.squad, seed=args.seed)
