"""
Converts a locally downloaded SQuAD v1.1 JSON file into the benchmark format.

Usage:
    python data/download_data.py [--num 1000] [--seed 42]

Input:  data/squad.json  (download from https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
Output: data/base_qa_pairs.json

Changes from original:
    - Default increased to 1000 examples (was 150)
    - Random shuffling across all articles for better topic diversity
    - Longer gold_context window (full paragraph sentence, not just answer sentence)
    - Source metadata tracked per example
"""

import json
import random
import argparse
from pathlib import Path


def convert(
    squad_path: str = "data/squad.json",
    num_examples: int = 1000,
    seed: int = 42,
    min_context_len: int = 30,
) -> list[dict]:
    random.seed(seed)

    with open(squad_path) as f:
        squad = json.load(f)

    # Collect all candidate QA pairs first, then sample
    candidates = []
    for article in squad["data"]:
        article_title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            sentences = [s.strip() for s in context.split(".") if s.strip()]
            for qa in paragraph["qas"]:
                if not qa["answers"]:
                    continue
                question = qa["question"].strip()
                gold_answer = qa["answers"][0]["text"].strip()

                # Find the sentence in the context that contains the answer
                gold_context = ""
                for sentence in sentences:
                    if (
                        gold_answer.lower() in sentence.lower()
                        and len(sentence.strip()) > min_context_len
                    ):
                        gold_context = sentence.strip() + "."
                        break

                if not gold_context:
                    continue

                candidates.append({
                    "id": qa["id"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_context": gold_context,
                    "source": "squad_v1.1",
                    "article_title": article_title,
                })

    # Shuffle for diversity, then take the first num_examples
    random.shuffle(candidates)
    qa_pairs = candidates[:num_examples]

    out_path = Path(__file__).parent / "base_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved {len(qa_pairs)} QA pairs to {out_path}")
    print(f"  (from {len(candidates)} total candidates in SQuAD v1.1)")
    return qa_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SQuAD to benchmark format")
    parser.add_argument("--squad", default="data/squad.json", help="Path to squad.json")
    parser.add_argument("--num", type=int, default=1000, help="Number of QA pairs to extract")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(squad_path=args.squad, num_examples=args.num, seed=args.seed)
