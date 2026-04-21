"""
Converts a locally downloaded SQuAD v1.1 JSON file into the benchmark format.

Usage:
    python data/download_data.py

Input:  data/squad.json  (download from https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
Output: data/base_qa_pairs.json
"""

import json
import random
from pathlib import Path


def convert(squad_path: str = "data/squad.json", num_examples: int = 150, seed: int = 42) -> None:
    random.seed(seed)

    with open(squad_path) as f:
        squad = json.load(f)

    qa_pairs = []
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                if not qa["answers"]:
                    continue
                question = qa["question"].strip()
                gold_answer = qa["answers"][0]["text"].strip()

                # Find the sentence in the context that contains the answer
                gold_context = ""
                for sentence in context.split("."):
                    if gold_answer.lower() in sentence.lower() and len(sentence.strip()) > 20:
                        gold_context = sentence.strip() + "."
                        break

                if not gold_context:
                    continue

                qa_pairs.append({
                    "id": qa["id"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_context": gold_context,
                })

                if len(qa_pairs) >= num_examples:
                    break
            if len(qa_pairs) >= num_examples:
                break
        if len(qa_pairs) >= num_examples:
            break

    out_path = Path(__file__).parent / "base_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved {len(qa_pairs)} QA pairs to {out_path}")


if __name__ == "__main__":
    convert()
