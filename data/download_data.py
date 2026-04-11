"""
Downloads and formats a subset of TriviaQA for use as the benchmark seed dataset.
Outputs: data/base_qa_pairs.json
"""

import json
import random
from pathlib import Path
from datasets import load_dataset


def download_and_format(num_examples: int = 150, seed: int = 42) -> None:
    random.seed(seed)

    print("Loading TriviaQA (rc.wikipedia split)...")
    dataset = load_dataset("trivia_qa", "rc.wikipedia", split="train", trust_remote_code=True)

    qa_pairs = []
    for item in dataset:
        question = item["question"].strip()
        gold_answer = item["answer"]["value"].strip()

        # Pull the first Wikipedia search result that has non-empty text
        contexts = item.get("search_results", {})
        search_contexts = contexts.get("search_context", []) if contexts else []
        gold_context = ""
        for ctx in search_contexts:
            ctx = ctx.strip()
            if gold_answer.lower() in ctx.lower() and len(ctx) > 40:
                # Keep the single most relevant sentence
                for sentence in ctx.split("."):
                    if gold_answer.lower() in sentence.lower() and len(sentence.strip()) > 20:
                        gold_context = sentence.strip() + "."
                        break
            if gold_context:
                break

        if not gold_context:
            continue

        qa_pairs.append({
            "id": item["question_id"],
            "question": question,
            "gold_answer": gold_answer,
            "gold_context": gold_context,
        })

        if len(qa_pairs) >= num_examples:
            break

    out_path = Path(__file__).parent / "base_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved {len(qa_pairs)} QA pairs to {out_path}")


if __name__ == "__main__":
    download_and_format()
