"""
Faithfulness and correctness scorer.

For each (question, context, generated_answer, gold_answer) tuple, computes:

  faithfulness_score  — NLI entailment score: does the answer follow from
                        the (potentially perturbed) context?
                        High score on a perturbed context = bad (the model
                        is faithfully propagating a lie).

  correctness_score   — Does the answer match the gold answer?
                        Combines exact match and token-level F1.

  hallucination_flag  — True if the answer contradicts BOTH the perturbed
                        context AND the gold answer.

Uses cross-encoder/nli-deberta-v3-base for NLI (premise → hypothesis).
"""

from __future__ import annotations

import re
import string
from collections import Counter

import torch
from transformers import pipeline as hf_pipeline


_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
_LABEL_MAP = {"ENTAILMENT": 2, "NEUTRAL": 1, "CONTRADICTION": 0}


class FaithfulnessScorer:
    def __init__(self, nli_model: str = _NLI_MODEL, device: int | None = None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self._nli = hf_pipeline(
            "text-classification",
            model=nli_model,
            device=device,
            top_k=None,          # return all labels
            truncation=True,
            max_length=512,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        generated_answer: str,
        perturbed_context: str,
        gold_answer: str,
        gold_context: str,
    ) -> dict:
        """
        Returns a scoring dict for a single example.

        Keys:
            faithfulness_score   float [0, 1]  — entailment score vs. perturbed context
            correctness_em       bool           — exact match with gold answer
            correctness_f1       float [0, 1]  — token F1 with gold answer
            hallucination_flag   bool
            context_contradiction bool         — model answer contradicts perturbed context
            gold_context_faithful float        — entailment score vs. gold context (sanity check)
        """
        # 1. NLI: perturbed context → answer
        faith_scores = self._nli_scores(premise=perturbed_context, hypothesis=generated_answer)
        faithfulness_score = faith_scores["entailment"]
        context_contradiction = faith_scores["contradiction"] > 0.5

        # 2. NLI: gold context → answer (sanity / comparison)
        gold_faith = self._nli_scores(premise=gold_context, hypothesis=generated_answer)
        gold_context_faithful = gold_faith["entailment"]

        # 3. Correctness vs gold answer
        em = exact_match(generated_answer, gold_answer)
        f1 = token_f1(generated_answer, gold_answer)

        # 4. Hallucination: answer conflicts with both perturbed context AND gold
        gold_answer_contradiction = self._nli_scores(
            premise=gold_answer, hypothesis=generated_answer
        )["contradiction"] > 0.5
        hallucination_flag = context_contradiction and (f1 < 0.2) and gold_answer_contradiction

        return {
            "faithfulness_score": round(faithfulness_score, 4),
            "correctness_em": em,
            "correctness_f1": round(f1, 4),
            "hallucination_flag": hallucination_flag,
            "context_contradiction": context_contradiction,
            "gold_context_faithful": round(gold_context_faithful, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nli_scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Run NLI and return a dict with 'entailment', 'neutral', 'contradiction' scores."""
        result = self._nli(f"{premise} [SEP] {hypothesis}")
        # result is a list of [{"label": ..., "score": ...}, ...]
        scores = {item["label"].lower(): item["score"] for item in result[0]}
        # Normalise key names
        normalised = {}
        for raw_label, score in scores.items():
            if "entail" in raw_label:
                normalised["entailment"] = score
            elif "contra" in raw_label:
                normalised["contradiction"] = score
            else:
                normalised["neutral"] = score
        return normalised


# ---------------------------------------------------------------------------
# Standalone metric functions (no model needed)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> bool:
    return _normalise(prediction) == _normalise(gold)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalise(prediction).split()
    gold_tokens = _normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    scorer = FaithfulnessScorer()

    result = scorer.score(
        generated_answer="Steve Jobs founded Microsoft.",
        perturbed_context="Microsoft was founded by Steve Jobs in 1975.",
        gold_answer="Bill Gates",
        gold_context="Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    )
    print("Scores:", result)
