"""
Paraphrase perturbation (control condition).

Uses a local Ollama model to rephrase the gold context while preserving all
factual content. This is the control condition: surface form changes but
meaning is intact. A robust RAG model should be unaffected by this perturbation.

Requires Ollama running locally: https://ollama.com
    ollama pull llama3.2
"""

import ollama

_SYSTEM_PROMPT = (
    "You are a precise paraphrasing assistant. "
    "Rewrite the given sentence using different wording, "
    "but preserve every factual detail exactly — names, dates, numbers, and "
    "all other specifics must remain correct. "
    "Return only the rewritten sentence with no extra commentary."
)


def paraphrase_context(context: str, model: str = "llama3.2:3b") -> str:
    """
    Return a paraphrase of *context* that preserves factual content.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
    )
    return response["message"]["content"].strip()


def batch_paraphrase(contexts: list[str], model: str = "llama3.2:3b") -> list[str]:
    """Paraphrase a list of contexts, returning results in the same order."""
    results = []
    for ctx in contexts:
        try:
            results.append(paraphrase_context(ctx, model=model))
        except Exception as e:
            print(f"[paraphrase] Error on context '{ctx[:60]}...': {e}")
            results.append(ctx) 
    return results


if __name__ == "__main__":
    examples = [
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    for ex in examples:
        para = paraphrase_context(ex)
        print(f"Original   : {ex}")
        print(f"Paraphrase : {para}")
        print()
