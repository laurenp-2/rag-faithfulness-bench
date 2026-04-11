"""
Paraphrase perturbation (control condition).

Calls the Claude API to rephrase the gold context while preserving all
factual content.  This is the control condition: surface form changes but
meaning is intact.  A robust RAG model should be unaffected by this
perturbation.
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a precise paraphrasing assistant. "
    "Rewrite the given sentence using different wording, "
    "but preserve every factual detail exactly — names, dates, numbers, and "
    "all other specifics must remain correct. "
    "Return only the rewritten sentence with no extra commentary."
)


def paraphrase_context(context: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """
    Return a paraphrase of *context* that preserves factual content.

    Uses Claude Haiku for speed and cost efficiency since this is
    a high-volume preprocessing step.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    return message.content[0].text.strip()


def batch_paraphrase(contexts: list[str], model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """Paraphrase a list of contexts, returning results in the same order."""
    results = []
    for ctx in contexts:
        try:
            results.append(paraphrase_context(ctx, model=model))
        except Exception as e:
            print(f"[paraphrase] Error on context '{ctx[:60]}...': {e}")
            results.append(ctx)  # fall back to original on error
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
