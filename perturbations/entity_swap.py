"""
Entity-swap perturbation.

Extracts named entities from the gold context using SpaCy, then replaces
them with plausible same-type alternatives drawn from a small lookup table.
The result: a context that looks syntactically valid but is factually wrong.
"""

import random
import re
import spacy

# ---------------------------------------------------------------------------
# Lookup tables: entity type → list of plausible swap candidates
# Kept intentionally small; extend for a fuller benchmark.
# ---------------------------------------------------------------------------
SWAP_CANDIDATES: dict[str, list[str]] = {
    "PERSON": [
        "Steve Jobs", "Elon Musk", "Jeff Bezos", "Mark Zuckerberg",
        "Warren Buffett", "Tim Cook", "Sundar Pichai", "Satya Nadella",
        "Larry Page", "Sergey Brin", "Jack Dorsey", "Reed Hastings",
        "Jensen Huang", "Sam Altman", "Ada Lovelace", "Alan Turing",
    ],
    "ORG": [
        "Apple", "Google", "Amazon", "Meta", "Tesla", "Netflix",
        "IBM", "Oracle", "Intel", "Nvidia", "Salesforce", "Adobe",
        "Spotify", "Uber", "Airbnb", "Twitter",
    ],
    "GPE": [  # countries, cities, states
        "Germany", "Japan", "Canada", "France", "Australia",
        "Brazil", "India", "China", "the United Kingdom", "South Korea",
        "Seattle", "Boston", "Chicago", "Austin", "Toronto",
    ],
    "DATE": [
        "1968", "1972", "1984", "1991", "1998", "2001", "2007",
        "2010", "2015", "2019", "2023",
    ],
    "CARDINAL": [
        "two", "three", "five", "seven", "ten", "twelve", "twenty",
        "fifty", "one hundred",
    ],
    "NORP": [  # nationalities, religious groups
        "British", "French", "German", "Japanese", "Canadian",
        "Australian", "Brazilian", "Korean",
    ],
}


def _load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "SpaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _load_nlp()
    return _nlp


def swap_entities(context: str, seed: int | None = None) -> str:
    """
    Return a copy of *context* with at least one named entity replaced by
    a plausible same-type alternative.  If no swappable entity is found the
    original string is returned unchanged.
    """
    rng = random.Random(seed)
    nlp = _get_nlp()
    doc = nlp(context)

    # Collect entities that have swap candidates, preserving order
    swappable = [
        ent for ent in doc.ents
        if ent.label_ in SWAP_CANDIDATES
    ]

    if not swappable:
        return context

    # Pick one entity to swap (prefer the first substantive one)
    target = rng.choice(swappable)
    candidates = [c for c in SWAP_CANDIDATES[target.label_] if c.lower() != target.text.lower()]
    if not candidates:
        return context

    replacement = rng.choice(candidates)
    # Use regex so the replacement is exact (handles sentence boundaries)
    perturbed = re.sub(
        r"\b" + re.escape(target.text) + r"\b",
        replacement,
        context,
        count=1,
    )
    return perturbed


if __name__ == "__main__":
    examples = [
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein was born in Germany in 1879.",
    ]
    for ex in examples:
        print(f"Original : {ex}")
        print(f"Perturbed: {swap_entities(ex, seed=0)}")
        print()
