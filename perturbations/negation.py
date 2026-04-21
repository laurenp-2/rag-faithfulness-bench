"""
Negation perturbation.

Uses SpaCy dependency parsing to locate the main verb of the gold context
sentence and inserts a negation auxiliary ("did not", "does not", "was not",
etc.).  The result: the context directly contradicts the true fact.
"""

import spacy

# Maps auxiliary / tense markers to their negated forms
_NEG_MAP = {
    # be-verbs
    "is": "is not",
    "are": "are not",
    "was": "was not",
    "were": "were not",
    "be": "not be",
    "been": "not been",
    # have-verbs
    "has": "has not",
    "have": "have not",
    "had": "had not",
    # modals
    "can": "cannot",
    "could": "could not",
    "will": "will not",
    "would": "would not",
    "shall": "shall not",
    "should": "should not",
    "may": "may not",
    "might": "might not",
    "must": "must not",
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


def _negate_sentence(sent) -> str | None:
    """
    Try to negate a single SpaCy Span (sentence).
    Returns the negated string, or None if no clean negation was found.
    """
    tokens = list(sent)


    root = next((t for t in tokens if t.dep_ == "ROOT"), None)
    if root is None:
        return None


    for tok in tokens:
        if tok.dep_ in ("aux", "auxpass") and tok.i < root.i:
            lemma = tok.lemma_.lower()
            if lemma in _NEG_MAP:
                negated_form = _NEG_MAP[lemma]
                # Preserve capitalisation of the original auxiliary
                if tok.is_sent_start or tok.i == sent.start:
                    negated_form = negated_form[0].upper() + negated_form[1:]
                result = (
                    sent.text[: tok.idx - sent.start_char]
                    + negated_form
                    + sent.text[tok.idx - sent.start_char + len(tok.text):]
                )
                return result


    if root.pos_ == "VERB" and root.tag_ in ("VBD", "VBZ", "VBP", "VB"):
        # Insert "did not" before the root; convert root to base form
        base_form = root.lemma_
        insertion = "did not " + base_form
        # Preserve sentence-start capitalisation
        if root.i == sent.start:
            insertion = insertion[0].upper() + insertion[1:]
        result = (
            sent.text[: root.idx - sent.start_char]
            + insertion
            + sent.text[root.idx - sent.start_char + len(root.text):]
        )
        return result

    return None


def negate_context(context: str) -> str:
    """
    Return a copy of *context* with the core factual claim negated.
    If the context contains multiple sentences, the first sentence is negated.
    Falls back to a simple string insertion if SpaCy parsing yields nothing.
    """
    nlp = _get_nlp()
    doc = nlp(context)

    for sent in doc.sents:
        negated = _negate_sentence(sent)
        if negated is not None:

            before = context[: sent.start_char]
            after = context[sent.end_char:]
            return before + negated + after


    for word in ["was ", "is ", "were ", "are ", "has ", "have ", "did "]:
        if word in context:
            return context.replace(word, word.rstrip() + " not ", 1)

    return "It is not true that " + context[0].lower() + context[1:]


if __name__ == "__main__":
    examples = [
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        "The company has offices in Seattle.",
        "Einstein developed the theory of relativity.",
        "Water boils at 100 degrees Celsius.",
    ]
    for ex in examples:
        print(f"Original : {ex}")
        print(f"Negated  : {negate_context(ex)}")
        print()
