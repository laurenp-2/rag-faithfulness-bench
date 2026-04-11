"""
LLM answer generator.

Takes a question and a list of retrieved context strings, constructs a
prompt, and calls the Claude API to produce an answer grounded in the
provided context.
"""

from __future__ import annotations

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the information provided in the context. "
    "If the context does not contain enough information to answer the question, "
    "respond with exactly: I don't know. "
    "Do not add information from your own knowledge. "
    "Keep your answer concise — one sentence or less."
)

_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""


def generate_answer(
    question: str,
    contexts: list[str],
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 128,
) -> dict:
    """
    Generate an answer given a question and retrieved contexts.

    Returns a dict with:
        - answer (str): the generated answer
        - model (str): model used
        - abstained (bool): True if model said "I don't know"
        - context_used (str): concatenated context passed to the model
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    context_block = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))
    user_message = _USER_TEMPLATE.format(context=context_block, question=question)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text.strip()
    abstained = answer.lower().startswith("i don't know") or answer.lower() == "i don't know."

    return {
        "answer": answer,
        "model": model,
        "abstained": abstained,
        "context_used": context_block,
    }


if __name__ == "__main__":
    contexts = ["Microsoft was not founded by Bill Gates and Paul Allen in 1975."]
    result = generate_answer("Who founded Microsoft?", contexts)
    print("Answer    :", result["answer"])
    print("Abstained :", result["abstained"])
