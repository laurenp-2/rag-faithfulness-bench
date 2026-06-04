"""
LLM answer generator.

Supports three backends selected by model name prefix:
  - Ollama (local): any model without a recognised API prefix
      e.g. "llama3.2:3b", "mistral:7b", "qwen2.5:7b"
  - OpenAI API: model names starting with "gpt-"
      e.g. "gpt-4o-mini"  (set OPENAI_API_KEY env var)
  - Anthropic API: model names starting with "claude-"
      e.g. "claude-haiku-4-5-20251001"  (set ANTHROPIC_API_KEY env var)

Requires Ollama running locally for Ollama models: https://ollama.com
"""

from __future__ import annotations

import os

_DEFAULT_MODEL = "llama3.2:3b"

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

_USER_TEMPLATE_WITH_SCORE = """\
Context (retrieval similarity: {score:.3f}):
{context}

Question: {question}

Answer:"""


def _is_openai(model: str) -> bool:
    return model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude-")


def _generate_ollama(user_message: str, model: str) -> str:
    import ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"].strip()


def _generate_openai(user_message: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=150,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _generate_anthropic(user_message: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=150,
    )
    return response.content[0].text.strip()


def generate_answer(
    question: str,
    contexts: list[str],
    model: str = _DEFAULT_MODEL,
    similarity_score: float | None = None,
) -> dict:
    """
    Generate an answer given a question and retrieved contexts.

    Args:
        question: The question to answer.
        contexts: List of retrieved context strings.
        model: Model identifier (Ollama tag, OpenAI model name, or Anthropic model ID).
        similarity_score: If provided, prepended to the context block to signal
                          retrieval confidence (for the similarity-conditioned experiment).

    Returns a dict with:
        - answer (str): the generated answer
        - model (str): model used
        - abstained (bool): True if model said "I don't know"
        - context_used (str): concatenated context passed to the model
        - similarity_score (float | None): the FAISS score if supplied
    """
    context_block = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))

    if similarity_score is not None:
        user_message = _USER_TEMPLATE_WITH_SCORE.format(
            score=similarity_score,
            context=context_block,
            question=question,
        )
    else:
        user_message = _USER_TEMPLATE.format(context=context_block, question=question)

    if _is_openai(model):
        answer = _generate_openai(user_message, model)
    elif _is_anthropic(model):
        answer = _generate_anthropic(user_message, model)
    else:
        answer = _generate_ollama(user_message, model)

    abstained = answer.lower().startswith("i don't know") or answer.lower() == "i don't know."

    return {
        "answer": answer,
        "model": model,
        "abstained": abstained,
        "context_used": context_block,
        "similarity_score": similarity_score,
    }


if __name__ == "__main__":
    contexts = ["Microsoft was not founded by Bill Gates and Paul Allen in 1975."]
    result = generate_answer("Who founded Microsoft?", contexts)
    print("Answer    :", result["answer"])
    print("Abstained :", result["abstained"])
