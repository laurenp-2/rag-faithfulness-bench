# Retrieval Perturbation as a Stress-Test for RAG Faithfulness

> **A Counterfactual Benchmark for Evaluating Graceful Degradation in RAG Pipelines**
>
> Lauren Pothuru · Cornell University

Standard RAG benchmarks measure whether models produce correct answers. This benchmark asks a harder question: **when retrieval is wrong, does the model know it's wrong?** We stress-test RAG pipelines by systematically corrupting retrieved context via three perturbation types and measuring faithfulness, correctness, hallucination, and abstention across each failure mode.

---

## Motivation

RAG systems are widely deployed, but retrieval errors are inevitable in production. A retriever might surface a stale document, a mismatched passage, or a near-duplicate with a wrong date. Existing evaluation frameworks (RAGAS, ARES) measure end-to-end answer quality — they do not isolate what happens when the retrieved context is the source of the error.

This benchmark fills that gap. It answers three questions:

1. Does the LLM faithfully follow corrupted context even when the context is wrong?
2. Does the LLM hallucinate when context and ground truth conflict?
3. Does the LLM appropriately abstain ("I don't know") when context is unreliable?

---

## How it works

For each QA pair, we generate four versions of the retrieved context and run the full RAG pipeline on each:

```
gold context ──┬── original      →  baseline
               ├── entity_swap   →  wrong but plausible entity
               ├── negation      →  factual claim directly contradicted
               └── paraphrase    →  surface form changed, facts preserved (control)
                        │
                        ▼
               FAISS retriever (sentence-transformers)
                        │
                        ▼
               Claude LLM generator
                        │
                        ▼
               NLI faithfulness scorer (DeBERTa)
               + exact match / token F1
               + hallucination flag
               + abstention flag
```

---

## Perturbation types

| Type | What changes | What stays the same | Example |
|---|---|---|---|
| `original` | nothing | everything | *Microsoft was founded by Bill Gates in 1975.* |
| `entity_swap` | one named entity → plausible same-type alternative | sentence structure, all other facts | *Microsoft was founded by **Steve Jobs** in 1975.* |
| `negation` | core verb negated via dependency parse | all entities and dates | *Microsoft was **not** founded by Bill Gates in 1975.* |
| `paraphrase` | surface wording | all factual content (control condition) | *Bill Gates established Microsoft back in 1975.* |

`paraphrase` is the control: a robust model should be unaffected by it. Divergence from the baseline on `entity_swap` and `negation` reveals the failure modes.

---

## Metrics

| Metric | What it measures | Ideal direction |
|---|---|---|
| `faithfulness_rate` | NLI entailment: does the answer follow from the (perturbed) context? | Low on corrupted conditions |
| `correctness_em` | Exact match with gold answer | High on all conditions |
| `correctness_f1` | Token-level F1 with gold answer | High on all conditions |
| `hallucination_rate` | Answer contradicts both the perturbed context and the gold answer | Low on all conditions |
| `abstention_rate` | Model says "I don't know" | High on corrupted conditions |
| `contradiction_rate` | Answer explicitly contradicts the retrieved context | Moderate on negation/entity_swap |

---

## Setup

**Requirements:** Python 3.10+, an Anthropic API key.

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure API key
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=your_key_here

# 3. Download and format the seed dataset (~150 TriviaQA examples)
python data/download_data.py
```

---

## Running the benchmark

```bash
# Smoke-test on 10 examples (fast, ~2 minutes)
python experiments/run_benchmark.py --limit 10

# Full run on all 150 examples
python experiments/run_benchmark.py

# Custom configuration
python experiments/run_benchmark.py \
    --data data/base_qa_pairs.json \
    --output results/outputs.json \
    --k 3 \
    --limit 50
```

Output is written to two files:

- `results/outputs.json` — one record per (QA pair × perturbation type) with full scores
- `results/summary.json` — aggregated metrics table, also printed to stdout

Re-run the summary at any time:

```bash
python eval/metrics.py results/outputs.json
```

---

## Repository structure

```
rag-faithfulness-bench/
│
├── data/
│   ├── download_data.py        # downloads + formats TriviaQA subset
│   └── base_qa_pairs.json      # ~150 seed QA pairs (generated)
│
├── perturbations/
│   ├── entity_swap.py          # SpaCy NER + same-type lookup-table swap
│   ├── negation.py             # dependency-parse negation of main verb
│   └── paraphrase.py           # Claude API paraphrase (control condition)
│
├── pipeline/
│   ├── retriever.py            # FAISS dense retriever (sentence-transformers)
│   └── generator.py            # Claude API answer generation with strict prompt
│
├── eval/
│   ├── faithfulness_scorer.py  # NLI faithfulness + EM/F1 + hallucination flag
│   └── metrics.py              # aggregate summary table per perturbation type
│
├── experiments/
│   └── run_benchmark.py        # main entry point (orchestrates full pipeline)
│
├── results/
│   ├── outputs.json            # per-example results (generated)
│   └── summary.json            # aggregate metrics (generated)
│
├── requirements.txt
└── .env.example
```

---

## Dataset

Seed data is drawn from **TriviaQA** (`rc.wikipedia` split). Each example contains a question, a gold answer, and a gold context sentence extracted from the Wikipedia passage that contains the answer. We use 100–150 examples for the prototype.

Example entry in `data/base_qa_pairs.json`:

```json
{
  "id": "tc_33",
  "question": "Who founded Microsoft?",
  "gold_answer": "Bill Gates",
  "gold_context": "Microsoft was founded by Bill Gates and Paul Allen in 1975."
}
```

---

## Models used

| Component | Model | Notes |
|---|---|---|
| Retriever embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, ~80MB |
| Paraphrase generation | `claude-haiku-4-5-20251001` | Cost-efficient preprocessing |
| Answer generation | `claude-haiku-4-5-20251001` | Strict context-grounded prompt |
| NLI faithfulness scoring | `cross-encoder/nli-deberta-v3-base` | State-of-the-art NLI on SNLI/MNLI |

---

## Limitations

- **Scale:** 100–150 examples is sufficient for a prototype but limits statistical power. A full study would use 1,000+.
- **Single LLM:** Results are specific to Claude Haiku. Behavior may differ across model families and sizes.
- **English only:** All perturbations and NLI models are English-language.
- **Entity swap coverage:** The lookup table covers common entity types; rare or domain-specific entities may not be swapped.
- **Negation depth:** The dependency-parse negation handles simple declarative sentences; complex or compound sentences may not negate cleanly.

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{pothuru2025ragfaithfulness,
  title   = {Retrieval Perturbation as a Stress-Test for RAG Faithfulness:
             A Counterfactual Benchmark},
  author  = {Pothuru, Lauren},
  year    = {2025},
  note    = {Cornell University}
}
```
