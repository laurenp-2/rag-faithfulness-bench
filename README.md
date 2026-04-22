# Retrieval Perturbation as a Stress-Test for RAG Faithfulness

> A Counterfactual Benchmark for Evaluating Graceful Degradation in RAG Pipelines
>
> Lauren Pothuru · Cornell University · Anote Take-Home Research Assignment

Standard RAG benchmarks measure whether models produce correct answers. This benchmark asks a harder question: **when retrieval is wrong, does the model know it's wrong?**

Full details are in the [draft paper](writeup/latex.tex) and blog post (coming soon).

---

## What this is

A lightweight benchmark that stress-tests RAG pipelines by systematically corrupting retrieved context and measuring how models respond. We apply three perturbation types to gold context passages — entity swaps, negations, and paraphrases (control) — then score faithfulness, hallucination, and abstention across each condition.

This isolates a blind spot in existing evaluation frameworks (RAGAS, ARES): they measure end-to-end correctness but do not expose what happens when the *retrieved context itself* is the source of the error.

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) (runs locally, no API key needed).

```bash
# 1. Pull a model
ollama pull llama3.2:3b

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Download seed data (~150 TriviaQA examples)
python data/download_data.py
```

---

## Running the benchmark

```bash
# Quick smoke test (10 examples, ~3 min)
python experiments/run_benchmark.py --limit 10

# Full run (150 examples)
python experiments/run_benchmark.py

# Compare models
python experiments/run_benchmark.py --model llama3.1:8b
python experiments/run_benchmark.py --model mistral:7b
```

Results are written to `results/<model>/outputs.json` and `results/<model>/summary.json`.

---

## Repository structure

```
rag-faithfulness-bench/
├── data/                   # seed QA pairs + download script
├── perturbations/          # entity_swap, negation, paraphrase
├── pipeline/               # FAISS retriever + Ollama generator
├── eval/                   # NLI faithfulness scorer, metrics, visualization
├── experiments/            # run_benchmark.py (main entry point)
├── results/                # per-model outputs and summary tables
├── generate_figures.py     # produces writeup figures from results
├── error_analysis.py       # qualitative failure mode analysis
└── writeup/
    ├── latex.tex           # draft paper
    └── figures/            # generated plots (PDFs)
```

---

## Key results

Across four models (Llama 3.2 3B, Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B):

- **Faithfulness** drops sharply under `entity_swap` and `negation` conditions
- **Hallucination rate** increases when context conflicts with ground truth
- **Abstention** is largely absent — models rarely say "I don't know" even when context is unreliable
- **Paraphrase** (control) produces near-identical results to baseline, validating the perturbation design

Full tables and figures are in the paper.
