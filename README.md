# Retrieval Perturbation as a Stress-Test for RAG Faithfulness

> A Counterfactual Benchmark for Evaluating Graceful Degradation in RAG Pipelines
>
> Lauren Pothuru · Cornell University · Anote Take-Home Research Assignment

Standard RAG benchmarks measure whether models produce correct answers. This benchmark asks a different question: **when retrieval is wrong, does the model know it's wrong?**

Full details are in the [draft paper](writeup/latex.tex).

---

## What this is

A lightweight benchmark that stress-tests RAG pipelines by corrupting retrieved context and measuring how models respond. We apply three perturbation types to gold context passages (entity swaps, negations, and paraphrases/control) then score faithfulness, hallucination, and abstention across each condition.

This isolates a blind spot in existing evaluation frameworks (RAGAS, ARES): they measure end-to-end correctness but do not expose what happens when the retrieved context is the source of the error.

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) for local open-weight
models. OpenAI and Anthropic API keys are optional for the closed-source model runs.

```bash
# 1. Pull a model
ollama pull llama3.2:3b

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Download SQuAD v1.1 and build the 1,000-example benchmark split
curl -L -o data/squad.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
python data/download_data.py
```

---

## Running the benchmark

```bash
# Quick smoke test (10 examples, ~3 min)
python experiments/run_benchmark.py --limit 10

# Full run (1,000 examples × 4 conditions)
python experiments/run_benchmark.py

# Compare open-weight models
python experiments/run_benchmark.py --model llama3.1:8b
python experiments/run_benchmark.py --model mistral:7b
python experiments/run_benchmark.py --model qwen2.5:7b

# Closed-source models (require matching API keys)
OPENAI_API_KEY=... python experiments/run_benchmark.py --model gpt-4o-mini
ANTHROPIC_API_KEY=... python experiments/run_benchmark.py --model claude-haiku-4-5-20251001

# Similarity-score ablation
python experiments/run_benchmark.py --model llama3.2:3b --limit 200 --include-similarity
```

Results are written to `results/<model>/outputs.json` and `results/<model>/summary.json`.

Generate figures and analysis:

```bash
python generate_figures.py
python error_analysis.py --all-models
python eval/human_validation.py report --annotations data/annotation_sample.json
```

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

Across six models (Llama 3.2 3B, Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B,
GPT-4o mini, and Claude Haiku):

- **Faithfulness** drops sharply under `entity_swap` and `negation` conditions
- **Hallucination rates** remain low overall (0--9%), with entity swaps producing the highest rates
- **Abstention rises under corrupted retrieval**, but the cliff is much shallower than an ideal calibrated model
- **Closed-source models** show the strongest abstention calibration, with Qwen 2.5 7B the strongest open-weight model
- **Paraphrase** remains much closer to original than to corrupted conditions, though small surface-form effects are detectable at 1,000 examples
- **Human validation** confirms mostly fluent perturbations while surfacing quality limits in entity swaps and paraphrases

Full tables and figures are in the paper.
