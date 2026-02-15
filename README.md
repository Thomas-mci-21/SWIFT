<h1 align="center">SWIFT: Self-Learning When to Stop<br>in Fact-Checking Tasks</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/ACL_2026-ARR_Submission-blue.svg" alt="ACL 2026"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

<p align="center">
  <b>TL;DR</b>: We train a lightweight DeBERTa-v3 Critic (86M) to replace LLM self-assessment for stopping decisions in iterative fact-checking, achieving better accuracy with fewer retrieval steps.
</p>

---

## Overview

Iterative retrieval-augmented fact-checking methods let the LLM itself decide when to stop retrieving evidence. However, this **coupled** design causes the LLM to suffer from **overconfidence** — often giving final answers without searching for sufficient evidence, averaging merely 0.4–0.6 search steps per claim.

**SWIFT** (**S**elf-Learning **W**hen to Stop **I**n **F**act-Checking **T**asks) addresses this by decoupling the stopping decision from the reasoning LLM to a lightweight, independently trained **DeBERTa-v3-base Critic** (86M parameters). Specifically, SWIFT introduces a self-learning pipeline consisting of three phases:

1. **Self-Supervised Data Synthesis**: The generator LLM explores multi-step search trajectories. Intermediate judgments are labeled via hindsight correctness against ground truth, yielding training data without human annotation.
2. **Critic Training**: Trajectory samples are formatted as NLI premise–hypothesis pairs to fine-tune the lightweight Critic, which learns to assess whether the LLM's judgment is well-supported by the currently retrieved evidence.
3. **Critic-Gated Inference**: At each retrieval step, the Critic evaluates the generator's judgment. Retrieval continues only when the Critic's confidence falls below a threshold τ.

### Key Features

- **Self-Supervised Data Generation**: Automatically generates training data through multi-trajectory iterative retrieval — no human annotation needed
- **NLI-style Critic**: DeBERTa-v3-base initialized from NLI pre-training, using premise-hypothesis format for credibility assessment
- **Lightweight & Efficient**: 86M parameter Critic replaces expensive LLM self-assessment calls
- **Black-box LLM Compatible**: Works with any LLM API (no logprobs required)
- **Cross-LLM Generalization**: Critic trained on one LLM's trajectories generalizes to other LLMs

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
  - [Phase 1: Self-Supervised Data Synthesis](#phase-1-self-supervised-data-synthesis)
  - [Phase 2: Critic Training](#phase-2-critic-training)
  - [Phase 3: Inference](#phase-3-inference)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Installation

```bash
git clone <repo-url>
cd SWIFT
pip install -r requirements.txt
```

**Requirements**: Python >= 3.9, PyTorch >= 2.0, CUDA >= 11.8 (for Critic training/inference)

### API Configuration

Copy the example config and fill in your API keys:

```bash
cp common/shared_config.example.py common/shared_config.py
# Edit common/shared_config.py with your OpenAI-compatible API key
```

SWIFT uses DuckDuckGo for web search by default (free, no API key needed).

---

## Data Preparation

We evaluate on 3 fact-checking benchmarks. The SWIFT Critic is trained exclusively on the training split of FactCheck-Bench (504 claims), with no claim overlap between training and test sets. FacTool-QA and FELM-WK serve as **zero-shot out-of-distribution** evaluations to assess the Critic's generalization capability.

| Dataset | # Claims | Domain | Label Conversion | Source |
|:--------|:--------:|:------:|:-----------------|:-------|
| [FactCheck-Bench](https://github.com/yuxiaw/Factcheck-GPT) | 127 (test) / 504 (train) | News | Fine-grained → Binary (supported→True, refuted→False) | Wang et al., 2024 |
| [FacTool-QA](https://github.com/GAIR-NLP/factool) | 233 | General Knowledge | Knowledge-based QA subset | Chern et al., 2024 |
| [FELM-WK](https://github.com/tonytan48/FELM) | 184 | World Knowledge | World Knowledge subset (54% True, 46% False) | Chen et al., 2023 |

Place dataset files under `data/`:
```
data/
├── factcheckbench/
│   ├── raw.csv          # Full dataset
│   ├── train.csv        # Training split (504 claims)
│   └── test.csv         # Test split (127 claims)
├── factool_qa/raw.csv
└── felm_wk/raw.csv
```

---

## Quick Start

```bash
# Run SWIFT inference on FactCheckBench (requires trained Critic)
CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
    --experiment_name fcb_swift \
    --dataset factcheckbench \
    --mode swift \
    --critic_path checkpoints/swift_v5 \
    --threshold 0.7

# Run without search (LLM-only baseline)
python inference/inference.py \
    --experiment_name fcb_nosearch \
    --dataset factcheckbench \
    --mode no_search
```

---

## Pipeline

### Phase 1: Self-Supervised Data Synthesis

Generate multi-trajectory training data by running the LLM through iterative retrieval:

```bash
# Generate 3 trajectories with DuckDuckGo search
for t in 0 1 2; do
    python generation/generation.py \
        --experiment_name swift_v5_t${t} \
        --search_engine ddg \
        --temperature 0.7 \
        --trajectory_id ${t}
done
```

This produces ~6,000 raw samples (504 claims x 4 steps x 3 trajectories), balanced to ~3,000 for training.

### Phase 2: Critic Training

Train the DeBERTa-v3-base Critic:

```bash
# Prepare training data (claim-level split, verdict balancing)
python training/prepare_training.py --experiment_name swift_v5

# Train the Critic
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python training/main.py \
    --experiment_name swift_v5 --fp16
```

**Training details**:

| Config | Value |
|:-------|:------|
| Base model | DeBERTa-v3-base-mnli-fever-anli (86M) |
| Input format | NLI-style (premise: claim+evidence+rationale, hypothesis: judgment) |
| Data split | Claim-level stratified 80/20 (no leakage) |
| Balanced data | ~1,740 train / ~386 val (50:50 accept/reject) |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Epochs | 10 (early stopping, patience=3) |
| Hardware | Single NVIDIA A40, ~10 min |

### Phase 3: Inference

SWIFT supports 5 inference modes:

```bash
# SWIFT (ours) — Critic-gated stopping
CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
    --experiment_name fcb_swift --mode swift \
    --critic_path checkpoints/swift_v5 --threshold 0.7

# No-Search — LLM answers without retrieval
python inference/inference.py \
    --experiment_name fcb_nosearch --mode no_search

# Fixed-K — Always retrieve K rounds
python inference/inference.py \
    --experiment_name fcb_fixed3 --mode fixed_k --fixed_k 3

# NLI-Baseline — Off-the-shelf NLI model (no fine-tuning)
CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
    --experiment_name fcb_nli --mode nli_baseline --threshold 0.7

# LLM-Critic — LLM as Critic (prompt-based)
python inference/inference.py \
    --experiment_name fcb_llmcritic --mode llm_critic
```

---

## Project Structure

```
SWIFT/
├── common/
│   ├── modeling.py              # LLM API wrapper (OpenAI-compatible)
│   ├── utils.py                 # Utility functions & metrics
│   └── shared_config.example.py # API config template
├── search/
│   ├── query_ddg.py             # DuckDuckGo search
│   └── query_serper.py          # Serper (Google) search (optional)
├── generation/
│   ├── generation.py            # Phase 1: Self-practice data generation
│   └── prompts.py               # Prompt templates
├── training/
│   ├── main.py                  # Phase 2: DeBERTa Critic training
│   ├── config.py                # Training hyperparameters
│   ├── prepare_training.py      # Claim-level split & balancing
│   └── data_processing.py       # NLI-style input formatting
├── inference/
│   └── inference.py             # Phase 3: Multi-mode inference
├── utils/
│   ├── evaluation.py            # Evaluation metrics
│   └── ...                      # Precision/recall utilities
├── data/                        # Benchmark datasets (not included)
├── scripts/                     # Experiment scripts
├── config.py                    # Global configuration
├── evaluate_results.py          # Results evaluation
├── analyze_errors.py            # Error analysis
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@inproceedings{anonymous2026swift,
  title     = {SWIFT: Self-Learning When to Stop in Fact-Checking Tasks},
  author    = {Anonymous},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association
               for Computational Linguistics (ACL)},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgements

This work builds upon:
- [SIM-RAG](https://arxiv.org/abs/2410.17952) (SIGIR '25) — Self-practicing methodology for RAG
- [FIRE](https://github.com/mbzuai-nlp/fire) (NAACL '25) — Iterative retrieval for fact-checking
- [DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) — Pre-trained NLI model

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
