<h1 align="center">SWIFT: Self-learning When to Stop<br>in Fact-checking Tasks</h1>

<p align="center">
  <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a> -->
  <a href="#"><img src="https://img.shields.io/badge/ACL_2026-ARR_Submission-blue.svg" alt="ACL 2026"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

<p align="center">
  <b>TL;DR</b>: We train a lightweight DeBERTa-v3 Critic (86M) to replace LLM self-assessment for stopping decisions in iterative fact-checking, achieving better accuracy with fewer retrieval steps.
</p>

<p align="center">
  <img src="assets/framework.png" width="85%">
</p>

---

## Overview

Iterative retrieval-augmented fact-checking methods (e.g., [FIRE](https://github.com/mbzuai-nlp/fire)) let the LLM itself decide when to stop retrieving evidence. However, LLMs suffer from **overconfidence** — often giving final answers without searching for any evidence.

**SWIFT** addresses this by training a lightweight **DeBERTa-v3-base Critic** (86M parameters) as an independent judgment credibility assessor. The Critic is trained via **self-practice**: the LLM generates multi-round retrieval trajectories, and we use the correctness of each intermediate judgment as supervision. At inference time, the Critic gates the stopping decision — replacing costly and unreliable LLM self-assessment.

### Key Features

- **Self-Practice Data Generation**: Automatically generates training data through multi-trajectory iterative retrieval — no human annotation needed
- **NLI-style Critic**: DeBERTa-v3-base initialized from NLI pre-training, using premise-hypothesis format for credibility assessment
- **Lightweight & Efficient**: 86M parameter Critic replaces expensive LLM self-assessment calls
- **Black-box LLM Compatible**: Works with any LLM API (no logprobs required)
- **Cross-LLM Generalization**: Critic trained on GPT-4o-mini data generalizes to DeepSeek-V3

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
  - [Phase 1: Self-Practice Data Generation](#phase-1-self-practice-data-generation)
  - [Phase 2: Critic Training](#phase-2-critic-training)
  - [Phase 3: Inference](#phase-3-inference)
- [Main Results](#main-results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Installation

```bash
git clone https://github.com/Thomas-mci-21/SWIFT.git
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

We evaluate on 5 fact-checking benchmarks:

| Dataset | # Claims | Domain | Source |
|:--------|:--------:|:------:|:-------|
| [FactCheckBench](https://github.com/yuxiaw/Factcheck-GPT) | 127 (test) | LLM output verification | Wang et al., 2024 |
| [FacTool-QA](https://github.com/GAIR-NLP/factool) | 233 | Knowledge QA | Chern et al., 2024 |
| [FELM-WK](https://github.com/tonytan48/FELM) | 184 | World knowledge | Chen et al., 2023 |
| [BingCheck](https://github.com/AlibabaResearch/BingCheck) | 142 | News claims | Liu et al., 2024 |
| [HoVer](https://github.com/hover-nlp/hover) | 200 | Multi-hop claims | Jiang et al., 2020 |

FactCheckBench is split into train (504 claims) for Critic training and test (127 claims) for evaluation. All other datasets are used for **zero-shot out-of-domain** evaluation.

Place dataset files under `data/`:
```
data/
├── factcheckbench/
│   ├── raw.csv          # Full dataset
│   ├── train.csv        # Training split (504 claims)
│   └── test.csv         # Test split (127 claims)
├── factool_qa/raw.csv
├── felm_wk/raw.csv
├── bingcheck/raw.csv
└── hover/raw.csv
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

### Phase 1: Self-Practice Data Generation

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
| Base model | [DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) (86M) |
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

# LLM-Critic — GPT-4o-mini as Critic (prompt-based)
python inference/inference.py \
    --experiment_name fcb_llmcritic --mode llm_critic
```

---

## Main Results

Performance comparison on fact-checking benchmarks using GPT-4o-mini + DuckDuckGo search.

### Accuracy & F1 Scores

| Method | FCB Acc | FacTool Acc | FELM Acc | BingCheck Acc | HoVer Acc | Avg Steps |
|:-------|:-------:|:-----------:|:--------:|:-------------:|:---------:|:---------:|
| No-Search | 62.2% | 75.5% | 66.3% | 83.1% | 55.5% | 0 |
| Fixed-1 | 79.5% | 81.1% | 59.2% | 78.2% | 56.0% | 1.0 |
| Fixed-3 | 79.5% | 80.7% | 65.2% | **89.4%** | 60.5% | 3.0 |
| Fixed-5 | 81.9% | 78.1% | 67.4% | 88.7% | **65.5%** | 5.0 |
| FIRE (NAACL '25) | 73.2% | 82.0% | 63.0% | 83.1% | 62.5% | 0.59 |
| NLI-Baseline | 75.6% | 78.5% | 67.9% | 85.9% | 65.0% | 4.50 |
| LLM-Critic | 77.2% | 80.7% | 71.2% | 86.6% | 59.0% | 2.09 |
| **SWIFT (Ours)** | 77.2% | **82.8%** | 67.9% | 81.0% | 61.5% | 2.41 |

### SWIFT vs FIRE (True F1 / False F1)

| Dataset | SWIFT True F1 | FIRE True F1 | Delta | SWIFT False F1 | FIRE False F1 | Delta |
|:--------|:---:|:---:|:---:|:---:|:---:|:---:|
| FCB | **0.850** | 0.805 | +0.045 | 0.525 | 0.575 | -0.050 |
| FacTool | **0.886** | 0.884 | +0.002 | **0.649** | 0.596 | +0.053 |
| FELM | **0.749** | 0.712 | +0.037 | **0.556** | 0.485 | +0.071 |
| BingCheck | 0.871 | **0.885** | -0.014 | 0.640 | **0.684** | -0.044 |
| HoVer | **0.597** | 0.556 | +0.041 | 0.632 | **0.675** | -0.043 |

SWIFT outperforms FIRE on True F1 in **4 out of 5** datasets while using a decoupled, trainable stopping mechanism.

### Threshold Sensitivity

The Critic's confidence threshold controls the accuracy-efficiency trade-off:

| Threshold | FCB Acc | FCB Avg Steps | FacTool Acc | FacTool Avg Steps |
|:---------:|:-------:|:-------------:|:-----------:|:-----------------:|
| 0.3 | 75.6% | 2.06 | **85.0%** | 2.06 |
| 0.5 (argmax) | 77.2% | 2.44 | 82.8% | 2.30 |
| **0.7** | **82.7%** | 2.91 | 81.1% | 2.76 |

### Cross-LLM Generalization (FCB)

The Critic, trained only on GPT-4o-mini trajectories, generalizes to DeepSeek-V3:

| LLM | Method | True F1 | False F1 | Acc |
|:----|:-------|:-------:|:--------:|:---:|
| DeepSeek-V3 | No-Search | 0.644 | 0.519 | 59.1% |
| DeepSeek-V3 | **SWIFT** | **0.852** | **0.620** | **78.7%** |

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

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{peng2026swift,
  title     = {SWIFT: Self-learning When to Stop in Fact-checking Tasks},
  author    = {Peng, Zhiming},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association
               for Computational Linguistics (ACL)},
  year      = {2026}
}
```

---

## Acknowledgements

This work builds upon:
- [SIM-RAG](https://arxiv.org/abs/2410.17952) (SIGIR '25) — Self-practicing methodology for RAG
- [FIRE](https://github.com/mbzuai-nlp/fire) (NAACL '25) — Iterative retrieval for fact-checking
- [DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) — Pre-trained NLI model

We thank the authors for releasing their code and models.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
