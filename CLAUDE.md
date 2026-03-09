# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 交互规范

- 可以用英文思考，但**默认用中文回复用户**
- 论文相关内容（论文正文、公式推导、Related Work）用英文
- 请使用第一性原理思考。不要假设用户非常清楚自己想要什么和该怎么得到。保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来讨论。如果目标清晰但路径不是最佳，告诉用户并建议更好的办法

## 项目概述

SWIFT (Self-Learning When to Stop in Fact-Checking Tasks) — 用轻量级 DeBERTa-v3-base Critic (86M) 替代昂贵的 LLM 自评估，解决迭代 RAG 系统中 LLM 过度自信、搜索不充分的问题。

## 当前工作重点

**`swift_refine.md`** — 项目改进计划，包含两个核心组件：
- **Component A: Adversarial Evidence Retrieval** — 每步生成支持/反驳两个搜索查询
- **Component B: Verdict-Contrastive Stopping** — Critic 同时评估 True/False 两个假设，用 δ=|p_true-p_false| 作为 conclusiveness 指标

## 三阶段架构

```
Claims CSV → [Phase 1: Generation] → Training Data → [Phase 2: Training] → Critic Checkpoint → [Phase 3: Inference] → Predictions CSV
```

- **Phase 1 (`generation/`)**: LLM 迭代搜索 + 推理，通过 hindsight correctness 自标注训练数据
- **Phase 2 (`training/`)**: NLI 格式 (premise: claim+evidence+rationale, hypothesis: "judgment X is correct")，DeBERTa-v3-base 二分类微调
- **Phase 3 (`inference/`)**: 5 种推理模式 — `swift`(主方法), `nli_baseline`, `no_search`, `fixed_k`, `llm_critic`

## 常用命令

```bash
# Phase 1: 数据生成
python generation/generation.py --experiment_name swift_v5_t0 --search_engine ddg --temperature 0.7 --trajectory_id 0

# Phase 2: 训练数据准备 + Critic 训练
python training/prepare_training.py --experiment_name swift_v5
CUDA_VISIBLE_DEVICES=0 python training/main.py --experiment_name swift_v5 --fp16

# Phase 3: 推理 (SWIFT 模式)
python inference/inference.py --experiment_name fcb_swift --mode swift --critic_path checkpoints/swift_v5 --threshold 0.7

# 评估
python evaluate_results.py predictions/swift_v5_predictions.csv
python analyze_errors.py predictions/swift_v5_predictions.csv
```

## 关键设计决策

- **Critic 输入**: NLI premise-hypothesis 对，复用 MNLI/FEVER/ANLI 预训练知识
- **训练数据**: 自监督 hindsight correctness 标注，无需人工标注
- **Black-box 兼容**: 不需要 logprobs/attention，适用于任何 LLM API
- **Cross-LLM 泛化**: Critic 在一个 LLM 上训练可泛化到其他 LLM
- **全局配置**: `config.py` (数据路径、生成/推理参数), `training/config.py` (训练超参数)
- **LLM 调用**: `common/modeling.py` — OpenAI 兼容 API wrapper，带重试逻辑
- **API 密钥**: 复制 `common/shared_config.example.py` → `common/shared_config.py` 填入
