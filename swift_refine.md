# SWIFT Refinement Plan: Adversarial Retrieval & Verdict-Contrastive Stopping

## 1. 核心动机与论文 Story (Motivation & Framing)

针对现有 iterative fact-checking 方法（特别是将其与通用 QA 区分开来），SWIFT 致力于解决以下两个 Domain-specific 痛点：

1. **检索层的确认偏差 (Confirmation Bias in Retrieval):** LLM 倾向于生成单向 query，寻找支持其初始判断的证据，导致证据池严重倾斜。
2. **停止决策的过度自信 (Overconfident Stopping):** 传统的单一置信度 (Credibility) 评估无法区分“证据确实充分且指向同一结论”和“证据存在巨大争议但碰巧有一条支持了当前判断”。

**核心 Novelty (The SIGIR Defense):** 通用 QA 任务的答案空间是开放式的，无法进行对立假设的对比。而 Fact-checking 的判断天然是二元的 (True/False)。SWIFT 利用这一**物理层面的结构性差异**，提出用单一 Critic 模型对“正反双重视角”进行对比评估，从而量化证据的“决定力 (Conclusiveness)”。

---

## 2. 核心架构设计 (Core Architecture)

新版 SWIFT 包含两个新增核心组件，并带有备选的 Critic 对比增强训练机制。

### Component A: 轻量级对抗检索 (Adversarial Evidence Retrieval)
* **机制:** LLM 在每一步生成检索请求时，强制派生出一正一反两个 Query。
* **作用:** 模拟 Mixture-of-Agents (MoA) 中的多视角 Proposers，构建包含 Support 和 Refute 两面信息的混合证据池。

### Component B: 决定力对比门控 (Verdict-Contrastive Stopping)
* **机制:** 使用同一个 DeBERTa Critic 模型 $C$，对同一批混合证据，分别测试 $True$ 和 $False$ 两个假设。
* **公式:** * 假设为真时的支持度: $p_{true} = C(claim, evidence, hypothesis=True)$
  * 假设为假时的支持度: $p_{false} = C(claim, evidence, hypothesis=False)$
  * 证据决定力 (Conclusiveness): $\delta = |p_{true} - p_{false}|$
* **双重门控停止条件 (Dual-Gated Stopping Condition):**
  系统仅在以下条件同时满足时停止搜索并输出结果：
  $$p_j^{LLM} \ge \tau_1 \land \delta \ge \tau_2$$
  *(其中 $p_j^{LLM}$ 为当前 LLM 判断的 Credibility，$\tau_1, \tau_2$ 为超参数阈值)*

---

## 3. 案例演示 (Case Study for Intuition)

* **Claim:** 阿司匹林可以治疗新冠肺炎。
* **Evidence Pool (Adversarial):** * [支持方] 早期观察性研究认为有益...
  * [反驳方] WHO 明确指出无效且有出血风险...
* **Contrastive Evaluation:**
  * 评估 $Hypothesis=True$: 因为存在 WHO 的强烈反驳，Critic 认为当前证据池不支持 True，给出 $p_{true} = 0.2$。
  * 评估 $Hypothesis=False$: WHO 的证据强烈支持 False，给出 $p_{false} = 0.85$。
* **Conclusiveness:** $\delta = |0.2 - 0.85| = 0.65$。
* **决策:** $\delta$ 较大（如 $\delta > \tau_2=0.5$），说明证据不再模棱两可，已明确倒向 False，系统安全停止搜索。

---

## 4. 给 AI Agent 的代码修改指南 (Implementation Guide for Agents)

Agent 请注意：当前任务时间极紧，请遵循“最小代码改动”原则。优先修改 Inference 流程进行 Pilot Test。

### Phase 1: 推理端修改 (Inference Only - 优先执行)

**1. `generation/prompts.py`**
* **目标:** 修改 LLM 状态生成的 Prompt 模板。
* **动作:** 在要求输出 JSON 的 schema 中，将原有的单一 `search_query` 替换为 `search_query_support` 和 `search_query_refute`。
* **⚠️ 关键：Refute Query 的 Prompt 措辞必须明确，不能只写 "a query to refute the claim"。** LLM 若只被告知"生成一个反驳查询"，会倾向于在 support query 前加 "not" 或否定词，两个 query 几乎携带相同的信息，对抗性完全失效。

  正确的 Prompt 指令应当是：
  * `search_query_support`: `"A search query to find evidence that SUPPORTS the current judgment. Focus on authoritative sources, direct factual evidence, or official statements."`
  * `search_query_refute`: `"A search query to find evidence that CHALLENGES or CONTRADICTS the current judgment. Think about: counterexamples, expert disagreement, known exceptions, alternative scientific interpretations, or fact-checking organizations that dispute this claim. The query must be substantively different from search_query_support."`
* **Pilot Test 时必须人工检查 Refute Query 质量：** 随机抽取 10 条 (support, refute) query 对，确认两者从不同角度出发。若超过 30% 的 refute query 只是 support query 的简单否定，需在 Prompt 中加入 few-shot 示例。

**2. `generation/generation.py`**
* **目标:** 适配双 Query 搜索。
* **动作:** 解析 LLM 返回的 JSON。并行调用搜索引擎（如 DuckDuckGo）分别执行这两个 Query，并将返回的 snippets 合并为一个全局 `evidence_pool` 供 Critic 使用。

**3. `inference/inference.py`**
* **目标:** 实现 Verdict-Contrastive 逻辑和双重门控。
* **动作:** 拦截送给 Critic 的输入，构造**两次非对称调用**（asymmetric calls）：

  | | claim | evidence | judgment | rationale |
  |---|---|---|---|---|
  | **调用 A（actual）** | ✅ | ✅ | LLM 当前判断（如 True） | ✅ 传入 LLM 原始 rationale |
  | **调用 B（opposite）** | ✅ | ✅ | 对立判断（如 False） | ❌ **传空字符串 `""`** |

  * **⚠️ 为什么调用 B 必须传空 rationale？**

    LLM 输出的 `rationale` 是"我为什么认为这个 claim 是 **True**"的推理过程。如果把这段支持 True 的推理传入 Critic，同时让 Critic 评估假设 `"The judgment False is correct"`，Critic 的 premise 和 hypothesis 就产生了**逻辑矛盾**：

    > Premise 在说："根据推理，True 是正确的"
    > Hypothesis 在问："False 是否正确？"

    DeBERTa 对这种内部矛盾的反应是不可预测的，会严重污染 $p_\text{false}$ 的数值，使 $\delta$ 变成噪声。传空 rationale 让调用 B 变成**纯粹的证据问题**——"仅凭这些证据，False 这个判断合理吗？"——信号干净且逻辑自洽。

  * 计算 $\delta = |p_\text{actual} - p_\text{opposite}|$
  * 停止条件：`p_actual >= τ₁ AND δ >= τ₂`
  * **Debug 必须打印：** `[step={t}] judgment={j}, p_actual={:.3f}, p_opposite={:.3f}, δ={:.3f}, stop={bool}`

### Phase 2: 训练端增强 (Contrastive Retraining - 视 Phase 1 结果执行)
*(仅当旧 Critic 计算出的 $\delta$ 区分度不足、呈现随机噪声时执行此步)*

**1. `training/data_processing.py`**
* **目标:** 对现有的 self-practicing 轨迹数据进行离线增强 (Data Augmentation)。
* **动作:** 遍历现有的训练样本。对于每一个 `(Claim, Evidence_pool, Ground_Truth)`，派生出两条数据：
  * **样本 A (Positive/Negative):** Hypothesis 设为 True。如果 Ground_Truth == True，Label=1；否则 Label=0。
  * **样本 B (Positive/Negative):** Hypothesis 设为 False。如果 Ground_Truth == False，Label=1；否则 Label=0。
* **效果:** 原训练集大小翻倍，强制模型学习在同一证据池下，True 和 False 假设的支持度差异。然后运行标准训练脚本即可。
  * **注意：派生样本 A 和 B 中的 rationale 都传空字符串 `""`**，与推理时调用 B 的逻辑保持一致，避免引入新的 train/inference 不一致性。

---

## 5. Self-Learning 有效性分析 (Does Self-Practicing Still Work?)

这是改动后最需要回答的核心问题。**结论：Self-Practicing 的核心逻辑完全保留，但存在一个 train-inference distribution gap，Phase 2 重训是修复该 gap 的关键。**

### 什么变了，什么没变

| 组件 | 现有 SWIFT | 新 SWIFT | 是否需要重新运行 |
|------|-----------|---------|----------------|
| 轨迹数据生成 | 单向 query | **不变**（Generation 脚本不改） | ❌ 不需要重跑 |
| Self-Practicing 标注逻辑 | verdict = (judgment == ground_truth) | **不变** | ❌ 不需要重跑 |
| Critic 推理时的输入分布 | 单向证据 + rationale | 双向混合证据 + rationale 非对称 | Phase 1 直接适配 |
| Critic 训练数据格式 | 单向证据，单一 hypothesis 方向 | 需增加对立 hypothesis 方向样本 | Phase 2 离线增强 |

### Train-Inference Distribution Gap（必须正视）

**问题**：现有 Critic 训练时只见过"单向证据 + 单一 hypothesis 方向"的样本。推理时突然出现两个变化：(1) 证据池里混入了反驳方向的 snippets，(2) 需要对 opposite hypothesis 做打分（训练时从未出现过 rationale 为空的 opposite 样本）。

**严重程度**：中等。DeBERTa 的 NLI 预训练对 entailment/contradiction 有底层理解，Phase 1 pilot test 的目的就是实测这个 gap 到底有多大。

**Phase 2 如何修复（无需重跑 LLM generation，纯离线操作）：**

```
现有训练样本:
  (claim, evidence_single, judgment=True, rationale="因为...", verdict=1)

↓ 派生出两条对比样本:
  样本 A:  (claim, evidence_single, hypothesis=True,  rationale="",  label = ground_truth==True  ? 1 : 0)
  样本 B:  (claim, evidence_single, hypothesis=False, rationale="",  label = ground_truth==False ? 1 : 0)
```

两条样本用同一批证据，不同 hypothesis 方向，强制 Critic 学会区分"证据支持 True"和"证据支持 False"的程度差异，从而让 δ 有实质意义。

---

## 6. 极速执行 Checkpoints

- [ ] **Check 1:** 改好三个文件（`prompts.py` / `generation.py` / `inference.py`），选取 **50 条 FCB dev set** 试跑。
- [ ] **Check 2 — Refute Query 质量检查:** 人工抽看 10 条 (support, refute) query 对，确认 refute query 不只是 support 的简单否定。如质量差，加强 prompt 后重跑。
- [ ] **Check 3 — δ 区分度检查:** 观察日志中 δ 的分布。
  * **δ 有区分度**（clear case 的 δ 普遍 > 0.3，ambiguous case 的 δ < 0.2）→ 旧 Critic 可用，直接跑全量实验。
  * **δ 接近随机噪声**（分布集中在 0.1–0.2，无规律）→ 执行 Phase 2 数据增强 + 1 小时 A40 重训，随后重回 Check 3。
- [ ] **Check 4:** 全量实验完成，对比 SWIFT（原）vs SWIFT-Adv（新）的 Accuracy / F1 / Avg Steps，验证改进显著性。