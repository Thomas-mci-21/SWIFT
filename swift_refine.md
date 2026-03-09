# SWIFT: Self-Supervised Recoverability Learning for Fact-Checking

## 1. 核心问题定义

SWIFT 要解决的核心问题，不应再表述为“模型现在是否足够自信”，也不只是“下一步搜什么更有帮助”，而是：

> In iterative fact-checking, the central question is whether the current verdict is still recoverable by retrieval.

更具体地说，系统在任意一个中间 evidence state 上，需要判断的不是 generic confidence，而是：

**how much of the current stop-time error risk remains reducible under bounded retrieval intervention**。

也就是，面对当前状态，系统要回答三件事：

1. 如果现在停止并提交当前 verdict，出错风险有多高
2. 这部分风险中，还有多少是可以通过额外检索纠正的
3. 如果仍然可纠正，应该执行哪一种检索干预去实现这种纠正

这里最关键的对象不是 confidence，也不是 retrieval utility，而是：

**verdict recoverability**。

SWIFT 学习的是：从同一个中间证据状态出发，当前 verdict 是否仍然存在值得做的检索纠错空间。

---

## 2. 为什么这是 fact-checking 特有的问题

### 2.1 Fact-checking 的 stopping 本质上是 recoverability judgment

开放式 QA 的 stopping 往往围绕答案完整性：还缺不缺信息，答案是否已经足够完整。

但 binary fact-checking 的 stopping 语义不同。系统并不是在追求“知道更多”，而是在决定：

1. 当前 verdict 如果现在提交，会不会错
2. 如果会错，这个错误是否还来得及被检索纠正

因此，fact-checking 中真正合理的 stopping 定义不是“模型够不够确定”，而是：

> stop only when the remaining error risk of the current verdict is no longer materially recoverable by any affordable retrieval intervention.

这里的 stopping 不是 completeness control，而是 **recoverable risk control**。

### 2.2 检索在这里不是累积证据，而是执行纠错干预

fact-checking 的困难通常不只是证据不够，而是当前状态常常同时包含：

1. 支持与反驳并存
2. 弱来源与强来源混杂
3. 表面支持但语义错位的材料
4. 缺少能够裁决冲突的 authoritative evidence

因此，retrieval 在这里不应被理解为被动补充信息，而应理解为对当前 verdict 的主动纠错干预。系统下一步检索，不是为了“搜得更多”，而是为了改变当前 verdict 的未来错误风险。

换句话说，retrieval 的作用是：

1. 暴露当前 verdict 的脆弱性
2. 验证当前 verdict 是否站得住
3. 寻找能够裁决冲突的决定性证据

所以，SWIFT 的核心不是 confidence estimation，而是判断：

> is the current verdict already reliable enough to stop, or is it still salvageable by retrieval?

---

## 3. 方法主核：recoverable risk learning

### 3.1 状态定义

记第 $t$ 步的中间状态为：

$$
h_t = (c, E_t, \hat{y}_t)
$$

其中：

1. $c$ 是 claim
2. $E_t$ 是当前 evidence pool
3. $\hat{y}_t \in \{\mathrm{True}, \mathrm{False}\}$ 是当前 verdict

这个状态不是一句 isolated judgment，而是一个待裁决的证据局面。

### 3.2 当前停止风险

对状态 $h_t$，先定义当前若立即停止的错误风险：

$$
r_{\mathrm{stop}}(h_t)
$$

它表示：

> if we stop now and submit the current verdict, how likely is this verdict to be wrong?

这里的 $r_{\mathrm{stop}}$ 不是一般意义上的 uncertainty，而是一个决策绑定的量：

**the stop-time error risk of the current verdict**。

### 3.3 可达到风险与可纠正风险

关键创新不应停留在“再做一步值不值”，而应进一步定义：从当前状态出发，在给定检索预算内，系统理论上还能把风险降到多低。

定义 bounded retrieval intervention set 为 $\mathcal{A}$，则对应的最优可达到风险为：

$$
r_{\mathrm{reach}}(h_t) = \min_{a \in \mathcal{A}} \mathbb{E}[r_{\mathrm{stop}}(h_{t+1}^{(a)}) \mid h_t, a]
$$

也就是：从当前状态出发，在允许的一步干预或固定小预算干预下，未来 stop-time risk 最多还能被降到什么程度。

于是，定义当前状态的**可纠正风险**为：

$$
g(h_t) = r_{\mathrm{stop}}(h_t) - r_{\mathrm{reach}}(h_t)
$$

这个量表示：

> how much of the current stop-time error risk is still reducible by retrieval intervention.

它是整篇方法最核心的对象。

SWIFT 的 stopping 决策，学习的不是 verdict 是否正确本身，而是：

1. 当前若停下有多危险
2. 这个危险里还有多少是可纠正的

### 3.4 三类 intervention 的角色

为了让可纠正风险可被估计，SWIFT 使用三类结构化 retrieval intervention：

$$
a \in \mathcal{A} = \{\text{support}, \text{refute}, \text{resolve}\}
$$

它们的作用不是构成方法主核，而是作为对 verdict recoverability 的三种探测方式：

1. support：测试当前 verdict 是否能被更强证据稳固
2. refute：测试当前 verdict 是否存在易被击穿的反证
3. resolve：测试当前冲突是否可被 authoritative evidence 裁决

因此，support/refute/resolve 不是 query diversification trick，也不是独立组件；它们是 **recoverability probes**。

---

## 4. Self-Learning 机制

SWIFT 的 self-learning 不再写成“从 hindsight correctness 学一个 critic”，而应当写成：

> SWIFT self-supervises verdict recoverability from state-matched counterfactual intervention branches.

核心约束只有两个：

1. 同一个训练样本必须锚定在同一个中间 state
2. 不同 branch 之间只允许第一步 intervention 不同

从同一个 anchor state 出发，系统分叉 support、refute、resolve 三种干预，然后比较不同 branch 对未来 verdict risk 的影响。

这时监督信号学到的就不是“哪条轨迹最后成功了”，而是：

1. 当前状态本身是否危险
2. 这个危险是否仍然可被纠正
3. 哪种 intervention 最可能实现这种纠正

这与单条 trajectory hindsight labeling 的本质区别在于：

SWIFT 学的不是静态置信度，也不是单路径 hindsight utility，而是 **same-state recoverability under counterfactual retrieval interventions**。

---

## 5. 训练信号定义

### 5.1 基本训练单位：anchor state 与 branch set

训练样本的基本单位不是单步 query，也不是单条完整 trajectory，而是一个 anchor state：

$$
s_t = (c, E_t, \hat{y}_t)
$$

围绕这个 anchor state，构造一个小型 counterfactual branch set：

$$
\mathcal{B}(s_t) = \{b^{\text{support}}, b^{\text{refute}}, b^{\text{resolve}}\}
$$

每个 branch 都满足：

1. 起点 state 完全相同
2. 仅第一步 intervention 不同
3. 后续 rollout 使用共享预算和共享生成协议

这样比较的对象就不是不同初始条件下的不同轨迹，而是同一状态下不同干预的未来结果。

### 5.2 当前停止风险的监督

对 anchor state，定义 stop-time risk target：

$$
r^{*}(s_t)
$$

它表示当前 verdict 在该 state 上如果立即停止的经验错误风险。

在最直接的实现里，它可以由当前 verdict 相对 ground truth 的 stop-loss 给出；在更稳健的实现里，也可以通过同 state 上多次重采样 continuation 进行平滑估计。

关键点是：

$r^{*}(s_t)$ 必须始终绑定到“现在就停”这一决策，而不是绑定到最终整条轨迹的整体质量。

### 5.3 动作条件可纠正性的监督

对每个 intervention $a$，定义执行该干预后未来风险的经验目标：

$$
r^{*}_{\mathrm{post}}(s_t, a)
$$

它表示：从同一个 anchor state 出发，先执行动作 $a$，再在固定 rollout 协议下继续，最终可达到的 stop-time risk 有多低。

于是可得到动作条件风险下降：

$$
\Delta^{*}(s_t, a) = r^{*}(s_t) - r^{*}_{\mathrm{post}}(s_t, a)
$$

以及 anchor state 的总可纠正风险：

$$
g^{*}(s_t) = \max_{a \in \mathcal{A}} \Delta^{*}(s_t, a)
$$

这里的 $g^{*}(s_t)$ 是最关键的 supervision target，因为它直接对应：

> whether the current verdict is still salvageable by some retrieval intervention.

### 5.4 训练目标

训练上建议保留三类互相约束、但围绕同一中心对象的目标：

1. 一个 calibrated loss 拟合 $r^{*}(s_t)$
2. 一个 regression 或 ordinal loss 拟合 $g^{*}(s_t)$ 与 $\Delta^{*}(s_t, a)$
3. 一个 pairwise ranking loss，要求在同一 anchor state 下，真正更能降风险的 intervention 排名更高

这样做的含义是：

1. 模型既知道现在停有多危险
2. 又知道危险里有多少仍可纠正
3. 还知道纠正应优先走哪种干预

但整套训练仍然只围绕一个主核：**recoverable risk estimation**。

---

## 6. 推理时的方法形式

### 6.1 模型输出对象

推理时，模型对每个状态 $h_t$ 输出：

$$
R(h_t) \approx r_{\mathrm{stop}}(h_t)
$$

$$
G(h_t) \approx g(h_t)
$$

并可选地输出动作级分解：

$$
Q(h_t, a) \approx \Delta(h_t, a), \quad a \in \mathcal{A}
$$

这里应该注意优先级：

1. 主输出对象是 $R(h_t)$ 和 $G(h_t)$
2. $Q(h_t, a)$ 是对 $G(h_t)$ 的动作级展开

也就是说，方法首先判断“还可不可以被纠正”，然后才决定“该怎么纠正”。

### 6.2 停止规则

因此，最终 stopping rule 应写成：

$$
\mathrm{STOP}(t) \iff R(h_t) \le \alpha \quad \text{or} \quad G(h_t) \le c
$$

如果希望同时更保守，也可以写成：

$$
\mathrm{STOP}(t) \iff R(h_t) \le \alpha \quad \text{and} \quad G(h_t) \le c
$$

其中：

1. $\alpha$ 控制当前可接受的 stop-time risk
2. $c$ 控制继续检索所需的最小可纠正收益

这个 stopping rule 的语义是：

1. 如果当前风险已经足够低，可以停
2. 即便当前风险不低，但如果已经几乎不可纠正，也应该停
3. 只有当风险仍显著且仍可被检索纠正时，才值得继续

这使得 stopping policy 不再只是 confidence thresholding，而是一个 **risk-and-recoverability decision**。

### 6.3 下一步动作选择

若不停止，则选择：

$$
a_t^{*} = \arg\max_{a \in \mathcal{A}} Q(h_t, a)
$$

也就是，执行最能降低当前 stop-time risk 的干预。

因此，retrieval policy 与 stopping policy 是被同一个中心量耦合起来的：

**remaining recoverable risk**。

---

## 7. 与已有方向的边界

### 7.1 与 confidence estimation 的区别

confidence estimation 关注的是 belief strength：模型现在有多确定。

SWIFT 关注的是 recoverability：模型现在即便可能错，这个错误是否还能被检索纠正。

因此，二者回答的是不同问题：

1. confidence asks whether the model feels certain
2. SWIFT asks whether retrieval can still change the correctness of stopping now

### 7.2 与 generic retrieval value learning 的区别

generic retrieval value 往往只问：再搜一步对整体任务有没有收益。

SWIFT 学的不是一般任务收益，而是与当前 verdict 紧耦合的 recoverable risk reduction：

1. 目标只围绕当前 verdict 定义
2. 价值只以 stop-time error risk 的变化衡量
3. 监督只在 same-state counterfactual branches 上成立

所以，这里不是一般的 value-of-information，而是 **verdict-specific recoverability learning**。

### 7.3 与 query diversification 的区别

support/refute/resolve 不是为了让检索结果更丰富，而是为了从不同方向测试当前 verdict 是否仍可纠正：

1. support 检查 verdict 是否可被稳固
2. refute 检查 verdict 是否存在致命反证
3. resolve 检查冲突是否可被裁决

它们是 probe，不是方法目标本身。

### 7.4 与静态 contrastive training 的区别

这里的对比不发生在固定样本对之间，而发生在同一个中间 state 的可行动作之间。

比较对象不是 evidence pair，也不是 label pair，而是：

**which intervention can still rescue the current verdict state**。

---

## 8. 论文摘要与贡献表述

### 8.1 一句话摘要

> SWIFT reformulates iterative fact-checking as estimating whether the current verdict remains recoverable by retrieval. Instead of thresholding confidence or learning from a single hindsight trajectory, it self-supervises state-matched counterfactual intervention branches to estimate stop-time risk, remaining recoverable risk, and the intervention most likely to reduce that risk.

### 8.2 三条主贡献

建议只保留以下三条贡献：

1. A fact-checking-specific formulation of iterative verification as estimating whether the current verdict remains recoverable under bounded retrieval interventions, rather than merely whether it appears confident.
2. A self-supervised same-state counterfactual learning framework that trains on intervention branches from the same evidence state to estimate stop-time risk and remaining recoverable risk.
3. A unified policy that retrieves only when the current verdict is still salvageable by some intervention and otherwise stops, tying retrieval choice and stopping to the same recoverability signal.

### 8.3 只作为 supporting details 的内容

以下内容只能作为 supporting details，而不应再承担主方法叙事：

1. 轻量 critic 替代 LLM self-assessment
2. cross-LLM transfer
3. dual-hypothesis representation
4. support/refute/resolve 的具体 prompt 设计
5. calibration 和 threshold analysis

这些点可以增强结果，但不能再定义方法主核。

---

## 9. 一个具体例子

Claim：阿司匹林可以治疗新冠肺炎。

当前状态中已经有：

1. 一些早期观察性研究，表面上似乎支持
2. WHO 和系统综述，整体上更倾向反驳
3. 若干媒体材料把“降低并发症风险”和“治疗 COVID-19”混为一谈

此时系统不该只问“我现在更像 True 还是 False”，而是要问：

1. 如果现在判 False 并立刻停止，当前错误风险还有多高
2. 这部分风险中，还有多少是可被下一步检索纠正的
3. 哪种 intervention 最可能真正降低这部分风险

在这个例子里，support 和 refute 可能都只会继续堆叠已经存在的片面材料；真正高价值的往往是 resolve，因为当前问题不在于证据数量不够，而在于冲突的争议点没有被 authoritative source 裁决。

因此，系统更应该做的是：

1. 抽出冲突核心，到底是“治疗”还是“并发症相关效应”
2. 定向寻找 WHO、NIH、Cochrane 等权威证据
3. 判断这一步是否真正把 stop-time risk 压低

如果 resolve 之后当前风险已经很低，或虽然仍有风险但已经几乎不可纠正，那系统才应该停。

所以，停止的依据不是“我看起来挺自信”，而是：

**the remaining error is no longer worth trying to rescue**。

---

## 10. 实现指导

### 10.1 Generation

generation 阶段的核心任务不是简单多采样，而是构造可比较的 anchor states 与 intervention branches。

至少需要：

1. 一条主 trajectory 用于暴露自然中间状态
2. 若干可锚定的 anchor states
3. 从每个 anchor state 分叉出的 support/refute/resolve branches
4. 固定预算下的后续 rollout，用于估计 branch 的 post-intervention risk

关键不是 branch 数量本身，而是保证“同一 state 下不同 intervention 的未来后果可比较”。

### 10.2 Training

训练时应把主输出收缩到两个中心对象：

1. 当前 stop-time risk
2. 当前 remaining recoverable risk

动作级 $Q(h_t, a)$ 可以存在，但应被视为对 recoverability 的分解，而不是新的独立任务。

如果需要增加复杂度，复杂度也应只体现在这两个对象的监督方式上，例如：

1. state-matched branch aggregation
2. pairwise action ranking
3. conservative target estimation
4. risk calibration

而不是继续叠加新的 standalone modules。

### 10.3 Inference

推理时每一步只做三件事：

1. 估计当前如果停下的风险
2. 估计这部分风险还有多少可被检索纠正
3. 若仍可纠正，则执行最有希望降低该风险的 intervention；否则停止

于是 retrieval policy 和 stopping policy 由同一个对象共同定义：

**whether the current verdict remains recoverable**。

---

## 11. 方法边界与 paper positioning

这篇论文最应该守住的边界是：

1. 它不是一个更复杂的 stopping heuristic
2. 它不是一个 query diversification scheme
3. 它不是一个对抗式检索组件堆叠
4. 它也不是把 dual-hypothesis 或 calibration 拿来重新包装

它真正的新意应该被压缩成一句明确的话：

> SWIFT introduces self-supervised recoverability learning for iterative fact-checking: from the same evidence state, it learns whether the current verdict can still be rescued by retrieval, and stops when it can no longer be materially corrected.

这里真正要让评审记住的关键词只有三个：

1. same-state
2. recoverability
3. stop-time risk

support/refute/resolve、dual-hypothesis、lightweight critic、cross-LLM transfer 都只能为这三个词服务。

---

## 12. 一句话总结

整篇论文最终可以压成一句话：

> SWIFT is not about estimating whether the current verdict looks correct; it is about self-supervising, from the same evidence state, whether that verdict remains recoverable by retrieval, how much of its stop-time risk is still reducible, and stopping when no intervention can materially rescue it.

如果还要进一步压缩，最核心的 method identity 就是：

**same-state self-supervised recoverable-risk learning for fact-checking**。
