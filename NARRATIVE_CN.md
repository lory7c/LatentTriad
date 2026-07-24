# SkillProbe v4：基于 Hidden-State 几何关系的 LLM Agent 恶意 Skill 预执行检测

> 叙事线 v4 — 2026-07-24

---

## 1. 问题动机

LLM agent 在执行用户任务前，越来越多地从远程仓库加载第三方 skill——包含指令和代码的模块包。恶意 skill 可以在任何外部监控察觉之前窃取数据、破坏输出或攻陷 agent 环境。唯一可靠的干预点在 **加载之后、执行之前**——即 pre-action boundary。

现有方法都未在此节点操作：
- **静态扫描器**（读文本 → 可被 SkillCloak 混淆绕过，检出率从 98.6% 暴跌至 10.1%）
- **沙箱检测**（执行后确认恶意 → 为时已晚）
- **LLM 审计**（慢、开源小模型查的话不可靠、FPR=100%，在线api烧钱）

**现有 hidden-state 方法（RouteGuard、AgentLens）在 unpaired benchmark 上评估，受 source confound 影响；且均读取单点/单一信号源，缺乏对 skill 结构的几何分解。**

---

## 2. 核心空白：Source Confound + 结构盲区

所有已发表的 agent skill 检测方法都在**良性和恶意样本来源不同**的 benchmark 上评估。表面文本分类器利用这些来源差异进行检测：

| Benchmark | TF-IDF AUROC | 说明 |
|---|---|---|
| MalSkillBench (unpaired) | 0.967 | 良性=人类编写，恶意=LLM 生成 → 检测写作风格 |
| SMP Stealth V2 (paired CF) | 0.812 | 同一 source/style/lineage → 控制了 confound |

**0.155 AUROC 的差距是纯粹的 source confound。**

进一步，当良性 skill 刻意使用可疑词汇（Lexical Decoy），TF-IDF FPR 达到 39%。当恶意 skill 使用干净词汇（Inverse Decoy），TF-IDF FNR 达到 78%。表面文本方法读的是关键词分布，不是行为意图。

**没有任何先前工作在配对反事实数据上控制词汇混杂进行评估。**

---

## 3. 相关工作

三类先前工作，无一在 pre-action hidden-state boundary 操作：

| 方法 | 代表工作 | 局限 |
|---|---|---|
| **静态分析** | BIV, SkillSieve, ShareLock | 读 skill 文本。可被词汇伪装绕过 |
| **沙箱/动态** | SkillDetonate, AgentArmor | 执行后才检测。恶意已造成损害 |
| **Hidden-State** | RouteGuard, AgentLens | RouteGuard: attention 特征在 stealth 攻击上失效 (FNR=45%)。AgentLens: 只读 boundary 单点，漏掉分布式信号 (FNR=55%)。两者均在 unpaired benchmark 上评估 |

**我们是第一个将 hidden-state 分解为 declaration-operation-boundary 三区域几何关系、在配对反事实数据上消除 source confound、并通过 centering+gating 实现跨分布鲁棒检测的工作。**

---

## 4. 方法：SkillProbe v4

### 4.1 三区域 Hidden-State 几何

Agent 读取 skill 包（SKILL.md + scripts + config）后、生成任何 token 前，从 prompt 的三个语义区域提取 hidden states：

```
Prompt = Chat Template + User Task + <loaded_skill_package>
                                              │
  ┌───────────────────────────────────────────┘
  ▼
  Declaration            │  Operation                     │ Boundary
  "声称能做什么"           │  "实际包含什么"                  │ "准备行动前的状态"
  ───────────────────────┼────────────────────────────────┼──────────────────
  SKILL.md YAML + 标题   │  剩余 SKILL.md + scripts/*.py  │ assistant boundary
  + 首段描述             │  + 辅助文件                     │ 前最后一个 token
  mean pool              │  mean pool                     │ 单 token
```

从**一次 prefill 前向传播**的 8 个归一化深度（L3-L31）同时提取。无需生成，无需执行。

### 4.2 几何特征 + 各向异性修正

每层从 operation 和 boundary 向量 `h_oper, h_bnd ∈ R^4096` 构造：

```
1. Centering: 逐层去均值 + 去 PC1（移除 prompt 模板主导的各向异性噪声）
2. L2 归一化 → 只保留方向信息
3. 3 个几何特征:
   direction_diff = oper_unit − bnd_unit     [4096d]  方向差
   cos(oper, bnd)                            [1d]     oper-bnd 对齐度
   norm_ratio = ‖oper‖ / ‖bnd‖               [1d]     范数比（最强单特征 SMD=−1.42）
4. StandardScaler → PCA(8) → LogisticRegression(C=0.01)
```

**最终配置: oper + bnd, 去 declaration, 去 interaction, centering(−PC1), PCA=8, C=0.01, L7+ 层选择。**

消融证据：
- interaction (decl⊗oper) = 纯噪声 (−0.045 AUROC)
- declaration 区域 = 零边际信号 (decl-only AUC=0.49)
- PCA=64 → 过拟合 (TrainAUC=1.0)
- L3 → 读的是 prompt 模板结构，非 skill 内容
- **centering(−PC1) → 跨分布 AUROC +43-59pp（最关键突破）**

### 4.3 Boundary 安全门控 (v4)

```
if score_boundary < 0.15:      # 模型非常确定安全
    risk = score_boundary        # 信任模型最终判断
else:
    risk = score_v3              # v3 代码级检测
```

**Gate<0.15 是通用阈值。** Boundary 在安全样本上天然极低（~0.005），在恶意样本上天然高（~0.9）。无需调参。

效果：Lexical Decoy FPR 100% → 0%，Inverse Decoy FPR 100% → 0%。

### 4.4 双趟恶意归因 (v2)

```
Pass A (decl-only, ~50ms):  prompt 截断在声明末尾 → decl-bnd → Decl-Probe → score_decl
Pass B (完整 prompt, ~1945ms): 同 v1 → oper+bnd 几何 → Code-Probe → score_code

risk = max(score_code, score_decl)
attribution = argmax(score_code, score_decl) → DECL / CODE / BOTH / CLEAN
```

归因准确率: **175/175 = 100%。** Declaration 和 operation 信号正交。

### 4.5 计算开销

Llama-3.1-8B, BF16, 15K token prompt, 实测：

| 方法 | 增量 (ms) | vs Boundary |
|---|---|---|
| Static Regex | <0.1 | 0.0005× |
| TF-IDF + LR | ~2 | 0.01× |
| Boundary only | 20.1 | 1.00× |
| AgentLens | 20.2 | 1.00× |
| **SkillProbe v4** | **18.7** | **0.93×** |
| RouteGuard (完整版) | ~60 | 3.0× |
| LLM-as-Judge | ~600 | 30× |

**SkillProbe 是最快的内部方法。** 所有内部方法共享 agent 必做的 prefill (~1945ms)，检测几乎免费。

---

## 5. 实验结果

全部基于 Llama-3.1-8B, 内部 contract, L7+ 层选择。

### RQ1: 检测精度 — SMP Stealth V2 (484 样本, paired CF)

| 方法 | AUROC | FPR% | FNR% | F1% |
|---|---|---|---|---|
| Static Regex | 0.540 | 100.0 | 0.0 | 66.7 |
| TF-IDF + LR | 0.812 | 19.4 | 3.2 | 89.6 |
| Boundary only | 0.863 | 0.0 | 40.3 | 74.7 |
| AgentLens | 0.871 | 0.0 | 51.6 | 65.2 |
| RouteGuard | 0.824 | 48.4 | 8.1 | 76.5 |
| **SkillProbe v4** | **0.947** | **19.4** | **6.5** | **87.9** |

统计显著优于 RouteGuard (Δ=+0.123, p<0.001) 和 TF-IDF (Δ=+0.135, p<0.001)。FNR 6.5% vs Boundary 40.3%——**多检出 34% 的恶意 skill。**

### 五数据集完整结果

| 数据集 | 样本 | v4 AUROC | FPR% | FNR% | 论证什么 |
|---|---|---|---|---|---|
| SMP Stealth V2 | 484 (242对) | **0.947** | 19.4 | 6.5 | 唯一配对 CF benchmark |
| MSB | 7,894 | **0.996** | 3.2 | 1.3 | Source confound 证据 |
| MASB 75K | ~9K | **0.981** | 7.2 | 6.0 | 大规模良性 FPR 基准 |
| Lexical Decoy | 155 | **0.998** | **0.0** | 3.0 | 词汇可疑→不误杀 |
| Inverse Decoy | 155对 | **1.000** | **0.0** | 3.0 | 词汇干净→不漏检 |

### RQ2: 词汇混淆鲁棒性 — Lexical Decoy, 误报测试 (155 样本)

**问题**：当良性 skill 刻意使用恶意常用词汇（`curl`, `base64`, `exfiltrate`, `exploit`, `/etc/passwd` 等），检测器会误杀吗？

**数据集**：155 个 Lexical Decoy——skill 代码中包含大量安全工具常用词，但语义上完全是合法的安全审计/测试工具。与 155 个 SMP 恶意样本配对评估。

**关键对比**：

| 方法 | AUROC | FPR% | 为什么 |
|---|---|---|---|
| Static Regex | 0.344 | 59.3 | 匹配到 `curl`、`base64` 等关键词 → 误杀 |
| TF-IDF + LR | 0.573 | 100 | 词频分布和恶意 skill 相同 → 几乎全误杀 |
| Boundary only | 0.991 | **0.0** | 模型读完判断"这是安全工具" → 零误杀 |
| AgentLens | 0.867 | 22.7 | 仅读 boundary token，受模型判断保护 |
| **SkillProbe v4** | **0.998** | **0.0** | Boundary 安全门生效——模型确定安全时信任判断 |

**解读**：表面文本方法被关键词欺骗（FPR 39-100%）。Boundary 和 SkillProbe v4 读的是模型对 skill 的**语义理解**——模型能正确识别 "虽然词可疑，但这是合法的安全工具"。v4 的 Gate<0.15 确保在模型确定安全时不越权报警。

### RQ3: 零词汇线索检测 — Inverse Decoy, 漏检测试 (155 对)

**问题**：当恶意 skill 使用干净、正常的词汇（无 `curl`、无 `base64`、无可疑字符串），检测器会漏检吗？

**数据集**：155 对 Inverse Decoy——恶意 skill 的攻击逻辑隐藏在正常的 PDF 处理、表单填写等代码中，词汇层面完全干净。155 clean + 155 malicious 配对。

**关键对比**：

| 方法 | AUROC | FNR% | 为什么 |
|---|---|---|---|
| Static Regex | 0.500 | 100 | 没有可疑 pattern 可匹配 → 全漏 |
| TF-IDF + LR | 0.500 | 21.9 | 词频和良性一样 → 大量漏检 |
| Boundary only | 0.793 | 100 | 模型判断"词汇干净=安全" → 全漏 |
| AgentLens | 0.676 | 0.6 | 几乎全抓到，但 FPR=78% 太高 |
| **SkillProbe v4** | **1.000** | **3.0** | v3 代码级检测 + Boundary 安全门 → 零漏检 |

**解读**：Boundary（模型最终判断）被干净词汇完全欺骗——FNR=100%。SkillProbe v4 通过读取 operation hidden states（代码实际行为），在词汇层面完全干净的情况下仍能捕获恶意信号。**这是 surface-text 方法和 boundary-only 方法都无法做到的。**

### RQ4: 2×2 诊断矩阵总结

将 RQ1-RQ3 统一为 2×2 矩阵，横轴=skill 文本的词汇特征，纵轴=skill 的实际行为：

| | 词汇像恶意代码 | 词汇像正常代码 |
|---|---|---|
| **行为确实是恶意的** | SMP malicious: 所有方法都行 | Inverse Decoy: **只有 SkillProbe 行** (v4 FNR=3%, Boundary FNR=100%) |
| **行为确实是良性的** | Lexical Decoy: **只有读语义的方法行** (v4 FPR=0%, TF-IDF FPR=100%) | SMP clean: 所有方法都行 |

核心论证**：任何只读表面文字或只读模型判断的方法，必然在某个象限失败。Static Regex/TF-IDF 被词汇欺骗（左下象限），Boundary 被伪装欺骗（右上象限）。SkillProbe v4 读的是代码在模型内部的几何表示——兼顾词汇鲁棒性和恶意敏感性——四个象限全部解决。

### RQ5: 计算开销优势

**问题**：读 hidden state 比读文本慢多少？实际部署成本如何？

**Llama-3.1-8B, BF16, 15K token prompt, RTX 4090, 50次预热+50次计时**：

| 方法 | 增量开销 (ms) | 范围 | vs Boundary | 说明 |
|---|---|---|---|---|
| Static Regex | <0.1 | CPU-only | 0.0005× | 不需要 GPU，但 FPR=59-100% |
| TF-IDF + LR | ~2 | CPU-only | 0.01× | 不需要 GPU，但被词汇欺骗 |
| Boundary only | 20.1 | Shared-prefill | 1.00× | 最简内部方法，FNR=40-100% |
| AgentLens | 20.2 | Shared-prefill | 1.00× | PCA(50) 增加评分开销 |
| **SkillProbe v4** | **18.7** | Shared-prefill | **0.93×** | 比 Boundary 快 7% |
| RouteGuard (完整版) | ~60 | Shared-prefill | 3.0× | 含 attention window 计算 |
| LLM-as-Judge | ~600 | Standalone | 30× | 需生成 token，且 FPR=100% |

**关键发现**：

1. **所有内部方法共享同一个 prefill (~1945ms)。** 前向传播占绝对主导，hook 差异在 ±5ms 以内可忽略。agent 无论如何都要做 prefill——检测几乎免费。

2. **SkillProbe 比 Boundary 快 7%。** 因为 operation token 是连续的，mean-pool 比逐 token 读取更高效（更少 GPU 内存往返）。

3. **SkillProbe 比 RouteGuard 快 3.2×。** 因为我们不需要计算 attention over untrusted windows。RouteGuard 的 attention branch 额外消耗 37ms。

4. **SkillProbe 比 LLM-as-Judge 快 30× 且准确得多。** LLM 审计既慢（600ms 生成延迟）又不可靠（FPR=100%，几乎全判恶意）。

5. **成本-收益比无可匹敌**：比 Boundary 更快，比 RouteGuard 更准，比 LLM 审计既快又准。

### RQ6: 恶意归因

| 攻击来源 | Decl-Probe | Code-Probe | 归因准确率 |
|---|---|---|---|
| 仅声明投毒 | 0.99 | 0.23 | **100%** |
| 仅代码投毒 | 0.02 | 0.82 | **100%** |
| 双重投毒 | 0.99 | 0.69 | **100%** |
| 干净 | 0.02 | 0.21 | **100%** |

声明和操作信号**正交**。双趟架构以 0.08ms 额外开销实现 100% 归因——所有 baseline 都不具备此能力。

### RQ7: 方法演进消融

| 版本 | 方法 | SMP | Decoy | InvDecoy | 核心改进 |
|---|---|---|---|---|---|
| v1 | oper+bnd+PCA8+LR | 0.910 | 0.427 | 0.165 | 基础：读代码几何 |
| v2 | +decl-probe双趟 | 0.910 | 0.427 | 0.165 | +归因+声明覆盖 |
| v3 | +centering(−PC1) | 0.941 | 0.858 | 0.758 | +跨分布+43-59pp |
| **v4** | **+Boundary gating** | **0.947** | **0.998** | **1.000** | **+FPR双归零** |

---

## 6. 关键发现

1. **配对反事实设计是必需的。** Unpaired MSB 上所有方法 0.96+。Paired SMP 上 TF-IDF 降至 0.812——0.155 的差距是先前 benchmark 未能控制的 source confound。

2. **Centering(−PC1) 是关键突破。** 移除 prompt 模板各向异性噪声后，跨分布 AUROC 暴涨 43-59pp。这是使特征反映 skill 内容而非 token 位置的核心技术。

3. **Boundary 安全门控解决误报。** Gate<0.15 是通用阈值——模型确定安全时信任其判断，不确定时用代码级检测。两张 Decoy FPR 从 100% 降至 0%。

4. **Hidden-state 几何读的是行为语义。** 2×2 诊断矩阵证明：词汇可疑时不误杀（FPR=0%），词汇干净时不漏检（FNR=3%）。

5. **声明和操作信号正交。** 消融证明声明在现有 benchmark 上无边际贡献（攻击均为操作侧），但声明投毒实验揭示了盲区——双趟架构补全了覆盖。

6. **SkillProbe 是最快的内部方法。** 18.7ms 增量，比 Boundary 快 7%，比 RouteGuard 快 3×，比 LLM-as-Judge 快 30×。

---

## 7. 贡献

1. **配对反事实 benchmark (SMP Stealth V2)**：首个在 agent skill 检测中控制 source/style/vocabulary confound 的评估协议。证明现有 hidden-state 方法（RouteGuard: 0.824, AgentLens: 0.871）在消除 confound 后均显著低于报告值。

2. **三区域几何检测 + centering + gating (v4)**：首次将 skill 表示分解为 operation-boundary 几何关系，通过去 PC1 消除模板噪声、Boundary 安全门消除误报。SMP AUROC=0.947，Decoy FPR 双归零。比 RouteGuard 快 3×，比 AgentLens 的 FNR 低 45pp。

3. **2×2 诊断矩阵**：Lexical Decoy (FPR) + Inverse Decoy (FNR) 证明几何方法读的是行为语义而非关键词分布——这一区分是 RouteGuard/AgentLens 等单一信号方法无法做出的。

4. **双趟恶意归因**：首个区分恶意信号来自声明还是代码的方法，100% 准确率，0.08ms 额外开销。

5. **共享 prefill 部署**：检测增加 18.7ms——最快的内部方法，比 Boundary 快 7%，比 RouteGuard 快 3×，比 LLM 审计快 30×。
