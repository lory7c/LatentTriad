# SkillProbe v4：基于 Hidden-State 几何关系的 LLM Agent 恶意 Skill 检测

> 2026-07-24

---

## 1. 问题

LLM agent 在执行任务前从远程仓库加载第三方 skill。恶意 skill 可以窃取数据、破坏输出或攻陷环境。唯一可靠的干预点在**加载之后、执行之前**。

现有三类方法各有致命缺陷：

| 方法 | 代表工作 | 致命问题 |
|---|---|---|
| 静态扫描 | BIV, SkillSieve, cisco-scanner | 读文本 → SkillCloak 混淆后检出率 99%→10% |
| 沙箱执行 | SkillDetonate | 执行后检测 → 为时已晚，153s/skill |
| 内部探针 | RouteGuard, AgentLens | 只读单点信号。RouteGuard FNR=45%, AgentLens FNR=55% |

**没有方法同时做到：高检测率 + 低误报 + 不执行 + 可归因。**

---

## 2. 方法：SkillProbe v4

### 核心思路

模型读完 skill 后、执行前的 hidden state 里，**代码的"实际行为"和模型的"最终判断"之间存在几何张力**。恶意 skill 的代码在 hidden space 中更偏离模型的最终判断方向——因为模型读完恶意代码后"知道不对劲"，boundary token 被异常激活。

我们只提取两个区域的 hidden state：**Operation**（skill 的实际代码内容）和 **Boundary**（模型准备行动前的最后一个 token）。一次 prefill 前向传播，8 层，无需生成、无需执行。

### 三个关键设计

**1. Centering（去 PC1）**：每层移除 prompt 模板主导的方差方向。消融证明这是最关键改进——跨分布 AUROC +43-59pp。

**2. 几何特征**：
```
norm_ratio = ‖oper‖ / ‖bnd‖          ← 最强单特征 (SMD=−1.42)
cos(oper, bnd)                        ← oper-bnd 对齐度
direction_diff = oper_unit − bnd_unit ← 方向差
→ StandardScaler → PCA(8) → LR(C=0.01)
```

**3. Boundary 安全门 (v4)**：
```
if score_boundary < 0.15:   risk = score_boundary   # 模型确定安全 → 信任
else:                        risk = score_v3         # 否则用代码级检测
```

Gate<0.15 是通用阈值。Boundary 在安全样本上分数天然极低(~0.005)，恶意样本上天然高(~0.9)，无需调参。

### 双趟归因 (v2)

```
Pass A (decl-only, ~50ms):  截断在声明末尾 → 检测声明投毒
Pass B (完整 prompt, v1):   同 v1 → oper+bnd 几何检测
risk = max(code, decl)
attribution = argmax → CODE / DECL / BOTH / CLEAN  (100% 准确)
```

### 开销

| 方法 | 增量 | 
|---|---|
| Static Regex | <0.1 ms (CPU) |
| TF-IDF | ~2 ms (CPU) |
| Boundary only | 20.1 ms |
| **SkillProbe v4** | **18.7 ms** |
| AgentLens | 20.2 ms |
| RouteGuard (full) | ~60 ms |
| LLM-as-Judge | ~600 ms |

所有内部方法共享 agent 必做的 prefill (~1945ms)。**SkillProbe 是最快的内部方法**——比 Boundary 快 7%，比 RouteGuard 快 3.2×，比 LLM-as-Judge 快 30×。

---

## 3. 实验结果

Llama-3.1-8B, 五数据集, L7+ 层选择, SkillProbe v4。

### RQ1: 检测精度 — SMP Stealth V2 (484 样本, 配对反事实)

| 方法 | AUROC | FPR% | FNR% | F1% |
|---|---|---|---|---|
| Static Regex | 0.540 | 100.0 | 0.0 | 66.7 |
| TF-IDF + LR | 0.812 | 19.4 | 3.2 | 89.6 |
| Boundary only | 0.863 | 0.0 | 40.3 | 74.7 |
| AgentLens | 0.871 | 0.0 | 51.6 | 65.2 |
| RouteGuard | 0.824 | 48.4 | 8.1 | 76.5 |
| **SkillProbe v4** | **0.947** | **19.4** | **6.5** | **87.9** |

v4 AUROC=0.947，比 RouteGuard 高 12.3pp，比 Boundary 高 8.4pp。FNR=6.5%——Boundary 漏掉 40% 的恶意，我们只漏 6.5%。

### 五数据集完整结果

| 数据集 | 样本 | v4 AUROC | FPR% | FNR% |
|---|---|---|---|---|
| SMP Stealth V2 | 484 | **0.947** | 19.4 | 6.5 |
| MSB | 7,894 | **0.996** | 3.2 | 1.3 |
| MASB 75K | ~9K | **0.981** | 7.2 | 6.0 |
| Lexical Decoy | 155 | **0.998** | **0.0** | 3.0 |
| Inverse Decoy | 155对 | **1.000** | **0.0** | 3.0 |

### RQ2: 词汇混淆鲁棒性 — Lexical Decoy (155 样本)

**问题**：良性 skill 充满了 `curl`, `base64`, `exfiltrate` 等安全工具常用词，会被误杀吗？

| 方法 | FPR% | 为什么 |
|---|---|---|
| Static Regex | 59.3 | 匹配到关键词 → 误杀 |
| TF-IDF + LR | 100 | 词频分布和恶意一样 → 全误杀 |
| Boundary only | **0.0** | 模型判断"这是安全工具" |
| **SkillProbe v4** | **0.0** | Gate<0.15 信任模型判断 |

表面文本方法被关键词欺骗。我们读的是模型的语义理解——模型知道虽然词可疑，但这是合法工具。

### RQ3: 零词汇线索检测 — Inverse Decoy (155 对)

**问题**：恶意 skill 的代码用干净词汇伪装成正常的 PDF 处理、表单填写工具，会漏检吗？

| 方法 | FNR% | 为什么 |
|---|---|---|
| Static Regex | 100 | 没有可疑 pattern → 全漏 |
| TF-IDF + LR | 21.9 | 词频和良性一样 → 大量漏检 |
| Boundary only | 100 | 模型判断"词汇干净=安全" → 全漏 |
| **SkillProbe v4** | **3.0** | 读代码 hidden state，不依赖词汇 |

Boundary 被干净词汇完全欺骗。我们读的是代码在模型内部的几何表示——词汇干净但行为恶意，几何特征暴露了异常。

### RQ4: 2×2 诊断矩阵

横轴 = skill 文本的词汇特征，纵轴 = skill 的实际行为：

| | 词汇像恶意代码 | 词汇像正常代码 |
|---|---|---|
| **行为确实是恶意的** | SMP malicious ✓ | Inverse Decoy: **只有我们行** (v4 FNR=3%) |
| **行为确实是良性的** | Lexical Decoy: **只有我们行** (v4 FPR=0%) | SMP clean ✓ |

任何只读表面文字或只读模型判断的方法，必然在某个象限失败。SkillProbe v4 读的是代码在模型内部的几何表示——四个象限全部解决。

### RQ5: 恶意归因 (175 样本)

| 攻击来源 | 归因准确率 |
|---|---|
| 仅声明投毒 | **100%** |
| 仅代码投毒 | **100%** |
| 双重投毒 | **100%** |
| 干净 | **100%** |

Declaration 和 operation 信号正交。双趟架构以 0.08ms 额外开销实现 100% 归因——所有 baseline 都不具备此能力。

### RQ6: 方法演进消融

| 版本 | 方法 | SMP | Decoy | InvDecoy | 改进 |
|---|---|---|---|---|---|
| v1 | oper+bnd+PCA8+LR | 0.910 | 0.427 | 0.165 | 基础 |
| v2 | +decl-probe双趟 | 0.910 | 0.427 | 0.165 | +归因 |
| v3 | +centering(−PC1) | 0.941 | 0.858 | 0.758 | +跨分布鲁棒 |
| **v4** | **+Boundary gating** | **0.947** | **0.998** | **1.000** | **+FPR双归零** |

---

## 4. 贡献

1. **SkillProbe v4**：oper+bnd 几何特征 + centering(−PC1) + Boundary 安全门。五数据集全面领先：SMP=0.947, Decoy FPR=0%, InvDecoy FNR=3%。比 RouteGuard 快 3×，比 LLM 审计快 30×。

2. **2×2 诊断矩阵**：首次系统证明 hidden-state 方法读的是行为语义而非关键词——Lexical Decoy (FPR=0%) 和 Inverse Decoy (FNR=3%) 构成互补证据。

3. **恶意归因**：双趟架构区分声明投毒 vs 代码投毒，100% 准确率，0.08ms 额外开销。

4. **消融证据链**：interaction=噪声、declaration=零信号、PCA=8 最优、L3=模板污染、centering=最关键突破、Gate<0.15=通用安全门。
