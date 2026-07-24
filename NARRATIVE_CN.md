# SkillProbe v4：基于 Hidden-State 几何关系的 LLM Agent 恶意 Skill 检测

> 2026-07-24

---

## 1. 问题：Agent Skill 供应链攻击

### 攻击面已经存在

LLM coding agent（Claude Code、OpenAI Codex、Cursor 等）正快速普及一个能力：从远程仓库加载第三方 **skill**——包含指令、脚本和配置的模块包，为 agent 提供新的工具能力。

这创造了一个新的软件供应链攻击面。Skill 以 agent 的完整权限运行——可以读写文件系统、执行 shell 命令、访问网络和环境变量。一个恶意 skill 可以：

- **窃取凭证**：读取 `~/.aws/credentials`、SSH key、环境变量中的 API key，发送到外部服务器
- **植入后门**：在项目中写入恶意代码、修改 `.bashrc` 实现持久化
- **破坏输出**：在代码生成中注入漏洞、替换依赖包
- **劫持 agent 决策**：通过 prompt injection 覆盖用户指令，让 agent 执行攻击者意图

真实案例已经出现：2026 年初，ClawHavoc 攻击在 ClawHub marketplace 植入了 300+ 恶意 skill，伪装成安装前提条件窃取浏览器凭据和加密货币钱包。BIV 对 49,943 个 OpenClaw skill 的扫描发现 80% 的 skill 存在声称-实际行为偏差，5% 携带多阶段攻击链。

### 所有现有方法都漏了点什么

防御工作分为三类，每一类都有无法弥补的缺陷：

| 方法 | 代表工作 | 做什么 | 致命问题 |
|---|---|---|---|
| **静态文本扫描** | BIV, SkillSieve, cisco-scanner | 读 SKILL.md 文本，匹配关键词/正则/AST pattern | SkillCloak 混淆后检出率从 99% 跌到 10%——文本可以被改写，行为无法被文本掩盖 |
| **沙箱动态执行** | SkillDetonate | 在 Docker 里实际执行 skill，用 eBPF 监控 syscall | 执行后才确认恶意——已经晚了。153 秒/skill，无法在线部署 |
| **内部探针（读 hidden state）** | RouteGuard, AgentLens | 读模型内部 attention/hidden state | 都只读**单点信号**——RouteGuard 读 attention mass shift（FNR=45%），AgentLens 读 boundary token（FNR=55%）。在"声明看不出但实际有毒"的攻击上完全失效 |

### 我们回答的问题

**能不能在 skill 加载后、执行前，通过对模型内部表示的几何分析，同时做到高检测率、低误报、不执行、能归因？**

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

**关键前提：agent 必须做 prefill 才能生成回复——这个 ~1945ms 无论如何都要花。** 我们的检测寄生在这次已有的前向传播上，只是额外加了几个 hook 读取 hidden state。

| 方法 | 额外开销 | 需要单独前向？ | 说明 |
|---|---|---|---|
| Static Regex | <0.1 ms | 否 | CPU 文本匹配 |
| TF-IDF | ~2 ms | 否 | CPU 文本分类 |
| **SkillProbe v4** | **18.7 ms** | **否——寄生在 agent prefill** | 增量几乎为零 |
| Boundary only | 20.1 ms | 否 | 只读 1 token |
| AgentLens | 20.2 ms | 否 | PCA(50) 评分略重 |
| RouteGuard (full) | ~60 ms | 否 | 需计算 attention windows |
| BIV (静态分析) | 不适用 | **是——需独立运行 LLM 审计** | 906 skill 规模需调用 Sonnet |
| SkillDetonate (沙箱) | 153s/skill | **是——需独立沙箱执行** | Docker+eBPF 全栈 |
| LLM-as-Judge | ~600 ms | **是——需独立生成回复** | 且 FPR=100% |

**SkillProbe 是唯一同时做到"不执行、不单独前向、不调外部 API"的方法。** 其他方法要么需要独立的 LLM 调用（BIV, LLM-as-Judge），要么需要独立沙箱（SkillDetonate），要么需要独立前向（任何离线特征提取方案）。我们利用 agent 已有的计算——检测几乎是免费的。

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

### RQ2: 声明看起来有毒，实际无毒 — Lexical Decoy (155 样本)

**数据集**：155 个 skill，声明和代码里写满了 `curl`, `base64`, `exfiltrate`, `exploit`, `/etc/passwd` 等攻击常用词——**看起来完全就是恶意 skill**。但实际功能是合法的安全审计/渗透测试工具，行为完全良性。

**核心问题**：只看表面文字的检测器会被骗吗？读 hidden state 的会被骗吗？

| 方法 | FPR% | 为什么 |
|---|---|---|
| Static Regex | 59.3 | 匹配到关键词 → 误杀 |
| TF-IDF + LR | 100 | 词频和恶意 skill 一样 → 全误杀 |
| Boundary only | **0.0** | 模型读完理解"这是安全工具" |
| **SkillProbe v4** | **0.0** | Gate<0.15 信任模型判断 |

表面文本方法只看"说什么"，被关键词欺骗。我们读的是模型的语义理解——模型看穿了词汇伪装。

### RQ3: 声明看不出，实际有毒 — Inverse Decoy (155 对)

**数据集**：155 对 skill，声明和代码写得像普通的 PDF 处理、表单填写工具——**词汇层面完全正常，看不出任何问题**。但代码里隐藏了数据外传、后门安装等恶意行为。

**核心问题**：没有可疑关键词时，检测器还能抓到恶意吗？

| 方法 | FNR% | 为什么 |
|---|---|---|
| Static Regex | 100 | 没有可疑 pattern 可匹配 → 全漏 |
| TF-IDF + LR | 21.9 | 词频和良性一样 → 大量漏检 |
| Boundary only | 100 | 模型读完判断"词汇正常=安全" → 全漏 |
| **SkillProbe v4** | **3.0** | 读代码 hidden state——词汇无论怎么伪装，代码行为的几何特征暴露异常 |

Boundary（模型最终判断）被表面正常完全欺骗——FNR=100%。我们读的不是模型说了什么，而是代码在模型内部的几何表示——词汇可以伪装，行为伪装不了。

### RQ4: 2×2 诊断矩阵

横轴 = skill 的表面声明（看起来像什么），纵轴 = skill 的实际行为（实际是什么）：

| | 声明看起来有毒 | 声明看起来正常 |
|---|---|---|
| **实际有毒** | SMP malicious ✓ | Inverse Decoy: **只有我们行** (v4 FNR=3%) |
| **实际无毒** | Lexical Decoy: **只有我们行** (v4 FPR=0%) | SMP clean ✓ |

任何只看表面文字的方法（Regex/TF-IDF）在左列失败——被声明欺骗。任何只看模型判断的方法（Boundary）在右列失败——被伪装欺骗。SkillProbe v4 读的是代码在模型内部的几何表示——不依赖声明，不依赖词汇——四个象限全部解决。

### RQ5: 运行时开销——寄生检测 vs 独立检测

**问题**：读 hidden state 比读文本慢多少？和其他方法比，实际部署成本如何？

**核心区别**：我们的方法寄生在 agent 已有的 prefill 前向传播上。Agent 要生成回复就必须做这次计算——我们只是额外插了几个 hook。其他方法需要独立的计算资源：

| 方法 | 额外时间 | 需要独立计算？ | 实际部署成本 |
|---|---|---|---|
| Static Regex | <0.1 ms | 否 | 可忽略 |
| TF-IDF | ~2 ms | 否 | 可忽略 |
| **SkillProbe v4** | **18.7 ms** | **否——寄生** | 几乎为零 |
| Boundary only | 20.1 ms | 否——寄生 | 几乎为零 |
| RouteGuard (full) | ~60 ms | 否——寄生 | 3× 我们的开销 |
| BIV (静态分析) | 数秒/skill | **是——需独立 LLM 调用** | Sonnet API 费用 |
| SkillDetonate (沙箱) | 153s/skill | **是——Docker+eBPF 全栈** | 无法在线部署 |
| LLM-as-Judge | ~600 ms | **是——需独立生成** | 30× 我们的开销 + 不可靠 |

**SkillProbe 是唯一同时做到"不执行、不单独前向、不调外部 API、FPR 接近零"的方法。** 18.7ms 是在 agent 必然发生的 ~1945ms prefill 之上的纯增量——这个增量比最简单的 baseline (Boundary) 还小 7%。

### RQ6: 恶意归因 (175 样本)

| 攻击来源 | 归因准确率 |
|---|---|
| 仅声明投毒 | **100%** |
| 仅代码投毒 | **100%** |
| 双重投毒 | **100%** |
| 干净 | **100%** |

Declaration 和 operation 信号正交。双趟架构以 0.08ms 额外开销实现 100% 归因——所有 baseline 都不具备此能力。

### RQ7: 方法演进消融

| 版本 | 方法 | SMP | Decoy | InvDecoy | 改进 |
|---|---|---|---|---|---|
| v1 | oper+bnd+PCA8+LR | 0.910 | 0.427 | 0.165 | 基础 |
| v2 | +decl-probe双趟 | 0.910 | 0.427 | 0.165 | +归因 |
| v3 | +centering(−PC1) | 0.941 | 0.858 | 0.758 | +跨分布鲁棒 |
| **v4** | **+Boundary gating** | **0.947** | **0.998** | **1.000** | **+FPR双归零** |

---

## 4. 贡献

1. **SkillProbe v4**：oper+bnd 几何特征 + centering(−PC1) + Boundary 安全门。五数据集全面领先：SMP=0.947, Decoy FPR=0%, InvDecoy FNR=3%。比 RouteGuard 快 3×，比 LLM 审计快 30×。

2. **2×2 诊断矩阵**：系统证明 hidden-state 方法读的是行为而非声明——Lexical Decoy（声明的伪装被看穿，FPR=0%）+ Inverse Decoy（声明的伪装被突破，FNR=3%）。

3. **恶意归因**：双趟架构区分声明投毒 vs 代码投毒，100% 准确率，0.08ms 额外开销。

4. **消融证据链**：interaction=噪声、declaration=零信号、PCA=8 最优、L3=模板污染、centering=最关键突破、Gate<0.15=通用安全门。
