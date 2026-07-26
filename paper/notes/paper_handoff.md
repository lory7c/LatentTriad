# SkillProbe 论文撰写交接文档

> 给撰稿专家：本文档说明项目全貌、核心叙事、数据来源、及文件对应关系。

---

## 一、项目一览

**SkillProbe v4** — 基于 LLM hidden-state 几何关系的 agent skill 恶意检测方法。在 agent 加载第三方 skill 后、执行前，通过一次 prefill 前向传播中 operation 和 boundary token 的几何特征（norm_ratio, cos, direction_diff），经 centering(−PC1) 去模板噪声 + Boundary 安全门控，实现高召回、低误报、可归因、寄生部署（18.7ms，比 Boundary 还快 7%）。

**核心数字**：SMP AUROC=0.947（SOTA），Lexical Decoy FPR=0%，Inverse Decoy FNR=3%，归因准确率 100%。

## 二、文件导航

### 叙事与实验设计

| 文件 | 用途 |
|---|---|
| `NARRATIVE_CN.md` | **中文叙事线（最新）**：动机→机制发现→方法→RQ1-RQ7→贡献。撰稿时以此为准 |
| `EXPERIMENT_DESIGN.md` | 实验设计：6 RQ, 6 数据集, 统一输入, 训练协议 |
| `GAP_REPORT.md` | 声明投毒盲区发现 + Dual-Pass 归因原始报告 |

### 实验数据

| 文件 | 用途 |
|---|---|
| `final.xlsx` | **主数据表（9 sheets）**：5 基准数据集 + 方法演进 + 恶意归因 + SkillHarm 外部验证 |
| `experiment_data.md` | xlsx 的 markdown 导出，方便快速浏览 |

### 方法代码

| 文件 | 用途 |
|---|---|
| `code/src/skillprobe_v4.py` | **v4 正式实现**：`SkillProbeV4` 类，含 centering、geometric_features、Boundary gate |
| `code/src/policy_relational_v3.py` | 几何特征构造（v1 基础） |
| `code/src/external_triad_contract_v1.py` | 统一 prompt 模板（所有数据集同一格式） |
| `code/src/foundation_loaded_prompt_contract_v1.py` | SMP 内部 contract 定义 |
| `code/experiments/extract_external_triad_features.py` | GPU 特征提取管线 |
| `code/experiments/foundation_v3_internal_methods.py` | 训练 & 评估管线 |

### 基线复现

| 文件 | 用途 |
|---|---|
| `code/experiments/routeguard_full_v3.py` | RouteGuard 论文复现（attention+hidden 双专家） |
| `code/experiments/foundation_v3_static_baselines.py` | TF-IDF, Static Regex 基线 |
| `code/experiments/agent_native_llm_judge_v4.py` | LLM-as-Judge (Mistral-7B) |

### 数据生成

| 文件 | 用途 |
|---|---|
| `scripts/generate_decoys.py` | Lexical Decoy 生成（声明有毒/实际无毒） |
| `scripts/generate_inverse_decoys_v2.py` | Inverse Decoy 生成（声明看不出/实际有毒） |
| `scripts/extract_skip_decl.py` | Dual-Pass 特征提取（Pass A decl-only + Pass B skip-decl） |
| `scripts/fix_prompt_uniqueness.py` | Prompt 去重（防止 label leakage） |

### 论文目录

| 文件 | 用途 |
|---|---|
| `paper/output/doc/main.tex` | **活跃手稿**（USENIX Security 2027 模板） |
| `paper/output/doc/usenix.sty` | USENIX 格式文件 |
| `paper/notes/paper_outline.md` | **全文大纲**（9 节 + 8 图 + 8 表） |
| `paper/notes/project_truth.md` | 项目真值（术语、约束） |
| `paper/notes/result_summary.md` | 结果锚点（哪些是强结论、哪些需限定） |
| `paper/notes/paper_handoff.md` | **本文件** |

---

## 三、核心叙事线（撰稿必须遵循）

### 机制优先结构（对标 RouteGuard 论文）

```
Attack exists → Existing defenses fail → We found a mechanism → We built a detector
```

**§1 Introduction**: Agent skill 供应链攻击（ClawHavoc, BIV 80%偏差）→ 三类防御缺陷 → **我们发现"Pre-Action Vigilance"**（恶意 skill 使 boundary token 异常激活，norm_ratio SMD=−1.42）→ SkillProbe v4 概述

**§3 Mechanism（全文最重要的一节）**: 在讲方法之前，先讲发现——norm_ratio 逐层衰减、和 RouteGuard attention hijacking 互补、cos/direction_diff 单独不显著。**这一节的论点决定了整篇论文的 credibility**。

### 关键术语（不可替换）

- **"Pre-Action Vigilance"** — 机制名称
- **"centering(−PC1)"** — 去 PC1 操作
- **"Boundary safety gate"** — Gate<0.15
- **"parasitic detection"** — 寄生在 agent prefill 上
- **"声明看起来有毒 / 实际无毒"** — Lexical Decoy
- **"声明看不出 / 实际有毒"** — Inverse Decoy

### 和 RouteGuard 的关系

RouteGuard 是 notre 最直接竞争者。我们不是 "第一个读 hidden state"——RouteGuard 也读。我们的差异化：
1. **读什么不同**：他们读 attention shift + hidden alignment，我们读 oper-bnd 几何关系
2. **centering(−PC1)**：他们没做，跨分布 +43-59pp
3. **归因**：他们不能区分声明 vs 代码投毒
4. **寄生部署**：他们需要 attention 计算（60ms），我们只需 hook（18.7ms）
5. **2×2 诊断矩阵**：他们只有单一分数

### 不自称 "第一个"，但强调 "做得最好" 的维度

- 配对反事实 benchmark（SMP）消除 source confound
- centering 跨分布鲁棒（SkillHarm 100% recall vs Boundary 0%）
- 归因能力（所有 baseline 不具备）
- 寄生部署（最快内部方法）

---

## 四、数据使用指南

### 六个数据集

| 数据集 | 样本 | 用途 | 在 xlsx 的 sheet |
|---|---|---|---|
| SMP Stealth V2 | 484 (242对) | **主 benchmark** | `SMP Stealth V2` |
| MSB | 7,894 | 外部验证 / confound 证据 | `MSB` |
| MASB 75K | ~9K | 大规模 FPR 基准 | `MASB 75K` |
| Lexical Decoy | 155 | 声明有毒/实际无毒 → FPR 测试 | `Lexical Decoy` |
| Inverse Decoy | 155对 | 声明看不出/实际有毒 → FNR 测试 | `Inverse Decoy` |
| SkillHarm | 931 | 外部第三方验证 | `SkillHarm` |

### 14 个基线

xlsx 每张 sheet 按统一排序：表面文本(1-5) → LLM 黑盒(6) → 内部探针(7-9) → 旧版(10-11) → 我们的(12-14)。绿色行 = 我们的方法迭代。

### 关键数字速查

- SMP v4 AUROC=**0.947**, FPR=19.4%, FNR=**6.5%**
- Lexical Decoy v4 FPR=**0%** (Boundary 0%, TF-IDF 100%)
- Inverse Decoy v4 FNR=**3%** (Boundary 100%)
- 归因准确率 **100%** (175/175)
- 开销 **18.7ms**（Boundary 20.1ms, RouteGuard ~60ms, LLM-as-Judge ~600ms）
- centering 消融：跨分布 **+43-59pp**
- SkillHarm v3 AUC=**0.973** Recall=**100%**（Boundary AUC=0.0）

---

## 五、撰写优先级

1. **§3 Mechanism**（最核心——没有机制发现，方法就没有根基）
2. **§6.1 SMP 主结果**（主要 benchmark）
3. **§6.2-6.3 2×2 + 归因**（差异化优势）
4. **§1 Introduction**（最后写——写完所有才写开头）
5. **§4 Method**（和 §3 一起写）
6. **§6.4-6.7 其余 RQ**（开销、消融、外部）
7. **§2, §5, §7-9**（背景、setup、讨论、结论）

## 六、待补充

- [ ] 所有 figure（目前只有 xlsx 表格数据，需画图）
- [ ] 和 BIV 的定量对比（目前只有 narrative 描述）
- [ ] SkillDetonate 的定量对比
- [ ] 统计检验（bootstrap CI, DeLong test）——部分数据在 FINAL_0709_TABLE.json
- [ ] 案例研究（具体恶意 skill 的归因示例）
- [ ] MASB RouteGuard L7+ 重跑（需 MSB route_hidden 重提取）
