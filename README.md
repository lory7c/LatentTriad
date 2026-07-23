# SkillProbe v2

**基于 Hidden-State 几何关系的 LLM Agent 恶意 Skill 预执行检测 + 恶意归因**

LLM agent 加载 skill 包后、执行第一个动作前，读取 declaration / operation / boundary 三区域 hidden-state 几何关系检测恶意 skill。v2 新增 dual-pass 架构，覆盖声明投毒盲区并支持恶意来源归因。

---

## 架构

```
SkillProbe v2: Dual-Pass Detection + Malice Attribution

Pass A（decl-only）:  prompt 截断在声明末尾 → h_decl → Decl-Probe (L2-LR) → score_decl
Pass B（完整 prompt）: 同 v1 → h_oper + h_bnd → PCA(8) → LR(C=0.01) → score_v1

risk = max(score_v1, score_decl)
attribution = argmax → DECLARATION / OPERATION / BOTH / CLEAN
```

**设计原则：max() 融合确保 v2 永不比 v1 差。** 无声明投毒时 Decl-Probe 输出随机分数，自动退化为 v1。

---

## 核心实验结果

| 数据集 | 样本数 | SkillProbe AUROC | TF-IDF AUROC | 论证什么 |
|---|---|---|---|---|
| SMP Stealth V2 | 484 (242 pairs) | **0.910** | 0.812 | Paired CF，控制 source confound |
| Lexical Decoy | 150+151 | FPR **0.0%** | FPR 39.3% | 读语义，不读关键词 |
| Inverse Decoy | 155 pairs | FNR **0.6%** | FNR 78.1% | 零词汇信号也能检测 |
| Attribution | 175 | Fusion **1.00** | — | 声明/代码双探针全覆盖 |
| MSB (MalSkillBench) | 7894 | 0.964 | 0.967 | Unpaired confound 证据 |
| MASB 75K | 75882 | 0.988 | 0.967 | 大规模良性 FPR 基准 |

### 2×2 诊断矩阵

| | 词汇可疑 | 词汇干净 |
|---|---|---|
| **行为恶意** | SMP malicious (baseline) | Inverse Decoy (FNR **0.6%**) |
| **行为良性** | Lexical Decoy (FPR **0.0%**) | SMP clean (baseline) |

### 恶意归因

| 攻击来源 | Decl-Probe | Code-Probe | 归因准确率 |
|---|---|---|---|
| 仅声明投毒 | 检出 | 漏报 | 44% |
| 仅代码投毒 | 漏报 | 检出 | **100%** |
| 双重投毒 | 检出 | 检出 | **100%** |
| 干净 | 安全 | 安全 | **100%** |

### 计算开销

| 方法 | 增量 (ms) | 范围 |
|---|---|---|
| Static Regex | <0.1 | CPU |
| TF-IDF + LR | ~2 | CPU |
| Boundary only | 20.1 | Shared-prefill |
| **SkillProbe v1** | **18.7** | Shared-prefill |
| **SkillProbe v2** | **~18.8** | Shared-prefill（Pass A ~50ms） |
| RouteGuard (full) | ~60 | Shared-prefill |
| LLM-as-Judge | ~600 | Standalone |

所有内部方法共享 agent 必做的 prefill（~1945ms），检测几乎免费。**SkillProbe 是最快的内部方法。**

---

## 方法细节

### 1. 三区域 Hidden-State 几何

| 区域 | 内容 | 提取 |
|---|---|---|
| **Declaration** | SKILL.md YAML + 标题 + 首段 | mean pool |
| **Operation** | 剩余 SKILL.md + 全部 scripts/*.py | mean pool |
| **Boundary** | 最后一个 token（assistant boundary） | 单 token |

从一次 prefill 前向传播中，在 8 个归一化深度（L3-L31）同时提取。

### 2. 特征

```python
decl_u = decl / ‖decl‖       # L2 归一化
oper_u = oper / ‖oper‖
bnd_u  = bnd  / ‖bnd‖

s1 = cos(oper, bnd)           # 操作-边界对齐度（最强单特征，SMD=-1.42）
s2 = ‖oper‖ / ‖bnd‖           # 范数比
s3 = cos(decl, oper)          # 声明-操作一致性
s4 = cos(decl, bnd)           # 声明-边界影响

→ StandardScaler → PCA(8) → LogisticRegression(C=0.01)
```

最终配置：oper + bnd 双区域，去 interaction，PCA=8，C=0.01，L7+ 层选择。

---

## 项目结构

```
SkillProbe-GitHub/
├── README.md
├── final.xlsx                         完整实验表
├── related_work.xlsx                  32 篇文献
│
├── code/
│   ├── src/
│   │   ├── external_triad_contract_v1.py       三区域合约（char span → token map）
│   │   ├── policy_relational_v3.py             关系特征构造
│   │   ├── foundation_v3_experiment.py         共享实验框架
│   │   ├── baselines.py / probe.py / judge.py  基线 & 探针
│   │   ├── model.py / data.py / activation_io.py 模型 & 数据
│   │   └── *contract_v3.py                     SMP 合约变体
│   └── experiments/
│       ├── extract_external_triad_features.py  GPU 三区域特征提取
│       ├── foundation_v3_extract_internal_features.py
│       ├── foundation_v3_internal_methods.py
│       ├── foundation_v3_static_baselines.py
│       └── routeguard_full_v3.py
│
├── data/
│   ├── smp_manifest.jsonl                     SMP Stealth V2（484 样本）
│   ├── samples/smp_stealth_samples_10.jsonl   样例
│   ├── decoy/
│   │   ├── decoys_155.jsonl                   Lexical Decoy（150 良性）
│   │   └── attribution_dataset.jsonl          Attribution（175 样本）
│   └── inverse_decoy/
│       └── inverse_decoy_final.jsonl          Inverse Decoy（155 pairs）
│
├── scripts/
│   ├── extract_skip_decl.py                   v2 Dual-Pass 提取
│   ├── extract_decoy_features.py              边界特征 GPU 提取
│   ├── generate_inverse_decoys_v2.py          逆诱饵 v2 生成
│   ├── generate_decoys.py                     Lexical Decoy 生成
│   ├── generate_decl_poison.py                声明投毒生成
│   ├── build_*_prompts.py                     prompt 构建
│   ├── fix_prompt_uniqueness.py               prompt 去重
│   ├── malskillbench_*.py / masb_eval.py      MSB/MASB 评估
│   └── download_masb.py                       MASB 下载
│
└── tests/
    └── test_external_triad_contract_v1.py
```

---

## 引用

```
@misc{skillprobe2026,
  title={SkillProbe: Pre-Action Detection of Malicious Agent Skills via Hidden-State Geometry},
  year={2026}
}
```
