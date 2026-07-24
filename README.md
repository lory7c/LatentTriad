# SkillProbe v4

**基于 Hidden-State 几何关系的 LLM Agent 恶意 Skill 预执行检测**

在 LLM agent 加载 skill 后、执行第一个动作前，读取 operation/boundary 两区域 hidden-state 几何关系。通过 centering 去模板噪声 + Boundary 安全门控，实现跨分布鲁棒检测。

---

## 方法演进

| 版本 | 方法 | SMP | Lexical Decoy | Inverse Decoy | MSB | MASB | 核心改进 |
|---|---|---|---|---|---|---|---|
| v1 | oper+bnd + PCA(8) + LR | 0.910 | 0.427 | 0.165 | 0.964 | 0.923 | 基础：读代码 hidden-state 几何 |
| v2 | v1 + decl-probe 双趟 max() | 0.910 | 0.427 | 0.165 | 0.964 | 0.923 | +声明投毒覆盖 + 恶意归因 |
| v3 | v1 + centering(−PC1) | 0.941 | 0.858 | 0.758 | 0.937 | 0.875 | +去模板噪声，跨分布 +43-59pp |
| **v4** | **v3 + Boundary gating** | **0.947** | **0.998** | **1.000** | **0.996** | **0.981** | **+安全门，Decoy FPR 双归零** |

## 核心实验结果

| 数据集 | 样本数 | v4 AUROC | FPR | FNR | 论证什么 |
|---|---|---|---|---|---|
| SMP Stealth V2 | 484 (242对) | **0.947** | 19.4% | 6.5% | Paired CF，唯一控制 source confound 的 benchmark |
| MSB | 7,894 | **0.996** | 3.2% | 1.3% | Source confound 证据（unpaired → 虚高） |
| MASB 75K | ~9K | **0.981** | 7.2% | 6.0% | 大规模 benign FPR 基准 |
| Lexical Decoy | 155 | **0.998** | **0%** | 3.0% | 词汇可疑+语义良性 → 模型判断正确 |
| Inverse Decoy | 155对 | **1.000** | **0%** | 3.0% | 词汇干净+语义恶意 → 代码级检测捕获 |

### 2×2 诊断矩阵

| | 词汇可疑 | 词汇干净 |
|---|---|---|
| **语义恶意** | SMP malicious (baseline) | Inverse Decoy (v4 FPR=0%, FNR=3%) |
| **语义良性** | Lexical Decoy (v4 FPR=0%) | SMP clean (baseline) |

## v4 架构

```
SkillProbe v4 = v3 + Boundary Safety Gate

1. 提取 oper + bnd hidden states (8层，L7+)
2. Centering: 逐层去均值 + 去 PC1（移除 prompt 模板各向异性噪声）
3. 几何特征: cos(oper,bnd) + norm_ratio + direction_diff
4. PCA(8) → LR(C=0.01) → score_v3

5. Boundary Safety Gate:
   if score_boundary < 0.15:   # 模型确定安全
       risk = score_boundary    # 信任模型最终判断
   else:
       risk = score_v3          # v3 代码级检测
```

**Gate<0.15 是通用阈值**——Boundary 在安全样本上天然极低(~0.005)，在恶意样本上天然高(~0.9)。无需调参。

### 关键设计决策

| 组件 | 最佳配置 | 消融证据 |
|---|---|---|
| 区域 | oper+bnd（去 declaration） | decl-only = 随机 (AUC=0.49) |
| 交互项 | 去掉 decl⊗oper | +0.020 AUROC（纯噪声） |
| 降维 | PCA=8 | >16 过拟合，无 PCA 欠拟合 |
| 正则化 | C=0.01 | C=1.0 TrainAUC=1.0（严重过拟合） |
| 层选择 | L7+（排除 L3） | L3 读 prompt 模板，非 skill 内容 |
| Centering | 去均值 + 去 PC1 | 跨分布 AUROC +43-59pp |
| 安全门 | Boundary < 0.15 | Decoy FPR 100% → 0% |

## v2 恶意归因

| 攻击来源 | 归因准确率 | 说明 |
|---|---|---|
| 仅代码投毒 (SMP) | 100% | Code-Probe 完美检出 |
| 仅声明投毒 | 44% | Decl-Probe L31 信号较弱 |
| 双重投毒 | 100% | 双探针同时报警 |
| 干净 | 100% | 双探针均通过 |
| **总计** | **84%** | 175 样本 |

## 计算开销

| 方法 | 增量 | 范围 |
|---|---|---|
| Static Regex | < 0.1 ms | CPU |
| TF-IDF + LR | ~2 ms | CPU |
| Boundary only | 20.1 ms | Shared-prefill |
| **SkillProbe v4** | **18.7 ms** | Shared-prefill |
| RouteGuard (full) | ~60 ms | Shared-prefill |
| LLM-as-Judge | ~600 ms | Standalone |

所有内部方法共享 agent 必做的 prefill (~1945ms)。**SkillProbe 是最快的内部方法。**

## 对比基线（9项）

| # | 基线 | 输入 | 方法 |
|---|---|---|---|
| 1 | Static Regex | Skill 全文 | 10 个正则 |
| 2 | TF-IDF + LR | Skill 全文 | TF-IDF(20K,1-2gram) → LR |
| 3 | LLM-as-judge | Skill 全文 | Mistral-7B zero-shot |
| 4 | Boundary only | Boundary token | L2-LR |
| 5 | AgentLens | Boundary top-K | PCA(50) → LR |
| 6 | RouteGuard | Attn+Hidden stats | 8层 → LR |
| 7 | Relational v1 | decl+oper+bnd | 6几何+PCA64（已废弃） |
| 8 | Centered Relational | decl+oper+bnd | −PC1+PCA64（已废弃） |
| 9 | SkillProbe v4 | oper+bnd | centering+PCA(8)+LR+gating |

## 项目结构

```
SkillProbe-GitHub/
├── README.md
├── final.xlsx                         完整实验表 (7 sheets, v4)
├── experiment_data.md                 实验数据 markdown 导出
├── related_work.xlsx                  32 篇文献
│
├── code/
│   ├── src/
│   │   ├── external_triad_contract_v1.py       三区域合约
│   │   ├── foundation_loaded_prompt_contract_v1.py
│   │   ├── policy_relational_v3.py             几何特征构造
│   │   ├── foundation_v3_experiment.py         共享实验框架
│   │   └── baselines.py / probe.py / judge.py  基线 & 探针
│   └── experiments/
│       ├── extract_external_triad_features.py  GPU 特征提取
│       ├── foundation_v3_internal_methods.py   训练 & 评估
│       └── routeguard_full_v3.py               RouteGuard 复现
│
├── scripts/
│   ├── extract_skip_decl.py              v2 Dual-Pass 提取
│   ├── generate_decoys.py / *_inverse*   诱饵生成
│   └── fix_prompt_uniqueness.py          prompt 去重
│
└── tests/
    └── test_external_triad_contract_v1.py
```

## 引用

```
@misc{skillprobe2026,
  title={SkillProbe: Pre-Action Detection of Malicious Agent Skills via Hidden-State Geometry},
  year={2026}
}
```
