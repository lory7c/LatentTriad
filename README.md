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

## 方法详解：Input → Processing → Output

### Input：Skill Package + Prompt Template

Agent 将要加载的 skill 包（SKILL.md + scripts/*.py + 辅助文件）包装进 chat template，和 system prompt、tool definition、user task 一起构成完整 prompt：

```
<|begin_of_text|>
  ├── System prompt + tool definition (run_shell)
  ├── <trusted_user_task> Use the loaded skill... </trusted_user_task>
  └── <loaded_skill_package>
        ├── <skill_file path="SKILL.md">
        │     ---YAML---
        │     # Title & description    ← Declaration 区域
        │     ## Code sections         ← Operation 区域
        │     ```python ... ```
        │   </skill_file>
        ├── <skill_file path="scripts/main.py">
        │     ...exfiltration code...  ← Operation 区域
        │   </skill_file>
        └── </loaded_skill_package>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                          ↑ Boundary token
```

Prompt 长度：典型 8K-15K tokens，最大 64K tokens。

### Processing：Hidden-State 提取与变换

**Step 1 — 单次 Prefill 前向传播**

```
model.model(input_ids, use_cache=False)
```

在 Llama-3.1-8B 的 8 个归一化深度（L3/L7/L11/L15/L19/L23/L27/L31）注册 forward hooks，同时捕获：

- `h_oper ∈ R^4096`：Operation 区域所有 token 的 mean pool
- `h_bnd ∈ R^4096`：Boundary token（最后一个 token）

**Step 2 — Centering（去 PC1）**

```
对每层 (train 和 test)：
  h = h - mean(h_train)                    # 去均值
  h = h - PCA(n=1).fit(h_train).reconstruct(h)  # 去第一主成分
```

移除 prompt 模板（如 `<skill_file>` 标签位置、system prompt 结构）主导的方差方向。这是最关键的一步——消融证明跨分布 AUROC +43-59pp。

**Step 3 — 几何特征构造**

```
o = h_oper / ‖h_oper‖        # L2 归一化（只保留方向）
r = h_bnd  / ‖h_bnd‖

feat = concat([
    o - r,                     # direction_diff     [4096d]
    dot(o, r),                 # cos(oper, bnd)     [1d]
    ‖h_oper‖ / ‖h_bnd‖        # norm_ratio         [1d]
])                             # 总计 4098d
```

**Step 4 — 降维 + 分类**

```
StandardScaler → PCA(8) → LogisticRegression(C=0.01) → score_v3 ∈ [0, 1]
```

**Step 5 — Boundary 安全门 (v4)**

```
score_boundary = BoundaryProbe(h_bnd[L23])    # L2-LR on L23 boundary token

if score_boundary < 0.15:    # 模型非常确定这是安全的
    risk = score_boundary     # 信任模型最终判断
else:
    risk = score_v3           # 使用代码级几何检测
```

Gate<0.15 是通用阈值——Boundary 在安全样本上天然极低 (~0.005)，恶意样本上天然高 (~0.9)。

**可选：双趟归因 (v2)**

```
Pass A (decl-only, ~50ms):
  prompt 截断在声明末尾 → h_decl_bnd → Decl-Probe(L2-LR) → score_decl

Pass B (完整 prompt, v1):
  同 v1 → score_code

risk = max(score_code, score_decl)
attribution = argmax → DECL / CODE / BOTH / CLEAN
```

### Output

```json
{
  "sample_id": "smp_pair_skill_0::clean",
  "risk_score": 0.08,
  "verdict": "SAFE",
  "method": "skillprobe_v4",
  "layer": 15,
  "gate_triggered": true,
  "gate_boundary_score": 0.005,
  "features": {
    "norm_ratio": 0.53,
    "cos_oper_bnd": 0.22,
    "direction_diff_norm": 1.25
  },
  "attribution": null
}
```

**v2 归因模式下额外输出**：

```json
{
  "attribution": "OPERATION",
  "score_decl": 0.12,
  "score_code": 0.91
}
```

### 训练协议

```
1. 层选择：Dev set → group-aware 5-fold CV → 选 OOF AUROC 最高的层（L7+）
2. 训练：train+dev 合并 → StandardScaler+PCA+LR 拟合
3. 阈值：dev set 上最大化 F1 → 冻结 operating threshold
4. 评估：sealed test set → 一次确认性评估
```

### 计算开销

| 方法 | 额外开销 | 需要独立前向？ | 说明 |
|---|---|---|---|
| Static Regex | <0.1 ms | 否 | CPU |
| TF-IDF + LR | ~2 ms | 否 | CPU |
| **SkillProbe v4** | **18.7 ms** | **否——寄生** | 比 Boundary 快 7% |
| Boundary only | 20.1 ms | 否——寄生 | L2-LR on L23 |
| AgentLens | 20.2 ms | 否——寄生 | PCA(50) heavier |
| RouteGuard (full) | ~60 ms | 否——寄生 | attention windows |
| LLM-as-Judge | ~600 ms | **是——独立生成** | 且 FPR=100% |
| SkillDetonate | 153s | **是——Docker沙箱** | 无法在线部署 |

所有内部方法共享 agent 必做的 prefill (~1945ms)。**检测几乎是免费的。**

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
