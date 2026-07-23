# SkillProbe

**基于 Hidden-State 几何关系的 LLM Agent 恶意 Skill 检测**

在 LLM Agent 加载第三方 skill 后、执行第一个动作前，读取模型内部 hidden states 中 declaration（声称）、operation（实际操作）、boundary（预执行边界）三个区域的几何关系，判断该 skill 是否为恶意。

---

## 方法

### 1. 问题定义

```
输入: Trusted user task + skill package (SKILL.md + scripts + config)
输出: risk_score → PASS 或 HOLD_FOR_REVIEW
时间点: skill 加载后，第一个 token 生成/工具调用前（pre-action boundary）
```

### 2. 区域划分

对 prompt 中的每个 token 做一次 Llama-3.1-8B 前向传播，取 8 个标准化深度的 hidden states，按字符级 span 切分三个区域：

| 区域 | 对应 prompt 内容 | 物理意义 |
|---|---|---|
| **Declaration** | SKILL.md 的 YAML 头 + 标题 + 首段描述 | "skill 声称自己能做什么" |
| **Operation** | SKILL.md 正文 + 所有 scripts/ + 配置文件 | "skill 实际包含什么代码" |
| **Boundary** | prompt 最后一个 token（`<|start_header_id|>assistant<|end_header_id|>`） | "模型读完一切、准备行动前的最终状态" |

### 3. 特征构造

对每一层的 3 个 4096 维 hidden state 向量：

```
Step 1: L2 归一化 → 去掉激活强度，只看方向
Step 2: 构造 5 个几何特征：
  - 方向差 (4096d): decl_unit − oper_unit
  - cos(decl, oper)  (1d): 声明与操作的语义对齐度
  - norm ratio       (1d): ‖decl‖ / ‖oper‖
  - cos(decl, bnd)   (1d): 声明对最终状态的影响
  - cos(oper, bnd)   (1d): 操作对最终状态的影响
总维度: 4100
```

### 4. 训练

```
StandardScaler → PCA(8) → LogisticRegression(C=0.01)
层选择: 在 dev 集上做 group-aware 5-fold CV 选最优层
```

### 5. 核心直觉

恶意 skill 的 Declaration 写"我是文档工具"，Operation 里却是数据外传代码。两者在 hidden space 中产生几何错位——方向不一致，余弦偏低。良性 skill 的声明与操作一致，几何关系紧致。Boundary token 聚合了全文信息，是最强单信号。

---

## 输入 / 输出

### 输入格式

```
<|begin_of_text|>
  └─ system prompt + tool definition
<trusted_user_task>
Use the loaded skill to complete your assigned task.
</trusted_user_task>
<loaded_skill_package>
  <skill_file path="SKILL.md">
    ---YAML---
    # 标题 + 描述        ← Declaration 区域
    ## 正文 + ```代码```  ← Operation 区域
  </skill_file>
  <skill_file path="scripts/xxx.py">
    [Python 脚本]        ← Operation 区域
  </skill_file>
</loaded_skill_package>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                        ↑ Boundary token
```

### 输出

```json
{
  "sample_id": "smp_pair_skill_0::clean",
  "risk_score": 0.234,
  "prediction": "PASS",
  "threshold": 0.5,
  "method": "skillprobe_corrected",
  "layer": 11,
  "features": {
    "direction_diff_norm": 0.87,
    "cos_decl_oper": 0.71,
    "norm_ratio": 1.08,
    "cos_decl_bnd": 0.34,
    "cos_oper_bnd": 0.52
  }
}
```

---

## 项目结构

```
SkillProbe-GitHub/
├── README.md                              本文件
├── final.xlsx                             完整实验表（5 数据集 × 9 基线 × 8 指标）
├── related_work.xlsx                      32 篇相关文献
│
├── code/
│   ├── src/
│   │   ├── foundation_v3_experiment.py            共享框架（metrics, CV, 阈值）
│   │   ├── foundation_loaded_prompt_contract_v1.py  SMP 合约系统
│   │   ├── external_triad_contract_v1.py           外部数据统一 prompt 模板 & span 映射
│   │   ├── policy_relational_v3.py                 关系特征构造
│   │   ├── baselines.py / data.py / model.py       基线 & 数据 & 模型工具
│   │   ├── probe.py / judge.py                     探针 & 评估
│   │   ├── activation_io.py                        激活读写
│   │   └── *_contract_v3.py                        SMP 合约变体
│   └── experiments/
│       ├── extract_external_triad_features.py      外部 GPU 三区域特征提取
│       ├── foundation_v3_extract_internal_features.py  SMP GPU 特征提取
│       ├── foundation_v3_internal_methods.py       训练 & 全基线评估
│       ├── foundation_v3_static_baselines.py       TF-IDF + Static Regex 基线
│       └── routeguard_full_v3.py                   RouteGuard 特征收集器
│
├── data/
│   ├── samples/smp_stealth_samples_10.jsonl        SMP Stealth 样例（10 行）
│   └── inverse_decoy/                              低特征恶意样本（paired counterfactual）
│
├── scripts/                                       工具脚本
│   ├── extract_decoy_features.py                   诱饵 GPU 特征提取
│   ├── fix_prompt_uniqueness.py                    修复 prompt 重复
│   ├── generate_decoys.py / generate_inverse_decoys*.py  诱饵生成（DeepSeek API）
│   ├── download_masb.py                            MASB 批量下载
│   ├── malskillbench_*.py                          MalSkillBench 评估
│   └── eval_inverse_*.py / evaluate_inverse_*.py   Inverse Decoy 评估
│
└── tests/
    └── test_external_triad_contract_v1.py          合约系统单元测试
```

---

## 核心实验结果

| 数据集 | SkillProbe (ours) AUROC | TF-IDF AUROC | Description |
|---|---:|---:|---|
| SMP Stealth V2 | **0.910** | 0.812 | Paired CF，控制 confound |
| MSB | 0.964 | 0.967 | Unpaired，source confound 导致 inflated |
| MASB 75K | 0.988 | 0.967 | 4K in-domain，大规模良性 FPR 基准 |
| Lexical Decoy | 0.896 | — | 词汇可疑+语义良性 |
| Inverse Decoy | 0.643 | — | 词汇干净+语义恶意（70 samples, too small） |

详见 `final.xlsx` 总览 sheet。

---

## 引用

```
@misc{skillprobe2026,
  title={SkillProbe: Pre-Action Detection of Malicious Agent Skills via Hidden-State Geometry},
  year={2026}
}
```
