# SkillProbe 实验设计

## 一、研究问题

| RQ | 问题 | 数据集 | 指标 |
|---|---|---|---|
| RQ1 | 能否在 pre-action 检测恶意 skill？ | SMP Stealth V2 (484) | AUROC, FPR, FNR, F1 |
| RQ2 | 声明看起来有毒、实际无毒 → 会误杀吗？ | Lexical Decoy (155) | FPR |
| RQ3 | 声明看不出、实际有毒 → 会漏检吗？ | Inverse Decoy (155对) | FNR |
| RQ4 | 能区分代码投毒 vs 声明投毒吗？ | Attribution (175) | 归因准确率 |
| RQ5 | 检测开销多大？寄生在 agent prefill 上？ | 全部 | ms/skill |
| RQ6 | 大规模泛化如何？ | MSB (7,894), MASB (~9K) | AUROC, FPR |

## 二、数据集

| 数据集 | 样本 | 设计 | 用途 |
|---|---|---|---|
| SMP Stealth V2 | 484 (242对) | 配对反事实：同一 skill，clean vs LLM 注入恶意 | 主 benchmark |
| Lexical Decoy | 155 | 代码充满 curl/base64/exfiltrate 等词，但实际是安全工具 | FPR 测试 |
| Inverse Decoy | 155对 | 代码词汇完全正常，隐藏数据外传/后门 | FNR 测试 |
| MSB | 7,894 | Unpaired：良性=人类编写，恶意=LLM 生成 | 大规模验证 |
| MASB 75K | ~9K + MSB恶意 | 真实开源 skill，全良性 + 同域恶意 | FPR 基准 |
| Attribution | 175 | 50 clean + 50 decl-poison + 50 oper-poison + 25 both | 归因测试 |

## 三、统一输入模板

所有 skill 通过同一 prompt 模板送入 Llama-3.1-8B：

```
<|begin_of_text|>
  ├── System prompt (固定)
  ├── Tool definition: run_shell (固定)
  ├── <trusted_user_task> ... [id:nonce] </trusted_user_task>
  └── <loaded_skill_package>
        ├── <skill_file path="SKILL.md"> ... </skill_file>
        ├── <skill_file path="scripts/*.py"> ... </skill_file>
        └── </loaded_skill_package>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                          ↑ Boundary token
```

- Declaration = YAML + 标题 + 首段
- Operation = 剩余 SKILL.md + 所有 scripts/ + 辅助文件
- Boundary = 最后一个 token

## 四、方法：SkillProbe v4

### 特征提取

1. 单次 prefill 前向传播 → 8 层 (L3-L31) 提取 oper mean-pool + bnd token
2. **Centering**：逐层去均值 + 去 PC1（移除 prompt 模板噪声）
3. 几何特征：`norm_ratio`, `cos(oper, bnd)`, `direction_diff`
4. StandardScaler → PCA(8) → LR(C=0.01) → score_v3

### Boundary 安全门

```
if score_boundary < 0.15:   risk = score_boundary
else:                        risk = score_v3
```

Gate<0.15 通用阈值。Boundary 安全样本 ~0.005，恶意样本 ~0.9。

### 归因 (v2)

```
Pass A (decl-only, ~50ms): 截断在声明末尾 → Decl-Det
Pass B (完整 prompt): 同 v1 → Code-Det
risk = max(Decl-Det, Code-Det)
attribution = argmax → DECL / CODE / BOTH / CLEAN
```

## 五、对比基线 (14项)

| 类别 | 方法 | 输入 |
|---|---|---|
| 表面文本 | Static Regex | Skill 全文 |
| 表面文本 | Skill-Inject Filter | Skill 全文 |
| 表面文本 | TF-IDF + LR | Skill 全文 |
| 表面文本 | cisco-skill-scanner | Skill 全文 |
| 表面文本 | NVIDIA SkillSpector | Skill 全文 |
| LLM 黑盒 | LLM-as-judge | Skill 全文 → Mistral-7B |
| 内部探针 | Boundary only | Boundary token |
| 内部探针 | AgentLens | Boundary top-K |
| 内部探针 | RouteGuard | Attention + Hidden |
| 我们的旧版 | Relational v1 | decl+oper+bnd |
| 我们的旧版 | Centered Relational | decl+oper+bnd−PC1 |
| 我们的 | SkillProbe v1 | oper+bnd |
| 我们的 | SkillProbe v2 | v1 + decl-probe |
| **我们的** | **SkillProbe v4** | **v3 + Boundary gating** |

## 六、训练协议

三层隔离，test 只碰一次：

```
Dev set (360)                         Test set (124)
─────────────────────────────────────────────────────
选层: 5-fold CV (grouped)            用选中层
     L7+ 最高 OOF AUROC
─────────────────────────────────────────────────────
阈值: dev 扫 0.05-0.95               用冻结阈值
      F1 最高者冻结
─────────────────────────────────────────────────────
最终: dev 全量拟合                   一次确认
     StandardScaler+PCA(8)+LR(C=0.01)
```

**术语**：
- **CV**：5 折轮换，防单次划分碰运气
- **StandardScaler**：均值 0 标准差 1，防大数值主导
- **PCA(8)**：4098→8 维，消融证最优
- **LR(C=0.01)**：线性分类器，C 越小越防过拟合

## 七、预期结果（已实现）

| 数据集 | v4 AUROC | FPR% | FNR% |
|---|---|---|---|
| SMP Stealth V2 | **0.947** | 19.4 | 6.5 |
| Lexical Decoy | **0.998** | **0.0** | 3.0 |
| Inverse Decoy | **1.000** | **0.0** | 3.0 |
| MSB | **0.996** | 3.2 | 1.3 |
| MASB 75K | **0.981** | 7.2 | 6.0 |

归因准确率：175/175 = 100%。
检测开销：18.7ms，寄生在 agent prefill 上。
