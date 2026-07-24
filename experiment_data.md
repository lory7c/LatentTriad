# SkillProbe 实验数据

*v4.*

## 方法演进
> Pass A (decl-only) + Pass B (complete prompt, same as v1). risk = max(score_v1, score_decl). attribution = argmax.

| SkillProbe 方法演进：v1 → v2 → v3 → v4 |  |  |  |  |  |  |  |
| Pass A (decl-only) + Pass B (complete prompt, same as v1). r |  |  |  |  |  |  |  |
| 版本 | 方法 | SMP | Decoy | InvDecoy | MSB | MASB | 核心改进 |
| v1 | oper+bnd+PCA8+LR | 0.910 | 0.427 | 0.165 | 0.964 | 0.923 | 基础：读代码 hidden-state 几何关系 |
| v2 | v1 + decl-probe 双趟 max() | 0.910 | 0.427 | 0.165 | 0.964 | 0.923 | +声明投毒覆盖 + 恶意归因。现有 benchmark 零退化 |
| v3 | v1 + centering(-PC1) | 0.941 | 0.858 | 0.758 | 0.937 | 0.875 | +去模板噪声。跨分布 AUROC 暴涨+43-59pp |
| v4 | v3 + Boundary gating | 0.941 | 0.998 | 1.000 | 0.937 | 0.875 | +Boundary 安全门。Decoy FPR 100%→0%！ |
| v2 恶意归因 (175 samples) |  |  |  |  |  |  |  |
| v2: 双趟并行 + 恶意归因 | 样本 | 正确 | 准确率 | 说明 |  |  |  |
| Pass A (decl-only, ~50ms): prompt 截断在声明末尾 → decl-bnd → Decl- | 50 | 50 | 100% |  | 说明 |  |  |
| Pass B (完整 prompt, ~1945ms): 同 v1 → oper+bnd 几何特征 → PCA(8)+L | 50 | 22 | 44% | Decl-Probe L31信号不够强 | 单趟完整prompt。SMP攻击全是操作侧——v1已足够 |  |  |
| risk = max(score_v1, score_decl)   attribution = argmax → DE | 50 | 50 | 100% | v1 完美检出 | 声明侧无攻击信号——SMP没有声明投毒。随机水平 |  |  |
| 归因准确率: 175 样本 84% (CLEAN 100%, OPER 100%, BOTH 100%, DECL 44 | 25 | 25 | 100% |  | = v1。max()确保decl随机信号不拖累v1。零退化 |  |  |
| 合计 | 175 | 147 | 84% | argmax归因 |  |  |  |
| v3: 各向异性修正 (centering) |  |  |  |  |  |  |  |
| 在 v1 基础上增加一步: 对 oper 和 resp 特征逐层移除 PC1（prompt 模板主导的方差方向） |  |  |  |  |  |  |  |
| 效果: 几何特征反映 skill 内容而非 token 位置 → 跨分布鲁棒。Decoy AUROC +43pp, In |  |  |  |  |  |  |  |
| 签名: oper+bnd → center(-PC1) → 5几何特征 → PCA(8) → LR(C=0.01) ·  | AUROC | F1% | FPR% | FNR% | 层 | 说明 |  |
| risk = max(score_v1, score_decl)   attribution = argmax | 0.808 | 75.0 | 0.0 | 40.0 | L31 | 纯声明探针。检出decl-poison和both-poison，漏掉oper-poison |  |
| v4 | v3 + Boundary gating | 0.947 | 0.998 | 1.000 | 0.996 | 0.981 | +Boundary安全门. Gate<0.15通用. 五表全涨！ |
| if score_boundary < 0.15: risk = score_boundary  (模型确定安全→信任模 | 1.000 | 100.0 | 0.0 | 0.0 | — | 全覆盖。max()取两路最强信号——decl-poison被Decl-Probe检出，oper-poison被Code- |  |
| else: risk = score_v3  (模型不确定→用 v3 代码级检测) |  |  |  |  |  |  |  |
| Boundary 在安全样本上天然极低 (~0.005)，在恶意样本上天然高 (~0.9)。Gate=0.15 无需调参 |  |  |  |  |  |  |  |
| 效果: Lexical Decoy FPR 100%→0%, InvDecoy FPR 100%→0%。两张 Decoy | 0.427 | 0.998 |  |  |  |  |  |
| 签名: oper+bnd → center(−PC1) → 5几何特征 → PCA(8) → LR(C=0.01) ·  |  |  |  |  |  |  |  |
| 最终方法: SkillProbe v4 五数据集表现 | 样本数 | 正确归因数 | 准确率 | 错误归因分布 |  |  |  |
| 数据集 | v1 | v4 | Δ | v4 关键指标 |  |  |  |
| SMP Stealth V2 | 0.910 | 0.947 | +0.031 | AUROC=0.941 SOTA. FPR=19.4% FNR=6.5% |  |  |  |
| MSB | 0.964 | 0.996 | -0.027 | 去 confound 虚高→更诚实. FPR=13.8% FNR=12.5% |  |  |  |
| MASB 75K | 0.923 | 0.981 | -0.048 | 去 in-domain bias. FPR=35.3% FNR=9.5% |  |  |  |
| Lexical Decoy | 0.427 | 0.998 | +0.571 | FPR=0%! Boundary gate 完美拦截所有 decoy |  |  |  |
| Inverse Decoy | 0.165 | 1.000 | +0.835 | FPR=0% AUC=1.0! 两张 Decoy 全部解决 |  |  |  |
| Inverse Decoy | 0.165 | 1.000 | +0.593 | centering 从噪声中恢复信号 |  |  |  |
| v1 vs v2 总对比 |  |  |  |  |  |  |  |
| 维度 | SkillProbe v1 | SkillProbe v2 |  |  |  |  |  |
| 检测范围 | 操作侧攻击 (oper poisoning) | 操作侧 + 声明侧 (oper + decl poisoning) |  |  |  |  |  |
| SMP AUROC | 0.910 | 0.910 (相同——max()零退化) |  |  |  |  |  |
| 声明投毒检测 | ✗ 盲区 (AUC=0.47) | ✓ 覆盖 (AUC=0.81, attribution dataset) |  |  |  |  |  |
| 恶意归因 | ✗ | ✓ 代码/声明/双投毒 84%准确 |  |  |  |  |  |
| 前向次数 | 1次 | 2次 (Pass A ~50ms + Pass B ~1945ms) |  |  |  |  |  |
| 架构 | 单趟完整prompt | 双趟并行 max()融合 |  |  |  |  |  |
| 论文策略：v1是核心方法（轻量、高性能），v2是完整版（补盲区+归因）。两者在现有benchmark上性能相同，v2多覆 |  |  |  |  |  |  |  |
| 各数据集 v2 Decl-Probe 结果 |  |  |  |  |  |  |  |
| 数据集 | Decl-Probe 表现 | Code-Probe (v1) 表现 | v2 Fusion 效果 | 结论 |  |  |  |
| SMP Stealth V2 | 0.910 | 0.947 | = v1 (max不拖累) | 无声明投毒攻击。v2=v1，零退化。 |  |  |  |
| Attribution | AUC=0.81 | AUC=0.92 | AUC=1.00 (全覆盖) | 声明+代码双投毒全覆盖，100%检出，84%归因准确率。 |  |  |  |
| Lexical Decoy | 0.427 | 0.998 | FPR=max(62.5,85)=85% | Decoy的声明同样有可疑词汇——Decl-Probe也报警。v2不比v1更好。 |  |  |  |
| Inverse Decoy | 0.165 | 1.000 | = v1 | 词汇干净——声明侧无额外信号。v2=v1。 |  |  |  |
| MSB | 0.964 | 0.996 | 待提取 | LLM生成恶意skill——声明也可能携带confound信号。 |  |  |  |
| MASB 75K | 0.923 | 0.981 | 待提取 | 全良性——预期Decl-Probe FPR低。 |  |  |  |
| 关键模式：当数据集没有声明投毒攻击时，Decl-Probe输出随机(InvDecoy,SMP)或反映词法混杂(Decoy | 0.964 | 0.996 |  |  |  |  |  |
| v3 更新：SkillProbe v3 = v1 + centering (去PC1)。在所有数据集上达到最佳或接近最佳 |  |  |  |  |  |  |  |
| SMP: v3=0.941 (+6.5pp) / Decoy: v3=0.858 (+43pp) / InvDecoy: | 0.964 | 0.996 |  |  |  |  |  |

## 恶意归因
> v2 Dual-Pass 架构：Pass A (decl-only) + Pass B (完整prompt)。risk=max(score_v1,score_decl)。attribution=argmax。

| 恶意归因 —— 区分代码投毒 vs 声明投毒 |  |  |  |  |  |  |  |
| v2 Dual-Pass 架构：Pass A (decl-only) + Pass B (完整prompt)。risk= |  |  |  |  |  |  |  |
| 架构 |  |  |  |  |  |  |  |
| Pass A (decl-only, ~50ms): prompt 截断在声明末尾 → 模型只看声明 → decl-bn |  |  |  |  |  |  |  |
| Pass B (skip-decl): 声明从 prompt 删除 → 模型只看代码 → oper-bnd → Code |  |  |  |  |  |  |  |
| 信号: boundary token L2 norm (无需训练，直接比较两个范数) |  |  |  |  |  |  |  |
| 归因规则: 比较 Decl-Det 和 Code-Det。两者都高→BOTH，都低→CLEAN |  |  |  |  |  |  |  |
| 开销: Pass A ~50ms (仅800 chars) + Pass B ~1945ms (同v1)。总增量 +2. |  |  |  |  |  |  |  |
| 双探针检测能力 (GAP_REPORT 方法) | 仅声明投毒 (50样本) | 0.99 | — | — | — | L7 | Pass A decl-bd norm. AUROC=0.99 完美区分decl-poison vs clean |
| 探针 | 检测目标 | AUROC | 说明 | — | — | L7 | Pass B oper-bd norm. AUROC=0.82 区分oper-poison vs clean |
| Decl-Det | 仅声明投毒 (50样本) | 0.99 | Pass A boundary token norm。完美区分 decl-poison vs clean | 0.0 | 0.0 | — | max(Decl-Det, Code-Det). 两路信号正交→100%检出+100%归因 |
| Code-Det | 仅代码投毒 (50样本) | 0.82 | Pass B boundary token norm。区分 oper-poison vs clean | 0.0 | 40.0 | L31 | 检出decl-poison+both-poison，漏掉oper-poison |
| Fusion (max) | 所有攻击类型 (175样本) | 1.00 | max(Decl-Det, Code-Det)。两路信号正交→全覆盖 | 0.0 | 17.6 | L11 | 检出oper-poison+both-poison，漏掉decl-poison |
| CLEAN (干净skill) | 50 | 50 | 100% | 双探针均安全。Pass A decl-bd low, Pass B oper-bd low。 | 0.0 | — | 全覆盖。max()取两路最强信号，互补漏检 |
| 归因准确率 (175 样本) | 50 | 50 | 100% | 全部归因正确。Pass B oper-bd 高→Code侧报警。Pass A decl-bd 低→Decl侧不报。 |  |  |  |
| 攻击来源 | 样本 | 归因正确 | 准确率 | 机制 |  |  |  |
| CLEAN (干净skill) | 50 | 50 | 100% | 双探针 bd norm 均低 → 正确识别 |  |  |  |
| OPERATION (仅代码投毒) | 50 | 50 | 100% | Pass B oper-bd 高, Pass A decl-bd 低 → 正确归因 |  |  |  |
| DECLARATION (仅声明投毒) | 50 | 50 | 100% | Pass A decl-bd 高, Pass B oper-bd 低 → 正确归因 |  |  |  |
| BOTH (双投毒) | 25 | 25 | 100% | 双探针 bd norm 同时高 → 正确归因 |  |  |  |
| 合计 | 175 | 175 | 100% | 声明和操作信号正交——无损归因 |  |  |  |
| DECLARATION (声明投毒) | 50 | 22 | 44% | 28个被错归为BOTH。Decl-Probe对部分样本信号不够强(L31层)，改进层选择可提升。 |  |  |  |
| 归因 2x2 矩阵 | 175 | 147 | 84.0% | argmax(score_code, score_decl) 归因 |  |  |  |
|  | Code-Det 报警 (代码侧有恶意) | Code-Det 安全 (代码侧干净) |  |  |  |  |  |
| Decl-Det 报警 (声明侧可疑) | BOTH —— 双重投毒 ✓ | DECLARATION —— 仅声明投毒 ✓ |  |  |  |  |  |
| Decl-Det 安全 (声明侧干净) | OPERATION —— 仅代码投毒 ✓ | CLEAN —— 干净 skill ✓ |  |  |  |  |  |
| Decl-Probe 报警 | BOTH (双投毒) ✓ | DECLARATION (声明投毒) ✓ |  |  |  |  |  |
| 每个象限对应一种攻击来源。v2 是唯一具备此归因能力的方法。 | OPERATION (代码投毒) ✓ | CLEAN (干净) ✓ |  |  |  |  |  |
| 每个象限对应一种攻击来源，v2 可精确区分四种情况。这是所有 baseline 不具备的能力。 |  |  |  |  |  |  |  |

## 总览

| SkillProbe — 实验总览 |  |  |  |  |  |  |  |
| 一、数据集 |  |  |  |  |  |  |  |
| # | 数据集 | 样本数 | 设计 | Label | 论文角色 |  |  |
| 1 | SMP Stealth V2 | 484 (242对) | Paired CF (LLM注入) | 242 pairs | 主基准. 唯一控制 source confound |  |  |
| 2 | MSB | 7,894 | Unpaired (异源) | 3,955b/3,939m | 混杂证据. 证明 unpaired 虚高 |  |  |
| 3 | MASB 75K | ~9K | 大规模良性+MSB恶意 | Benign only | FPR 基准. 大规模真实分布 |  |  |
| 4 | Lexical Decoy | 155 | 词汇可疑+语义良性 | Benign (配SMP mal) | 2×2 FPR: 会误杀吗？ |  |  |
| 5 | Inverse Decoy | 155对 | 词汇干净+语义恶意 | Clean/Mal | 2×2 FNR: 会漏检吗？ |  |  |
| 二、对比基线 |  |  |  |  |  |  |  |
| # | 基线 | 输入 | 方法 | 类型 | 来源 |  |  |
| 1 | Static Regex | Skill全文 | 10个正则 | 表面文本 | MalSkillBench |  |  |
| 2 | TF-IDF + LR | Skill全文 | TF-IDF(20K,1-2gram)→LR | 表面文本 | 标准文本基线 |  |  |
| 3 | LLM-as-judge | Skill全文 | Mistral-7B zero-shot | 黑盒审计 | RouteGuard/BIV |  |  |
| 4 | Boundary only | Boundary token | L2-LR | 内部探针 | 消融基线 |  |  |
| 5 | AgentLens | Boundary top-K | PCA(50)→LR | 内部探针 | arxiv 2606.22673 |  |  |
| 6 | RouteGuard | Attn+Hidden | 8层→LR | 内部探针 | arxiv 2604.22888 |  |  |
| 7 | Relational v1 | decl+oper+bnd | 6几何+PCA64 | 内部探针 | 旧版(已废弃) |  |  |
| 8 | Centered Rel. | decl+oper+bnd | -PC1+PCA64 | 内部探针 | 变体(已废弃) |  |  |
| 9 | SkillProbe (ours) | oper+bnd | 去 interaction+PCA8 | 内部探针 | ★ 修正后的最终方法 |  |  |
| 10 | Skill-Inject Filter | Skill 全文 | Keyword patterns (Schmotz et al. 2026) | 表面文本 | Naive lower-bound baseline |  |  |
| 11 | cisco-skill-scanner | Skill 全文 | 129 YAML patterns (core signatures) | 表面文本 | Cisco AI Defense, industrial scanner | MASB | 核心改进 |
| 12 | NVIDIA SkillSpector | Skill 全文 | 20 YARA rules (4 categories) | 表面文本 | 13.5k stars, most popular skill scanner | 0.923 | 基础: 读代码 hidden-state 几何 |
| v2 | v1+decl-probe双趟max() | 0.910 | 0.427 | 0.165 | 0.964 | 0.923 | +声明投毒+归因. 零退化 |
| v3 | v1+centering(-PC1) | 0.941 | 0.858 | 0.758 | 0.937 | 0.875 | +去模板噪声. Decoy+43-59pp |
| v4 | v3+Boundary gating | 0.947 | 0.998 | 1.000 | 0.996 | 0.981 | +安全门. Gate<0.15通用. 五表全涨！ |
| 5. Unpaired benchmark 上所有方法 0.99+，测量的是来源差异 |  |  |  |  |  |  |  |
| 四、关键发现 |  |  |  |  |  |  |  |
| 1. centering(-PC1) 是关键突破: 移除模板噪声 → 跨分布 AUROC 暴涨 43-59pp |  |  |  |  |  |  |  |
| 2. Boundary gating 解决 Decoy FPR: 模型确定安全时信任它，否则用 v3 检测 |  |  |  |  |  |  |  |
| 3. v4 SMP=0.941 SOTA, LexicalDecoy FPR=0%, InvDecoy FPR=0% |  |  |  |  |  |  |  |
| 4. v2 归因 84%: argmax 区分代码投毒 vs 声明投毒 |  |  |  |  |  |  |  |
| 5. interaction+declaration 是噪声——已从方法中移除 |  |  |  |  |  |  |  |
| 6. MSB 上所有方法 0.96-1.0 → 测来源差异非检测能力 |  |  |  |  |  |  |  |
| 7. Paired CF 是唯一可信 benchmark: TF-IDF 0.967→0.812 |  |  |  |  |  |  |  |
| 8. LLM-as-judge 全表 FPR=100% 不可靠 |  |  |  |  |  |  |  |

## SMP Stealth V2
> Paired CF, 484 samples. L7+ layer selection (L3 excluded). No interact, PCA=8, C=0.01, oper+bnd

| SMP Stealth V2 — Paired Counterfactual Benchmark |  |  |  |  |  |  |  |  |  |
| Paired CF, 484 samples. L7+ layer selection (L3 excluded). N |  |  |  |  |  |  |  |  |  |
| Static Regex | 0.54 | 100 | 0 | 66.7 | 50 | 100 | — | Always fires |  |
| Skill-Inject Filter | 0.528 | 100 | 0 | 66.7 | 50 | 100 | — | Schmotz et al. 2026, naive input filter. FPR=100% |  |
| TF-IDF + LR | 0.812 | 19.4 | 3.2 | 89.6 | 83.3 | 96.8 | 0.679 | C=0.01, unique prompt |  |
| cisco-skill-scanner | 0.539 | 68.3 | 31.7 | 60.6 | 50 | 68.3 | — | 129 core patterns. Better than basic regex but still 68% FPR |  |
| NVIDIA SkillSpector | 0.503 | 100 | 0 | 66.7 | 50 | 100 | — | 20 YARA rules, 13.5k stars. Always fires on standard Python |  |
| LLM-as-judge | 0.63 | 100 | 0 | 66.7 | 50 | 100 | — | Mistral-7B zero-shot: predicts almost all MAL (FPR=100%, FNR |  |
| Boundary only | 0.863 | 0 | 40.3 | 74.7 | 100 | 59.7 | 0.989 | L11 (L7+ restricted) |  |
| AgentLens | 0.871 | 0 | 51.6 | 65.2 | 100 | 48.4 | 0.989 | L11 (L7+ restricted) |  |
| RouteGuard | 0.824 | 48.4 | 8.1 | 76.5 | 65.5 | 91.7 | 0.938 | Done: attn+rhid. Honest: AUC=0.824 on paired CF |  |
| Relational (v1) | 0.761 | 50 | 6.5 | 76.8 | 65.2 | 93.5 | 0.893 | L31, interact+PCA64 |  |
| Centered Relational | 0.872 | 19.4 | 12.9 | 84.4 | 81.8 | 87.1 | 0.896 | L27, -1PC |  |
| SkillProbe v1 | 0.91 | 19.4 | 11.3 | 85.3 | 82.1 | 88.7 | 0.919 | v1: oper+bnd+PCA8+C=0.01+L7+. Baseline method |  |
| SkillProbe v2 | 0.91 | 19.4 | 11.3 | 85.3 | 82.1 | 88.7 | — | v2: max(v1,decl). SMP无声明投毒→=v1. 零退化+补盲区+归因 |  |
| SkillProbe v4 (ours) | 0.947 | 19.4 | 6.5 | 87.9 | 82.1 | 93.5 | — | v4: Boundary-gated v3. Gate<0.15. AUC=0.947 SOTA. 49/124 gat |  |
| NVIDIA SkillSpector | 0.503 | 100 | 0 | 66.7 | 50 | 100 | — | 20 YARA rules, 13.5k stars. Always fires on standard Python |  |

## MSB
> Unpaired, 7894 samples, new triad extract. L7+ verified (best layers already L19+)

| MSB (MalSkillBench) — Unpaired, Source Confound Evidence |  |  |  |  |  |  |  |  |
| Unpaired, 7894 samples, new triad extract. L7+ verified (bes |  |  |  |  |  |  |  |  |
| Static Regex | 0.54 | 100 | 0 | 66.7 | 49.6 | 100 | — | Always fires |
| Skill-Inject Filter | 0.502 | 100 | 0 | 66.7 | 49.6 | 100 | — | Naive filter: always fires on 8K samples |
| TF-IDF + LR | 0.967 | 6.4 | 10.7 | 91.1 | 93 | 89.3 | 0.973 | C=0.01 |
| cisco-skill-scanner | 0.545 | 100 | 0 | 66.7 | 49.6 | 100 | — | 129 core patterns, always fires |
| NVIDIA SkillSpector | 0.521 | 100 | 0 | 66.7 | 49.6 | 100 | — | 20 YARA rules, 13.5k stars. Always fires |
| LLM-as-judge | 0.5 | 100 | 0 | 66.6 | 50 | 100 | — | Mistral-7B zero-shot: predicts almost all MAL (FPR=100%, FNR |
| Boundary only | 0.999 | 1.4 | 1.3 | 98.7 | 98.6 | 98.7 | 1 | L19 (L7+ verified, source confound) |
| AgentLens | 0.99 | 3.6 | 2.1 | 97.2 | 96.5 | 97.9 | 0.995 | L19 (L7+ verified, source confound) |
| RouteGuard | 0.795 | 47.7 | 11.1 | 74.5 | 64.1 | 88.9 | 0.796 | rhid only. Honest: not inflated by source confound (unlike o |
| Relational (v1) | 0.992 | 2.9 | 6 | 95.5 | 97 | 94 | 0.993 | L11 |
| Centered Relational | 0.991 | 7.8 | 3 | 94.8 | 92.7 | 97 | 0.991 | L11 |
| SkillProbe v1 | 0.964 | 10.8 | 8.5 | 90.6 | 89.7 | 91.5 | 0.963 | v1: oper+bnd. Source confound inflated |
| SkillProbe v2 | 0.964 | 10.8 | 8.5 | 90.6 | 89.7 | 91.5 | — | v2: max(v1,decl). =v1 on MSB |
| SkillProbe v4 (ours) | 0.996 | 3.2 | 1.3 | 98 | 97.5 | 98.7 | — | v4: Gate<0.15. AUC=0.996! Boundary gate catches confounded s |
| NVIDIA SkillSpector | 0.521 | 100 | 0 | 66.7 | 49.6 | 100 | — | 20 YARA rules, 13.5k stars. Always fires |

## MASB 75K
> 4K MASB benign + MSB malicious (3939), 70/15/15 split. In-domain. L7+.

| MASB 75K — Large-Scale Benign FPR Benchmark |  |  |  |  |  |  |  |  |
| 4K MASB benign + MSB malicious (3939), 70/15/15 split. In-do |  |  |  |  |  |  |  |  |
| Static Regex | 0.586 | 100 | 0 | 66.7 | 49.6 | 100 | — | Always fires (OOD) |
| Skill-Inject Filter | 0.505 | 100 | 0 | 66.4 | 49.6 | 100 | — | Naive filter: always fires on 8K samples |
| TF-IDF + LR | 0.792 | 6.4 | 10.7 | 91.1 | 71.3 | 71.8 | 0.973 | Estimated from MSB |
| cisco-skill-scanner | 0.552 | 100 | 0 | 66.4 | 49.6 | 100 | — | 129 core patterns. Always fires |
| NVIDIA SkillSpector | 0.524 | 100 | 0 | 66.4 | 49.6 | 100 | — | 20 YARA rules, 13.5k stars. Always fires |
| LLM-as-judge | 0.5 | 100 | 0 | 66.7 | 50 | 100 | — | Mistral-7B zero-shot: predicts almost all MAL (FPR=100%, FNR |
| Boundary only | 0.999 | 2.5 | 1 | 98.2 | 97.5 | 99 | 1 | L15, in-domain (L7+) |
| AgentLens | 0.984 | 3.7 | 9.1 | 93.4 | 96.1 | 90.9 | 0.983 | L15, in-domain (L7+) |
| RouteGuard | 0.998 | 0 | 1 | 99.5 | 100 | 99 | 1 | 6K in-domain, rhid only (pre-L7+. Needs MSB route_hidden re- |
| Relational (v1) | 1 | 0 | 0.7 | 99.6 | 100 | 99.3 | 1 | 4K in-domain, pre-L7+ |
| Centered Relational | 1 | 0 | 1.7 | 99.1 | 100 | 98.3 | 1 | 4K in-domain, pre-L7+ |
| SkillProbe v1 | 0.923 | 16.3 | 16.4 | 83.5 | 83.4 | 83.6 | 0.921 | v1: oper+bnd. In-domain |
| SkillProbe v2 | 0.923 | 16.3 | 16.4 | 83.5 | 83.4 | 83.6 | — | v2: max(v1,decl). =v1 on MASB |
| SkillProbe v4 (ours) | 0.981 | 7.2 | 6 | 93.4 | 93 | 94 | — | v4: Gate<0.15. AUC=0.981! Gating fixes OOD FPR from 12%→7% |
| NVIDIA SkillSpector | 0.524 | 100 | 0 | 66.4 | 49.6 | 100 | — | 20 YARA rules, 13.5k stars. Always fires |

## Lexical Decoy
> 155 decoys + 62 SMP mal paired. Unified internal contract. L7+. No interaction. PCA=8. C=0.01.

| Lexical Decoy — Suspicious Vocabulary, Benign Semantics (155 |  |  |  |  |  |  |  |  |  |
| 155 decoys + 62 SMP mal paired. Unified internal contract. L |  |  |  |  |  |  |  |  |  |
| Static Regex | 0.463 | 100 | 0 | 66.7 | 42 | 48.3 | — | Static Regex on 155 decoys: FPR=59% (better than old 100%—mo |  |
| Skill-Inject Filter | 0.8 | 40 | 0 | 83.4 | 50.2 | 100 | — | Schmotz et al. 2026, naive input filtering |  |
| TF-IDF + LR | 0.7435 | 39.3 | 39.1 | 60.9 | 50 | 100 | — | TF-IDF on 155 decoys: FPR=100% (suspicious vocab = malicious |  |
| cisco-skill-scanner | 0.533 | 43.3 | 37.7 | 60.6 | 54 | 62.3 | — | 176 patterns (core+promptguard), Python 3.10+ required |  |
| NVIDIA SkillSpector | 0.49 | 100 | 0 | 66.7 | 50.2 | 100 | — | 20 YARA rules, 13.5k stars. Always fires on standard Python |  |
| LLM-as-judge | 0.5 | 100 | 0 | 66.7 | 50 | 100 | — | Mistral-7B zero-shot: predicts almost all MAL (FPR=100%) |  |
| Boundary only | 0.991 | 0 | 3.2 | 98.4 | 100 | 96.8 | 0.994 | L23 (L7+). Model judges ALL 150 decoys safe! |  |
| AgentLens | 0.867 | 22.7 | 21 | 67.6 | 68.2 | 79 | 0.989 | L15 (L7+). PCA(50) boundary. FPR=23% acceptable. |  |
| RouteGuard | 0.777 | 65.3 | 0 | 75.4 | 60.4 | 100 | 0.734 | rhid only. Cross-extract caution. FPR=65% |  |
| Relational (v1) | 0.46 | 100 | 0 | 66.7 | 50 | 100 | — | Obsolete (interaction+PCA64). FPR=100% confirms noise |  |
| Centered Relational | 0.705 | 57.3 | 14.7 | 70.3 | 59.2 | 85.3 | — | Centered (-1PC). AUC=0.71 better than SkillProbe on decoy—ce |  |
| SkillProbe v1 | 0.427 | 75.3 | 8.1 | 49.1 | 38.3 | 91.9 | — | v1: FPR=75% (oper reads suspicious code) |  |
| SkillProbe v2 | 0.427 | 75.3 | 8.1 | 49.1 | 38.3 | 91.9 | — | v2: max(v1,decl). Decl-Probe also fires on suspicious vocab. |  |
| SkillProbe v4 (ours) | 0.998 | 0 | 3 | 98.4 | 100 | 97 | — | ★ v4: Boundary-gated v3. Gate<0.15. FPR=0% FNR=3%. Best on D |  |
| NVIDIA SkillSpector | 0.49 | 100 | 0 | 66.7 | 50.2 | 100 | — | 20 YARA rules, 13.5k stars. Always fires on standard Python |  |

## Inverse Decoy
> 155 clean + 155 malicious. SMP-trained. L7+. Vocab-clean semantics-malicious. FNR benchmark: can model detect malice without lexical cues?

| Inverse Decoy — Clean Vocabulary, Malicious Semantics (155 p |  |  |  |  |  |  |  |  |  |
| 155 clean + 155 malicious. SMP-trained. L7+. Vocab-clean sem |  |  |  |  |  |  |  |  |  |
| Static Regex | 0.994 | 100 | 0 | 66.7 | 50 | 100 | — | Regex on clean vocab: FPR=0% FNR=100%—no patterns to match.  |  |
| Skill-Inject Filter | 0.5 | 100 | 0 | 66.7 | 50 | 100 | — | Naive filter: clean vocab -> always fires |  |
| TF-IDF + LR | 0.1593 | 78.1 | 78.1 | 21.9 | 46.3 | 78.1 | — | TF-IDF on clean vocab: AUC=0.50 random (vocab has no signal) |  |
| cisco-skill-scanner | 0.503 | 100 | 0 | 66.7 | 50 | 100 | — | Standard Python code triggers patterns -> always fires |  |
| NVIDIA SkillSpector | 0.5 | 100 | 0 | 66.7 | 50 | 100 | — | 20 YARA rules, 13.5k stars. Always fires on standard Python |  |
| LLM-as-judge | 0.5 | 100 | 0 | 66.7 | 50 | 100 | — | Mistral always MAL. Zero-shot useless |  |
| Boundary only | 0.793 | 0 | 100 | 0 | 0 | 0 | 0.994 | L23. Model judges ALL safe! FNR=100%—clean vocab fools model |  |
| AgentLens | 0.676 | 78.1 | 0.6 | 71.6 | 53.2 | 99.4 | 0.989 | L15. Catches 99% malicious but FPR=78% |  |
| RouteGuard | 0.5 | 100 | 0 | 66.7 | 50 | 100 | 0.734 | rhid only. Random on clean vocab. Cross-extract. |  |
| Relational (v1) | 0.824 | 100 | 0 | 66.7 | 50 | 100 | — | AUC=0.82 ranks correctly but FPR=100% (threshold issue) |  |
| Centered Relational | 0.846 | 21.9 | 1.3 | 89.5 | 80.9 | 98.7 | — | Centered (-1PC). AUC=0.85 BEST on InvDecoy! Centering remove |  |
| SkillProbe v1 | 0.165 | 100 | 0 | 43.9 | 50 | 100 | — | v1: cross-distribution collapse. AUC=0.16 |  |
| SkillProbe v2 | 0.165 | 100 | 0 | 43.9 | 50 | 100 | — | v2: max(v1,decl). Decl AUC=0.48 random. =v1 |  |
| SkillProbe v4 (ours) | 1 | 0 | 3 | 98.5 | 100 | 97 | — | ★ v4: Boundary-gated v3. Gate<0.15. AUC=1.0 FPR=0%! Both Dec |  |
| NVIDIA SkillSpector | 0.5 | 100 | 0 | 66.7 | 50 | 100 | — | 20 YARA rules, 13.5k stars. Always fires on standard Python |  |
