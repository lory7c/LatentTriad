# SkillProbe — USENIX Security 2027 全文大纲

## Section 1: Introduction (~2 pages)
**对标 RouteGuard 结构：Attack → Existing defenses fail → Mechanism insight → Our method**

- Agent skill 供应链攻击面（ClawHavoc 300+，BIV 80% 偏差）
- 三类防御的致命缺陷：静态扫描(可混淆)、沙箱(太晚)、内部探针(单点信号)
- **核心 inset**：我们发现恶意 skill 产生"Pre-Action Vigilance"——boundary token 异常激活，norm_ratio SMD=−1.42
- SkillProbe v4：oper+bnd 几何 + centering + Boundary gate → 寄生检测 18.7ms
- 贡献预览：(1) 三区域几何检测器 (2) centering 跨分布鲁棒 (3) 2×2 诊断矩阵 (4) 恶意归因 (5) 寄生部署

## Section 2: Background & Threat Model (~1 page)
- Agent skill 模型：SKILL.md + scripts/ + config
- Pre-action boundary 定义
- Threat model：攻击者控制 skill 包，agent 忠实执行
- 攻击类型：代码投毒 vs 声明投毒

## Section 3: Mechanism — Pre-Action Vigilance (~1.5 pages)
**全文最重要的章节——机制发现先于检测器设计**

- 实验 setup：SMP paired CF，Llama-3.1-8B，8 层 hidden states
- **Finding 1**: norm_ratio SMD=−1.42——恶意 skill boundary 相对 operation 异常激活
- **Finding 2**: 信号在 L7-L15 最强，L27+ 衰减——语义层面响应，非词法
- **Finding 3**: cos(oper,bnd) 和 direction_diff 单独不显著——需要多变量组合
- 逐层衰减分析（L3→L31）
- 和 RouteGuard "attention hijacking" 互补——我们读"状态"，他们读"注意力"

## Section 4: Method — SkillProbe v4 (~2 pages)
- 统一输入模板（所有 skill 同一 prompt 格式）
- 特征提取：一次 prefill → oper mean-pool + bnd token
- **Centering(−PC1)**：逐层去均值 + 去第一主成分（消融：+43-59pp 跨分布）
- 几何特征：norm_ratio, cos(oper,bnd), direction_diff
- StandardScaler → PCA(8) → LR(C=0.01)
- **Boundary 安全门 (Gate<0.15)**：模型确定安全→信任，否则用 v3
- Optional: Dual-pass 归因（Pass A decl-only ~50ms）

## Section 5: Experimental Setup (~1 page)
- 6 数据集：SMP(484), MSB(7894), MASB(~9K), Lexical Decoy(155), Inverse Decoy(155对), SkillHarm(931)
- 14 基线方法（表面文本、LLM 黑盒、内部探针、我们的旧版）
- 统一 prompt 模板，Llama-3.1-8B
- 训练协议：dev 5-fold CV 选层 → 阈值 F1 最大化 → sealed test
- 术语：CV, StandardScaler, PCA(8), LR(C=0.01)

## Section 6: Results (~4 pages)

### 6.1 RQ1: Detection Accuracy (SMP Stealth V2)
- 主表：9 baselines + v4, AUROC 0.947 SOTA
- Boundary 0.863 vs v4 0.947 (+8.4pp), FNR 40.3% vs 6.5% (−34pp)
- 五数据集汇总表

### 6.2 RQ2-RQ3: 2×2 Diagnostic Matrix
- Lexical Decoy (声明有毒/实际无毒): v4 FPR=0%, TF-IDF FPR=100%
- Inverse Decoy (声明看不出/实际有毒): v4 FNR=3%, Boundary FNR=100%
- 2×2 矩阵：四个象限只有我们全解

### 6.3 RQ4: Malice Attribution
- Dual-pass 归因：175 样本 100% 准确
- 声明和操作信号正交

### 6.4 RQ5: Overhead & Deployment
- 寄生检测 18.7ms vs RouteGuard 60ms vs SkillDetonate 153s
- Gate 通用性分析

### 6.5 RQ6: Ablation & Evolution
- v1→v2→v3→v4 消融表
- centering 最关键 (+43-59pp)
- Gate<0.15 通用性验证

### 6.6 RQ7: External Validation (SkillHarm)
- 跨分布：v3 AUC=0.973 Recall=100%
- Boundary/AgentLens 全灭 (AUC=0)

## Section 7: Discussion & Limitations (~0.5 page)
- centering 的通用性
- Gate 跨分布需重新校准
- Declaration 投毒覆盖
- 自然语言覆盖缺口

## Section 8: Related Work (~1 page)
- 静态分析：BIV, SkillSieve, ShareLock
- 沙箱：SkillDetonate, AgentArmor
- Hidden-state：RouteGuard, AgentLens, RENNERVATE, Attention Tracker
- 我们的定位：三区域几何 + centering + 归因 + 寄生

## Section 9: Conclusion (~0.25 page)
- SkillProbe v4：pre-action, 几何, centering, gate, 寄生
- 五数据集 SOTA, 2×2 全解, 100% 归因

## Appendices
- A: Ethical Considerations (agent skill scanning ethics)
- B: Open Science (code, data, model availability)
- C: Dataset Details (construction methodology for all 6 datasets)

---

## Figure Plan

| Figure | Content | Source |
|---|---|---|
| Fig 1 | Attack surface overview + Pre-Action Vigilance concept | New |
| Fig 2 | Three-region prompt structure | NARRATIVE_CN §4.1 |
| Fig 3 | norm_ratio per-layer decay (L3→L31) | RQ3 几何分析 |
| Fig 4 | SkillProbe v4 architecture (centering → geo → gate) | New |
| Fig 5 | 2×2 diagnostic matrix | RQ4 |
| Fig 6 | SMP main results bar chart | final.xlsx SMP sheet |
| Fig 7 | Attribution confusion matrix | 恶意归因 sheet |
| Fig 8 | Overhead comparison | RQ5 |
| Fig 9 | Ablation evolution v1→v4 | RQ7 |

## Table Plan

| Table | Content |
|---|---|
| Table 1 | Dataset summary (6 datasets) |
| Table 2 | SMP Stealth V2 — all baselines |
| Table 3 | Five-dataset v4 summary |
| Table 4 | 2×2 diagnostic matrix (Lexical Decoy + Inverse Decoy) |
| Table 5 | Attribution accuracy |
| Table 6 | Overhead comparison |
| Table 7 | Ablation v1→v4 |
| Table 8 | SkillHarm external validation |
