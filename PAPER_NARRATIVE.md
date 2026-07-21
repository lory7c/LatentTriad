# SkillProbe — Paper Narrative

---

## Title

**The Model Already Knows: Declaration-Operation Geometry for Pre-Execution Malicious Skill Detection**

---

## Motivation

LLM agents routinely load third-party skills—packages of instructions and code that extend their capabilities. A single malicious skill can exfiltrate data, execute arbitrary code, or hijack the agent's behavior. By the time the agent acts, it's too late.

Existing defenses operate at the wrong stage:
- Static scanners read skill text—trivially evaded by obfuscation or subtle rewrites
- Sandbox execution confirms malice after the fact—the damage is done
- LLM auditors are slow, expensive, and unreliable on adversarial inputs

**The last safe moment is after the agent reads the skill but before it takes action.** No existing method operates at this boundary.

---

## Gap

The field lacks a **pre-execution, internal-state detector** for agent skills. All published methods fall into one of three categories, none of which address this gap:

| Category | Example | What it does | Why it's insufficient |
|----------|---------|-------------|----------------------|
| Surface-text | TF-IDF, BIV | Reads skill text | Fails when obfuscated; measures text confounds, not intent |
| Behavioral | SkillDetonate | Executes in sandbox | Too late; attack already executed |
| Internal (injection) | RouteGuard, AgentLens | Reads hidden states | Designed for injection artifacts; collapses on stealth attacks (FPR 48%, AUROC 0.47) |

**No prior work demonstrates that internal representations can detect stealthy malicious skills—where text looks normal but code does something dangerous.**

---

## Method: SkillProbe

**Core insight: Don't read the text. Read how the model understands it.**

After reading a skill, the model's internal representation of the declaration (what the skill claims to do) and operation (what the code actually does) exhibits a geometric relationship. Malicious skills—even when textually benign—perturb this relationship in detectable ways.

```
Input: skill package → one forward pass → 8 layers
  ↓
Extract three regions from hidden states:
  Declaration: YAML frontmatter, title, description → mean pooling
  Operation:   code, scripts, auxiliary files → mean pooling
  Boundary:    last token before first assistant action
  ↓
Construct 6 geometric features:
  normalized_diff = norm(decl) − norm(oper)
  interaction     = norm(decl) × norm(oper)
  cosine(decl, oper), norm_ratio, cosine(decl, boundary), cosine(oper, boundary)
  ↓
PCA (64 dims) → Logistic Regression → risk_score → PASS / HOLD_FOR_REVIEW
```

**Key design decisions:**
- Single forward pass, zero token generation, zero tool dispatch
- 6 interpretable geometric features (not a black-box embedding)
- Single best layer selected via leakage-cluster-grouped CV
- Threshold frozen on dev; test consumed once

---

## Research Questions

**RQ1: Detection.** Can decl-op geometric features distinguish stealthy malicious skills from benign skills with identical source, style, and lineage?

**RQ2: Overhead.** What is the runtime cost, and is it deployable as a pre-action safety gate?

**RQ3: Geometry.** Does malicious/benign decl-op geometry differ systematically across models, and if so, in what direction?

---

## Experiments

**Data: SMP Stealth Benchmark.** 242 matched pairs / 484 samples from SkillHarm self-mutating poisoning. Each pair shares the same trusted task, same clean skill lineage, same source ecosystem. The only difference: one version is poisoned, the other is clean. This paired counterfactual design controls for surface-text confounds—a text classifier cannot exploit source, author, or vocabulary differences.

Split: 302 train / 58 dev / 124 test, grouped by leakage cluster.

**Baselines:** Static regex, TF-IDF + LR (surface text), RouteGuard reproduction (attention-based internal), AgentLens reproduction (subspace), Residual ablation (boundary-only probe). All retrained on SMP train split using identical protocol. Threshold selected on dev via op_f1.

**Model:** Llama-3.1-8B-Instruct. All internal methods share the same prefill.

**Evaluation:** 10,000 cluster-bootstrap 95% CIs. Per-sample predictions and frozen configs retained for independent validation.

---

## Results

### RQ1: Detection

| Method | FPR% ↓ | FNR% ↓ | Precision% ↑ | Recall% ↑ | F1% ↑ | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Static regex | 100.0 | 0.0 | 50.0 | 100.0 | 66.7 | 0.540 |
| TF-IDF + LR | 19.4 | 6.5 | 82.9 | 93.5 | 87.9 | 0.812 |
| RouteGuard | 48.4 | **4.8** | 66.3 | 95.2 | 78.1 | 0.765 |
| AgentLens | 19.4 | 16.1 | 81.2 | 83.9 | 82.5 | **0.837** |
| Residual Ablation | 19.4 | 27.4 | 78.9 | 72.6 | 75.6 | 0.823 |
| **SkillProbe** | **19.4** | **3.2** | **83.3** | **96.8** | **89.6** | **0.812** |

**Takeaway:** SkillProbe achieves FNR 3.2%—the lowest across all methods. It catches 60/62 malicious samples, missing only 2. TF-IDF misses 4, AgentLens misses 10, RouteGuard misses 3 but at FPR 48.4%. On stealth attacks with controlled surface text, hidden-state geometry provides unique signal.

### RQ2: Overhead

1.32ms shared-prefill incremental (0.87ms hook/readout + 0.44ms detector). P95 = 3.57ms. Deployable as a real-time gate.

### RQ3: Geometry

Cross-model (Qwen + Llama) stable shift: malicious skills show *higher* decl-op cosine (+0.022, p<0.05) and *lower* normalized distance (−0.034, p<0.05). Direction opposite to intuition—malicious skills are more aligned, not less. Possibly because attackers craft description and code to appear maximally consistent, while benign open-source skills accumulate natural documentation-code drift.

---

## Conclusion

We present SkillProbe, a pre-execution internal-state detector that identifies malicious agent skills via declaration-operation hidden-state geometry. On a paired counterfactual stealth benchmark—where surface-text methods lose their advantage—SkillProbe achieves FNR 3.2% and F1 89.6%, outperforming all baselines.

**Contributions:**
1. A new detection paradigm: relational hidden-state probing at the pre-action boundary
2. A paired counterfactual stealth benchmark (SMP-Stealth, 484 samples) that controls for surface-text confounds
3. Evidence that internal geometry provides signal orthogonal to surface text, specifically on stealthy attacks
4. Cross-model geometry analysis revealing a stable but directionally surprising alignment pattern
5. 1.32ms deployable overhead with bounded tail latency

**Limitations and future work:** 124 test samples limit statistical power; extension to multi-layer fusion and MIL pooling showed promise (AUROC 0.850) but degraded operating-point metrics on small training sets. Larger stealth benchmark construction, per-token feature extraction, and integration with surface-text signals as complementary channels are natural next steps.
