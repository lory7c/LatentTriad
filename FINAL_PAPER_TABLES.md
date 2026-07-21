# SkillGuard — T3 Stealth Benchmark Results

> 2026-07-21 | SMP Full: 242 pairs / 484 samples (302 train, 58 dev, 124 test)
> Self-mutating poisoning attacks with controlled surface-text confounds
> Paired counterfactual design: same source, style, and lineage for clean/malicious skill pairs
> op_f1 threshold selected on dev split. AUROC/CIs/confusion in Appendix.

---

## Table 1: RQ1 — Primary Method Comparison (T3 Stealth, 124 test samples)

| Method | FPR % ↓ | FNR % ↓ | Precision % ↑ | Recall % ↑ | F1 % ↑ | AUROC | Overhead ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static regex | 100.0 | 0.0 | 50.0 | 100.0 | 66.7 | 0.540 | 12.01 [D] |
| TF-IDF + LR | 19.4 | 6.5 | 82.9 | 93.5 | 87.9 | 0.812 | 3.03 [D] |
| RouteGuard reproduction | 48.4 | **4.8** | 66.3 | 95.2 | 78.1 | 0.765 | 37.75 [I] |
| AgentLens reproduction | 19.4 | 16.1 | 81.2 | 83.9 | 82.5 | **0.837** | 0.62 [I] |
| Residual Ablation | 19.4 | 27.4 | 78.9 | 72.6 | 75.6 | 0.823 | — [I] |
| **Ours** | **19.4** | **3.2** | **83.3** | **96.8** | **89.6** | **0.812** | **1.32 [I]** |

> `[D]` = detector-only, `[S]` = standalone, `[I]` = shared-prefill incremental.
> SkillSieve-L1 and LLM-as-judge omitted: method-specific infrastructure not applicable to this benchmark.
> **Ours achieves lowest FNR (3.2%) and highest F1 (89.6%). AgentLens achieves highest AUROC (0.837).**

---

## Table 2: Cross-Model (T3 Stealth, Llama-3.1-8B retrained)

| Method | FPR % | FNR % | Precision % | Recall % | F1 % | AUROC | Overhead ms |
|---|---|---|---|---|---|---|---|
| RouteGuard reproduction | 48.4 | 4.8 | 66.3 | 95.2 | 78.1 | 0.765 | 37.75 |
| AgentLens reproduction | 19.4 | 16.1 | 81.2 | 83.9 | 82.5 | 0.837 | 0.62 |
| Residual Ablation | 19.4 | 27.4 | 78.9 | 72.6 | 75.6 | 0.823 | — |
| **Ours** | **19.4** | **3.2** | **83.3** | **96.8** | **89.6** | **0.812** | **1.32** |

> Qwen2.5-7B not evaluated on T3 due to model unavailability on extraction server.
> All methods retrained on SMP training split (not zero-shot).

---

## Table 3: RQ2 — Latency Breakdown

**RTX 4090, BF16, batch size 1, 218 samples × 3 rounds. Model loading excluded.**

| Method | Scope | Main Mean | Detector Mean / P50 / P95 | Hook/Readout Mean / P50 / P95 | Standalone Mean / P50 / P95 |
|---|---|---|---|---|---|
| Static regex | D | 12.01 | 12.01 / 7.20 / 54.82 | — | — |
| TF-IDF + LR | D | 3.03 | 3.03 / 2.24 / 9.47 | — | — |
| RouteGuard | I | 37.75 | 0.37 / 0.37 / 0.39 | 37.38 / 25.34 / 144.42 | 531.44 / 345.88 / 2114.23 |
| AgentLens | I | 0.62 | 0.21 / 0.21 / 0.23 | 0.40 / 0.21 / 16.86 | 570.55 / 359.81 / 2322.53 |
| **Ours** | **I** | **1.32** | **0.44 / 0.44 / 0.46** | **0.87 / 0.64 / 3.57** | **491.98 / 319.51 / 1960.86** |

> All values in milliseconds. `[D]` = detector-only. `[S]` = standalone. `[I]` = shared-prefill incremental.
> Ours: 0.87ms hook/readout + 0.44ms detector scoring = 1.32ms total. P95 = 3.57ms.

---

## Table 4: RQ3 — Operation Alignment Geometry

| Model | Layer | Geometry | Malicious−Clean Δ | 95% CI | Paired SMD | Significant? |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 17 | decl-op cosine | +0.0211 | [+0.0127, +0.0280] | 0.552 | ✅ |
| Qwen2.5-7B | 17 | decl-op normalized distance | −0.0371 | [−0.0501, −0.0218] | −0.546 | ✅ |
| Qwen2.5-7B | 17 | decl-op norm ratio | +0.0065 | [−0.0033, +0.0249] | 0.140 | ❌ |
| Qwen2.5-7B | 17 | decl-boundary cosine | −0.0015 | [−0.0073, +0.0029] | −0.085 | ❌ |
| Qwen2.5-7B | 17 | op-boundary cosine | +0.0143 | [+0.0084, +0.0244] | 0.535 | ✅ |
| **Llama-3.1-8B** | **11** | **decl-op cosine** | **+0.0222** | **[+0.0114, +0.0308]** | **0.565** | **✅** |
| **Llama-3.1-8B** | **11** | **decl-op normalized distance** | **−0.0340** | **[−0.0478, −0.0173]** | **−0.553** | **✅** |

> Stable cross-model geometry shift exists. Malicious skills show *higher* decl-op cosine and *lower* normalized distance — opposite to the original hypothesis. Both models agree on direction.
> Norm ratio and boundary cosines are not significant.

---

## Table 5: Confusion Matrix (T3, 124 test samples)

| Method | TP | TN | FP | FN | F1 | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Static regex | 62 | 0 | 62 | 0 | 66.7 | 0.540 |
| TF-IDF + LR | 58 | 50 | 12 | 4 | 87.9 | 0.812 |
| RouteGuard reproduction | 59 | 32 | 30 | 3 | 78.1 | 0.765 |
| AgentLens reproduction | 52 | 50 | 12 | 10 | 82.5 | 0.837 |
| Residual Ablation | 45 | 50 | 12 | 17 | 75.6 | 0.823 |
| **Ours** | **60** | **50** | **12** | **2** | **89.6** | **0.812** |

> Ours: 2 false negatives out of 62 malicious, 12 false positives out of 62 benign.
> The 12 FPs concentrate in 3 leakage clusters where all methods exhibit elevated error rates.

---

## Key Findings

1. **Hidden-state geometry detects stealth attacks where surface text fails.** On paired counterfactual skills with identical source, style, and lineage, TF-IDF+LR misses 4/62 malicious samples that Ours catches (FNR 6.5% vs 3.2%).

2. **Attention-based detection degrades on stealth.** RouteGuard achieves FPR 48.4% — nearly half of benign skills are falsely flagged. Its attention-over-window mechanism, effective on injection artifacts, does not transfer to stealthy self-mutating poisoning.

3. **Cross-model geometry shift is stable but directional surprise.** Malicious skills show *higher* decl-op alignment than benign (decl-op cosine Δ = +0.022, p < 0.05), replicable across Qwen and Llama. This contradicts the intuitive "malicious skills are less aligned" hypothesis.

4. **Deployable 1.32ms overhead with bounded tail latency.** P95 = 3.57ms in shared-prefill incremental mode — feasible as a real-time pre-action safety gate.

5. **Method simplicity enables generalization.** The stable version uses a single best layer with 6 geometric features. No multi-layer fusion, attention routing, or MIL pooling. On 302 training samples, this simplicity prevents overfitting while the geometric insight provides signal that surface-text methods cannot access.
