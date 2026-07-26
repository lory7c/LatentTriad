# Project Truth — SkillProbe

## Active Manuscript
- **File**: `output/doc/main.tex`
- **Template**: USENIX Security 2027 (usenix.sty)
- **Venue**: USENIX Security 2027 (deadline ~Feb 2027)

## Contribution Type
**Method** — SkillProbe v4: Pre-execution malicious agent skill detection via hidden-state geometric analysis with centering and boundary gating.

## Central Claim
Malicious agent skills induce measurable geometric tension between operation hidden states and the pre-action boundary token (norm_ratio SMD=-1.42). This "Pre-Action Vigilance" signal, combined with centering (−PC1) to remove prompt-template anisotropy and a Boundary safety gate (Gate<0.15), enables detection that is simultaneously high-recall, low-FPR, attribution-capable, and computationally free (parasitic on agent prefill).

## Stable Constraints
- Method name: SkillProbe v4
- Core mechanism: oper+bnd geometric features + centering(−PC1) + Boundary gating
- Training: SMP dev (360 samples), L7+ layer selection, 5-fold CV
- Evaluation: 6 datasets (SMP, MSB, MASB, Lexical Decoy, Inverse Decoy, SkillHarm)
- Baselines: 14 methods (surface-text, LLM, internal probes, our old versions)
- Key metrics: AUROC, FPR, FNR, F1, attribution accuracy, overhead (ms)

## Terminology (do not change)
- "Pre-Action Vigilance" — the mechanism insight
- "centering(−PC1)" — not "debiasing" or "normalization"
- "Boundary safety gate" — not "threshold override" or "boundary filter"
- "parasitic detection" — not "shared prefill" or "zero-overhead"
- "声明看起来有毒/实际无毒" — Lexical Decoy
- "声明看不出/实际有毒" — Inverse Decoy
