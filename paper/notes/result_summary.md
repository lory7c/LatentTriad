# Result Summary — Figures & Tables Anchor

## Supported Findings (strong)
- SMP: v4 AUROC=0.947, vs RouteGuard +12.3pp, vs Boundary +8.4pp, FNR 6.5%
- Lexical Decoy: v4 FPR=0% (TF-IDF FPR=100%)
- Inverse Decoy: v4 FNR=3% (Boundary FNR=100%)
- Centering ablation: +43-59pp cross-distribution
- Gate<0.15: Lexical Decoy FPR 100%→0%, InvDecoy FPR 100%→0%
- Attribution: 175/175=100%
- Overhead: 18.7ms, 7% faster than Boundary
- SkillHarm: v3 AUC=0.973 Recall=100%, Boundary AUC=0

## Weak Findings (need caveat)
- MSB: v4 AUC=0.996 — source confound inflates all methods
- MASB: v4 AUC=0.981 — in-domain evaluation
- RouteGuard Decoy: cross-extract prevents fair eval
- SkillHarm v4 gate fails cross-dist (Boundary scores collapse) — needs in-domain Boundary training

## Figure Anchors
- Fig 3 (norm_ratio decay): L11 best, SMD=−1.42, L3 excluded
- Fig 5 (2×2 matrix): four quadrants, v4 only one solving all
- Fig 8 (overhead): parasitic detection 18.7ms vs others

## Table Anchors
- Table 2 (SMP): 9 baselines, v4 best on all metrics
- Table 4 (2×2): Lexical Decoy FPR + Inverse Decoy FNR
- Table 7 (ablation): v1→v2→v3→v4, centering is key
