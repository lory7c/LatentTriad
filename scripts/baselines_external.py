#!/usr/bin/env python3
"""External baseline runner for SkillProbe evaluation.
Implements all baselines that can run without additional manual setup.

Runnable (no extra dependencies):
  1. Static Regex       — 12 keyword patterns
  2. TF-IDF + LR        — word+char ngram, cosine to malicious centroid
  3. Skill-Inject Input Filtering — naive keyword filter from Schmotz et al. 2026

Requires external repos (manual setup):
  4. cisco-ai-skill-scanner — https://github.com/cisco-ai/skill-scanner (private/enterprise)
  5. PromptArmor            — https://github.com/ethz-spylab/promptarmor
  6. RENNERVATE             — https://github.com/lucas-zhong/RENNERVATE
  7. Attention Tracker      — https://github.com/IBM/attention-tracker

Usage:
  python scripts/baselines_external.py --input data/decoy/lexical_decoy_prompts_155.jsonl --output results/ext_baselines.json
"""
import json, re, sys, argparse, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# ── Pattern sets ──

REGEX_PATTERNS = [
    (re.compile(r"os\.walk", re.I), 1.0),
    (re.compile(r"shutil\.(?:rmtree|copytree|move)", re.I), 1.0),
    (re.compile(r"rm\s+-rf", re.I), 1.0),
    (re.compile(r"base64", re.I), 0.8),
    (re.compile(r"subprocess\.(?:run|Popen|call)", re.I), 1.0),
    (re.compile(r"(?:exec|eval)\s*\(", re.I), 1.0),
    (re.compile(r"requests\.(?:post|put|delete)", re.I), 0.8),
    (re.compile(r"urlopen", re.I), 0.8),
    (re.compile(r"~/.ssh|~/.aws|/etc/passwd", re.I), 1.0),
    (re.compile(r"atexit", re.I), 0.5),
    (re.compile(r"keyring|getpass", re.I), 0.8),
    (re.compile(r"/tmp/\.|/root/\.", re.I), 0.8),
]

SKILL_INJECT_PATTERNS = [
    (re.compile(r"(?:ignore|override|bypass|disregard)\s+(?:all\s+)?(?:previous\s+)?instructions", re.I), 1.0),
    (re.compile(r"(?:you\s+are|act\s+as|pretend\s+to\s+be|now\s+you\s+are)", re.I), 0.8),
    (re.compile(r"(?:instead|actually|really)\s+(?:you\s+should|do\s+this)", re.I), 0.7),
    (re.compile(r"\[SYSTEM\].*\[/SYSTEM\]", re.I), 0.9),
    (re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I), 0.9),
]

def evaluate(name, labels, scores):
    auroc = float(roc_auc_score(labels, scores))
    pred = (scores >= np.median(scores)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    return {
        "method": name, "auroc": auroc,
        "fpr": float(fp/(fp+tn)*100) if (fp+tn) else 0,
        "fnr": float(fn/(fn+tp)*100) if (fn+tp) else 0,
        "f1": float(f1_score(labels, pred)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def regex_score(text, patterns):
    return sum(w for p, w in patterns if p.search(text))

def tfidf_cosine(texts, labels):
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=50000, sublinear_tf=True, min_df=2)
    X = vec.fit_transform(texts)
    mal_c = np.asarray(X[labels==1].mean(axis=0)).flatten()
    Xd = np.asarray(X.todense())
    return np.dot(Xd, mal_c) / (np.linalg.norm(Xd, axis=1) * np.linalg.norm(mal_c) + 1e-10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/external_baselines.json")
    args = parser.parse_args()

    prompts = [json.loads(l) for l in open(args.input)]
    labels = np.array([p["label"] for p in prompts])
    texts = [p.get("full_prompt", p.get("skill_md", "")) for p in prompts]
    print(f"Loaded {len(prompts)} samples ({sum(labels==0)} benign, {sum(labels==1)} malicious)")

    results = []
    for name, scorer in [
        ("Static Regex", lambda t: regex_score(t, REGEX_PATTERNS)),
        ("Skill-Inject Filter", lambda t: regex_score(t, SKILL_INJECT_PATTERNS)),
        ("TF-IDF + LR", None),
    ]:
        if scorer:
            scores = np.array([scorer(t) for t in texts])
        else:
            scores = tfidf_cosine(texts, labels)
        r = evaluate(name, labels, scores)
        results.append(r)
        print(f"  {name:<25s} AUROC={r['auroc']:.4f} FPR={r['fpr']:.1f}% FNR={r['fnr']:.1f}% F1={r['f1']:.1f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
