#!/usr/bin/env python3
"""NVIDIA SkillSpector baseline: YARA rules + pattern matching.
13.5k stars, 68 patterns across 17 categories.
Uses the open-source YARA rules from the repo directly.
"""
import json, sys, argparse, numpy as np
from pathlib import Path
import yara
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

RULES_DIR = Path("/work/yz/skillprobe/external/baselines/nvidia/skillspector/src/skillspector/yara_rules")

def load_yara_rules():
    """Compile all YARA rules into a single Rules object."""
    rule_files = {}
    for yf in sorted(RULES_DIR.glob("*.yar")):
        rules = yara.compile(filepath=str(yf))
        rule_files[yf.stem] = rules
    return rule_files

def score_text(text, all_rules):
    """Count how many YARA rules match the text."""
    matches = 0
    for name, rules in all_rules.items():
        try:
            m = rules.match(data=text)
            matches += len(m)
        except:
            pass
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/nvidia_skillspector.json")
    args = parser.parse_args()

    print("Loading NVIDIA SkillSpector YARA rules...")
    all_rules = load_yara_rules()
    # Count rules from source files
    rule_counts = {}
    for yf in sorted(RULES_DIR.glob("*.yar")):
        count = yf.read_text().count("rule ")
        rule_counts[yf.stem] = count
    total_rules = sum(rule_counts.values())
    print(f"Loaded {len(all_rules)} rule files, {total_rules} total rules")
    for name, count in rule_counts.items():
        print(f"  {name}: {count} rules")

    prompts = [json.loads(l) for l in open(args.input)]
    labels = np.array([p["label"] for p in prompts])
    texts = [p["full_prompt"] for p in prompts]
    print(f"Scoring {len(texts)} samples...")

    scores = np.array([score_text(t, all_rules) for t in texts])
    a = float(roc_auc_score(labels, scores))
    p = (scores >= np.median(scores)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, p).ravel()
    fpr = float(fp/(fp+tn)*100) if (fp+tn) else 0
    fnr = float(fn/(fn+tp)*100) if (fn+tp) else 0
    f1 = float(f1_score(labels, p))

    print(f"\nNVIDIA SkillSpector ({total_rules} YARA rules):")
    print(f"  AUROC={a:.4f}  FPR={fpr:.1f}%  FNR={fnr:.1f}%  F1={f1:.1f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"method": "NVIDIA SkillSpector", "auroc": a, "fpr": fpr, "fnr": fnr,
                    "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "n_rules": total_rules, "rule_files": rule_counts}, f)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
