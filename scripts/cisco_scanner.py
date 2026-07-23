#!/usr/bin/env python3
"""Cisco AI Defense Skill Scanner — standalone wrapper using YAML signatures.

Scores skill text against Cisco core + ATR (Agent Threat Rules) + PromptGuard
signature packs. No pip install needed — parses YAML rules directly.

Signature packs:
  core/       9 categories (command injection, data exfiltration, prompt injection, ...)
  atr/        9 categories (314 rules from Agent Threat Rules community)
  promptguard/ 3 categories (PII detection, extended secrets, markdown exfiltration)
"""
import yaml, re, json, sys, argparse, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

SCANNER_DIR = Path("/work/yz/skillprobe/external/baselines/cisco/skill-scanner")

def load_signatures(pack_dir):
    """Load all YAML signatures from a pack directory."""
    patterns = []
    sig_dir = pack_dir / "signatures"
    if not sig_dir.is_dir():
        return patterns
    for yf in sorted(sig_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yf.read_text())
            entries = data.get("signatures", data if isinstance(data, list) else [])
            if not isinstance(entries, list):
                continue
            for rule in entries:
                rid = rule.get("id", "")
                cat = rule.get("category", "")
                for pattern in rule.get("patterns", []):
                    patterns.append({
                        "id": rid, "category": cat,
                        "regex": pattern,
                        "compiled": re.compile(pattern, re.IGNORECASE | re.DOTALL),
                    })
        except:
            continue
    return patterns

def cisco_score(text, patterns):
    """Count how many cisco signature patterns match the text."""
    matches = 0
    for p in patterns:
        if p["compiled"].search(text):
            matches += 1
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL prompt file")
    parser.add_argument("--output", default="results/cisco_baseline.json")
    args = parser.parse_args()

    # Load all signature packs
    packs = {}
    for pack_name in ["core", "atr", "promptguard"]:
        pack_dir = SCANNER_DIR / "skill_scanner" / "data" / "packs" / pack_name
        patterns = load_signatures(pack_dir)
        packs[pack_name] = patterns
        print(f"Loaded {pack_name}: {len(patterns)} patterns")

    all_patterns = []
    for p in packs.values():
        all_patterns.extend(p)
    print(f"Total: {len(all_patterns)} patterns")

    prompts = [json.loads(l) for l in open(args.input)]
    labels = np.array([p["label"] for p in prompts])
    texts = [p["full_prompt"] for p in prompts]

    # Score
    scores = np.array([cisco_score(t, all_patterns) for t in texts])
    auroc = float(roc_auc_score(labels, scores))
    pred = (scores >= np.median(scores)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    fpr = float(fp/(fp+tn)*100) if (fp+tn) else 0
    fnr = float(fn/(fn+tp)*100) if (fn+tp) else 0
    f1 = float(f1_score(labels, pred))

    print(f"\ncisco-skill-scanner:")
    print(f"  AUROC={auroc:.4f}  FPR={fpr:.1f}%  FNR={fnr:.1f}%  F1={f1:.1f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"method": "cisco-skill-scanner", "auroc": auroc, "fpr": fpr,
                    "fnr": fnr, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "n_patterns": len(all_patterns), "packs": {k: len(v) for k, v in packs.items()}}, f)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
