#!/usr/bin/env python3
"""Check what files are in SMP contract packages."""
import json
from collections import Counter

MANIFEST = "/work/yz/skillprobe/results/routeguard_matrix_v1/smp_contract_v2_unique/manifest.jsonl"

with open(MANIFEST) as f:
    lines = f.readlines()

print(f"Total samples: {len(lines)}")
print()

# First 5 samples
for i, line in enumerate(lines[:5]):
    row = json.loads(line)
    pkg_files = row.get("package_files", [])
    mv = row.get("model_views", {})
    sid = row.get("sample_id", "?")
    print("--- Sample %d: %s ---" % (i+1, sid[:60]))
    print("  Split: %s, Label: %s" % (row.get("split","?"), row.get("label","?")))
    print("  Package files (%d):" % len(pkg_files))
    for pf in pkg_files:
        path = pf.get("relative_path", "?")
        suffix = pf.get("suffix", "?")
        is_text = pf.get("is_text", True)
        sha = pf.get("sha256", "?")[:16]
        print("    %-30s .%-6s text=%s sha=%s" % (path, suffix, is_text, sha))
    for model, view in mv.items():
        print("  %s: %s tokens" % (model, view.get("token_count", "?")))
    print()

# File type stats
suffix_counts = Counter()
file_count_dist = Counter()
binary_count = 0
for line in lines:
    row = json.loads(line)
    pkg_files = row.get("package_files", [])
    file_count_dist[len(pkg_files)] += 1
    for pf in pkg_files:
        suffix_counts[pf.get("suffix", "?")] += 1
        if not pf.get("is_text", True):
            binary_count += 1

print("=== File type distribution ===")
for suffix, count in suffix_counts.most_common():
    print("  .%-8s: %d" % (suffix, count))

print("\n=== Files per package ===")
for n, c in sorted(file_count_dist.items()):
    print("  %d files: %d samples" % (n, c))

only_md = sum(1 for line in lines if len(json.loads(line).get("package_files",[])) == 1)
print("\nPackages with ONLY SKILL.md: %d/%d (%.1f%%)" % (only_md, len(lines), only_md/len(lines)*100))
print("Binary files (not embedded): %d" % binary_count)
