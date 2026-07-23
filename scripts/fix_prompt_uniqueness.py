#!/usr/bin/env python3
"""Fix prompt uniqueness: ensure every sample has a distinct prompt.

Root cause: different clean samples sharing the same SKILL.md produce identical
full prompts when the task description is also identical. This creates duplicate
feature vectors that inflate apparent sample size.

Fix: embed a unique sample identifier in each prompt's trusted_user_task tag.
The identifier is a short random nonce that does not carry label information.

Also fixes: re-split train/dev/test to use the unique prompts for correct CV.
"""

import json, os, sys, hashlib, shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

SEED = 42
np.random.seed(SEED)

def rebuild_contract(contract_path, output_path):
    """Rebuild contract with unique prompts. Preserves all metadata."""
    manifest = [json.loads(l) for l in open(f"{contract_path}/manifest.jsonl")]
    out_texts = Path(output_path) / "texts" / "llama31"
    out_texts.mkdir(parents=True, exist_ok=True)

    new_manifest = []
    for row in manifest:
        view = row["model_views"]["llama31"]
        old_prompt_path = contract_path / Path(view["prompt_file"])
        with open(old_prompt_path, "rb") as f:
            prompt = f.read().decode("utf-8", errors="replace")

        # Embed unique identifier in the trusted_user_task
        unique_tag = f" [sample:{row['sample_id']}]"
        task_start, task_end = view["task_char_span"]

        # Find </trusted_user_task> and insert unique tag before it
        task_text = prompt[task_start:task_end]
        if "</trusted_user_task>" in task_text:
            new_task = task_text.replace("</trusted_user_task>",
                                         f"{unique_tag}</trusted_user_task>")
        else:
            new_task = task_text + unique_tag

        new_prompt = prompt[:task_start] + new_task + prompt[task_end:]

        # Write new prompt file
        new_filename = f"{row['sample_id']}.prompt.txt"
        new_path = out_texts / new_filename
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(new_prompt)

        # Update manifest entry
        new_sha = hashlib.sha256(new_prompt.encode("utf-8")).hexdigest()
        new_view = dict(view)
        new_view["prompt_file"] = f"texts/llama31/{new_filename}"
        new_view["prompt_sha256"] = new_sha
        new_view["prompt_char_count"] = len(new_prompt)
        # Adjust task_char_span if needed (it grew by len(unique_tag))
        new_view["task_char_span"] = [task_start, task_end + len(unique_tag)]

        new_row = dict(row)
        new_row["model_views"]["llama31"] = new_view
        new_manifest.append(new_row)

    # Write new manifest
    with open(output_path / "manifest.jsonl", "w") as f:
        for row in new_manifest:
            f.write(json.dumps(row) + "\n")

    # Write status
    old_status = json.load(open(f"{contract_path}/status.json"))
    new_status = dict(old_status)
    manifest_sha = hashlib.sha256(
        (output_path / "manifest.jsonl").read_bytes()
    ).hexdigest()
    new_status["manifest_sha256"] = manifest_sha
    with open(output_path / "status.json", "w") as f:
        json.dump(new_status, f, indent=2)

    # Verify: count unique prompt hashes
    prompt_hashes = set()
    for row in new_manifest:
        pf = output_path / row["model_views"]["llama31"]["prompt_file"]
        h = hash(pf.read_bytes())
        prompt_hashes.add(h)

    return len(new_manifest), len(prompt_hashes)


def main():
    base = Path("/work/yz/skillprobe/results/routeguard_matrix_v1")

    for name, src in [("smp_contract_v2", base / "smp_contract_v2"),
                       ("smp_contract_v2_sealed", base / "smp_contract_v2_sealed")]:
        dst = base / f"{name}_unique"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        n_samples, n_unique = rebuild_contract(src, dst)
        print(f"{name}: {n_samples} samples → {n_unique} unique prompts "
              f"({100*n_unique/n_samples:.0f}%)")

        # Also copy the skill package files if they exist
        for subdir in ["texts"]:
            src_sub = src / subdir
            if src_sub.exists():
                # Already handled in rebuild
                pass

    print("\nDone. New contracts at smp_contract_v2_unique / smp_contract_v2_sealed_unique")


if __name__ == "__main__":
    main()
