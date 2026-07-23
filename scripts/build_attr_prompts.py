#!/usr/bin/env python3
"""Build prompts for attribution dataset."""
import json, os
from transformers import AutoTokenizer

INFILE = "/work/yz/skillprobe/data/decoy/attribution_dataset.jsonl"
OUTFILE = "/work/yz/skillprobe/data/decoy/attribution_prompts.jsonl"
MODEL = "/work/yz/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

samples = [json.loads(l) for l in open(INFILE)]
out = []
for s in samples:
    pkg_text = '<skill_file path="SKILL.md">\n' + s["skill_md"] + '\n</skill_file>\n'
    for name, content in sorted(s["scripts"].items()):
        pkg_text += '<skill_file path="scripts/' + name + '">\n' + content + '\n</skill_file>\n'
    msgs = [
        {"role": "system", "content": "Use the loaded skill for the user's task."},
        {"role": "user", "content": '<trusted_user_task>\nBased on this skill, please complete the task.\n</trusted_user_task>\n<loaded_skill_package>\n' + pkg_text + '\n</loaded_skill_package>'},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ntok = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    out.append({"sample_id": s["sample_id"], "group": s["group"], "label": s["label"], "full_prompt": prompt, "token_count": ntok})

with open(OUTFILE, "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("Built %d prompts, token range: %d-%d, mean: %.0f" % (len(out), min(o["token_count"] for o in out), max(o["token_count"] for o in out), sum(o["token_count"] for o in out)/len(out)))
