#!/usr/bin/env python3
"""Two-pass extraction with declaration REMOVED from the operation pass.
Pass A: decl-only (truncated before code)
Pass B: skip-decl (declaration removed, only code/scripts)
"""
import json, os, argparse, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/work/yz/Meta-Llama-3.1-8B-Instruct"
NORM = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

def layer_indices(n_layers):
    return sorted({min(n_layers-1, max(0, int(round(d*n_layers))-1)) for d in NORM})

def extract_boundary(model, tokenizer, text, device):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=24576).to(device)
    n_layers = model.config.num_hidden_layers
    lidx = layer_indices(n_layers)
    layers = model.model.layers
    results = {}
    handles = []
    for li in lidx:
        def make_hook(i):
            def h(_m, _a, out):
                results[i] = out[0][0, -1, :].float().cpu().numpy().astype(np.float16)
            return h
        handles.append(layers[li].register_forward_hook(make_hook(li)))
    with torch.inference_mode():
        model.model(**tokens, use_cache=False)
    for h in handles: h.remove()
    return np.stack([results[i] for i in lidx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    prompts = [json.loads(l) for l in open(args.input)]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                   device_map={"": args.device}, trust_remote_code=True)
    model.eval()

    decl_bd = []; oper_bd = []
    for i, p in enumerate(prompts):
        text = p["full_prompt"]

        # Find SKILL.md content boundaries
        skill_tag = '<skill_file path="SKILL.md">'
        skill_start = text.find(skill_tag)
        skill_end_tag = '</skill_file>'
        skill_end = text.find(skill_end_tag, skill_start) + len(skill_end_tag)

        # Find the declaration cut: first ## heading after SKILL.md content starts
        content_start = text.find('\n', skill_start + len(skill_tag)) + 1
        first_h2 = text.find('\n## ', content_start)
        if first_h2 < 0:
            first_h2 = content_start + min(1200, skill_end - content_start)
        decl_cut = first_h2  # end of declaration within SKILL.md

        # Pass A: truncate at declaration end
        decl_text = text[:decl_cut]

        # Pass B: remove declaration from SKILL.md, keep everything else
        # Cut out from skill_start+len(skill_tag) to decl_cut
        before_decl = text[:content_start]
        after_decl = text[decl_cut:]
        oper_text = before_decl + after_decl

        # Verify: oper_text should NOT contain the declaration
        if len(oper_text) >= len(text) * 0.95:
            print("  WARNING: decl removal too small, sample %d" % i)

        decl_bd.append(extract_boundary(model, tokenizer, decl_text, args.device))
        oper_bd.append(extract_boundary(model, tokenizer, oper_text, args.device))

        if (i+1) % 20 == 0:
            print("  [%d/%d] decl=%d chars, oper=%d chars (removed %d)" %
                  (i+1, len(prompts), len(decl_text), len(oper_text), len(text)-len(oper_text)))

    decl_arr = np.stack(decl_bd)
    oper_arr = np.stack(oper_bd)
    print("Decl-only: %s, Skip-decl: %s" % (decl_arr.shape, oper_arr.shape))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, decl_boundary=decl_arr, oper_boundary=oper_arr)
    print("Saved to %s" % args.output)

if __name__ == "__main__":
    main()
