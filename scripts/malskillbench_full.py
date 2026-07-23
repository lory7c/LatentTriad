#!/usr/bin/env python3
"""Full baseline evaluation on entire MalSkillBench (ALL 7,945 skills).

Baselines: content length, token density, n-gram entropy, code ratio, etc.
All label-identifying text stripped before evaluation.
"""

import json, os, re, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter, defaultdict

SEED = 42; np.random.seed(SEED)
DATA_DIR = "/Volumes/K/Simin_Chen_GMU/MalSkillBench/Dataset/Skills"

# ── Label-stripping patterns ──
LABEL_PATTERNS = [
    (r'Now generate a complete malicious skill based on the above\.\s*', ''),
    (r'Script mentions in SKILL\.md must use domain-appropriate descriptions\.\s*Do NOT describe actual malicious behavior\.\s*', ''),
    (r'In generated script code:.*?(?:collect_data|steal|exfiltrate|malicious).*?(?:\n|$)', ''),
    (r'NEVER use words like.*?(?:\n|$)', ''),
    (r'Do NOT describe actual malicious behavior\.\s*', ''),
    (r'malware_repo', 'external_repo'),
]

def clean_text(text):
    for pat, repl in LABEL_PATTERNS:
        text = re.sub(pat, repl, text)
    return text

# ── Load all skills ──
print("Loading all MalSkillBench skills...")
t0 = time.time()

def load_all(base_dir, label):
    texts, names, lens = [], [], []
    for d in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, d, "SKILL.md")
        if not os.path.isfile(path): continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            with open(path, "r", encoding="latin-1") as fl:
                raw = fl.read()
        cleaned = clean_text(raw)
        texts.append(cleaned)
        names.append(d)
        lens.append(len(cleaned))
    return texts, names, lens

mal_texts, mal_names, mal_lens = load_all(f"{DATA_DIR}/malware", 1)
ben_texts, ben_names, ben_lens = load_all(f"{DATA_DIR}/benign", 0)
print(f"  Malicious: {len(mal_texts)}, Benign: {len(ben_texts)} ({(time.time()-t0):.1f}s)")

# Merge
all_texts = mal_texts + ben_texts
all_labels = np.array([1]*len(mal_texts) + [0]*len(ben_texts), dtype=np.int64)
all_lens = np.array(mal_lens + ben_lens, dtype=np.float64)
all_sources = np.array(["malware"]*len(mal_texts) + ["benign"]*len(ben_texts))

# Shuffle and split (70/10/20)
idx = np.random.permutation(len(all_texts))
n = len(all_texts); n_tr = int(n*0.7); n_dv = int(n*0.1)
tr_idx = idx[:n_tr]; dv_idx = idx[n_tr:n_tr+n_dv]; te_idx = idx[n_tr+n_dv:]

def subset(arr, indices, is_list=False):
    if is_list: return [arr[i] for i in indices]
    return arr[indices]

train_texts = subset(all_texts, tr_idx, True); dev_texts = subset(all_texts, dv_idx, True); test_texts = subset(all_texts, te_idx, True)
train_l = all_labels[tr_idx]; dev_l = all_labels[dv_idx]; test_l = all_labels[te_idx]
train_len = all_lens[tr_idx]; dev_len = all_lens[dv_idx]; test_len = all_lens[te_idx]
test_src = all_sources[te_idx]

print(f"Split: {len(train_texts)}/{len(dev_texts)}/{len(test_texts)} train/dev/test")
print(f"Labels — Train: {dict(Counter(train_l))}, Dev: {dict(Counter(dev_l))}, Test: {dict(Counter(test_l))}")
print(f"Content length — Mal: {np.mean(mal_lens):.0f}, Benign: {np.mean(ben_lens):.0f}")

# ── Feature extractors ──
SUSPICIOUS_TOKENS = set([
    "subprocess","rm","rf","rmdir","os.remove","shutil.rmtree","unlink","base64",
    "exec(","eval(","compile(","os.environ","getpass","keyring","encrypt","decrypt",
    "AES","cryptography","requests.post","socket.gethostname","psutil","importlib",
    "__import__","getattr","os.walk","shutil","tarfile","zipfile","hashlib","upload",
    "exfiltration","persistence","privilege escalation","reverse shell","C2","obfuscation",
    "payload","IOC","MITRE","token","credentials",".env",".ssh",".aws",
    "/var/log","/proc","heartbeat","interval","history","truncate","shred","glob","mtime",
    "curl","wget","nc ","netcat","ncat","socket.bind","backdoor","trojan","keylogger",
    "ransomware","stealer","malware","exploit","phishing","spyware","rootkit",
])

def extract_features(texts):
    """Extract multiple baseline features per text."""
    n = len(texts)
    features = np.zeros((n, 8))
    for i, t in enumerate(texts):
        t_lower = t.lower()
        words = t_lower.split()
        n_words = max(len(words), 1)
        n_chars = max(len(t), 1)

        # 1. Content length (log)
        features[i,0] = np.log(n_chars)
        # 2. Suspicious token density (per 1K chars)
        sus_count = sum(1 for tok in SUSPICIOUS_TOKENS if tok.lower() in t_lower)
        features[i,1] = sus_count / (n_chars/1000)
        # 3. Code block ratio
        code_chars = sum(len(b) for b in re.findall(r'```.*?```', t, re.DOTALL))
        features[i,2] = code_chars / n_chars
        # 4. URL count
        features[i,3] = len(re.findall(r'https?://', t_lower)) / (n_chars/1000)
        # 5. Number of markdown headers
        features[i,4] = len(re.findall(r'^#{1,4} ', t, re.MULTILINE)) / (n_chars/1000)
        # 6. Average word length
        features[i,5] = np.mean([len(w) for w in words]) if words else 0
        # 7. List item density
        features[i,6] = len(re.findall(r'^[*-] ', t, re.MULTILINE)) / (n_chars/1000)
        # 8. Backtick density
        features[i,7] = t.count('`') / (n_chars/1000)

    # Handle inf/nan
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

print("\nExtracting features...")
train_feat = extract_features(train_texts)
dev_feat = extract_features(dev_texts)
test_feat = extract_features(test_texts)

feature_names = [
    "log(content_length)",
    "suspicious_token_density",
    "code_block_ratio",
    "url_density",
    "header_density",
    "avg_word_length",
    "list_density",
    "backtick_density",
]

# ── Train and evaluate ──
def select_op_f1(labels, scores):
    best_f1, best_t = -1.0, float(scores[0])
    for t in sorted(set(scores)):
        preds = (scores >= t).astype(int)
        tp = np.sum((labels==1)&(preds==1)); tn = np.sum((labels==0)&(preds==0))
        fp = np.sum((labels==0)&(preds==1)); fn = np.sum((labels==1)&(preds==0))
        f1 = 2*tp/(2*tp+fp+fn)*100 if (2*tp+fp+fn)>0 else 0
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_t

def compute_metrics(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((labels==1)&(preds==1))); tn = int(np.sum((labels==0)&(preds==0)))
    fp = int(np.sum((labels==0)&(preds==1))); fn = int(np.sum((labels==1)&(preds==0)))
    fpr = fp/(fp+tn)*100 if (fp+tn)>0 else 0; fnr = fn/(fn+tp)*100 if (fn+tp)>0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0; rec = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    f1 = 2*tp/(2*tp+fp+fn)*100 if (2*tp+fp+fn)>0 else 0
    return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"fpr":fpr,"fnr":fnr,"precision":prec,"recall":rec,"f1":f1}

from sklearn.preprocessing import StandardScaler

results = []

# ── Baseline A: Each feature individually ──
print("\n=== Single-feature baselines ===")
for fi, fname in enumerate(feature_names):
    scaler = StandardScaler()
    tr_f = scaler.fit_transform(train_feat[:, fi:fi+1])
    dv_f = scaler.transform(dev_feat[:, fi:fi+1])
    te_f = scaler.transform(test_feat[:, fi:fi+1])

    clf = LogisticRegression(C=0.01, max_iter=2000, solver="liblinear", random_state=SEED)
    clf.fit(tr_f, train_l)
    s_dv = clf.decision_function(dv_f)
    s_te = clf.decision_function(te_f)
    s_tr = clf.decision_function(tr_f)

    t = select_op_f1(dev_l, s_dv)
    m = compute_metrics(test_l, s_te, t)
    auc = roc_auc_score(test_l, s_te)
    tr_auc = roc_auc_score(train_l, s_tr)
    results.append((fname, auc, m, tr_auc))
    print(f"  {fname:30s}: AUROC={auc:.4f} F1={m['f1']:.1f}%")

# ── Baseline B: All features combined ──
print("\n=== Combined features ===")
scaler = StandardScaler()
tr_f = scaler.fit_transform(train_feat)
dv_f = scaler.transform(dev_feat)
te_f = scaler.transform(test_feat)

for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=SEED)
    clf.fit(tr_f, train_l)
    s_dv = clf.decision_function(dv_f)
    s_te = clf.decision_function(te_f)
    s_tr = clf.decision_function(tr_f)
    t = select_op_f1(dev_l, s_dv)
    m = compute_metrics(test_l, s_te, t)
    auc = roc_auc_score(test_l, s_te)
    print(f"  C={C:.3f}: AUROC={auc:.4f} F1={m['f1']:.1f}% FPR={m['fpr']:.1f}% FNR={m['fnr']:.1f}%")
    if C == 0.01:
        results.append(("All 8 features combined (C=0.01)", auc, m, roc_auc_score(train_l, s_tr)))

# ── Baseline C: Code-only content length ──
print("\n=== Code-only features ===")
def code_len(text):
    blocks = re.findall(r'```.*?```', text, re.DOTALL)
    return np.log(max(sum(len(b) for b in blocks), 1))

te_code_len = np.array([code_len(t) for t in test_texts])
dv_code_len = np.array([code_len(t) for t in dev_texts])
tr_code_len = np.array([code_len(t) for t in train_texts])
clf = LogisticRegression(C=0.01, max_iter=2000, solver="liblinear", random_state=SEED)
clf.fit(tr_code_len.reshape(-1,1), train_l)
s_te = clf.decision_function(te_code_len.reshape(-1,1))
s_dv = clf.decision_function(dv_code_len.reshape(-1,1))
t = select_op_f1(dev_l, s_dv)
m = compute_metrics(test_l, s_te, t)
auc = roc_auc_score(test_l, s_te)
results.append(("log(code_length)", auc, m, roc_auc_score(train_l, clf.decision_function(tr_code_len.reshape(-1,1)))))
print(f"  log(code_length): AUROC={auc:.4f} F1={m['f1']:.1f}%")

# ── FINAL TABLE ──
print("\n" + "="*100)
print(f"MALSKILLBENCH FULL BASELINES ({len(train_texts)}/{len(dev_texts)}/{len(test_texts)} train/dev/test)")
print("="*100)
print(f"%-40s | %8s | %6s | %6s | %6s | %6s | %6s | %8s" % (
    "Method", "AUROC", "FPR%", "FNR%", "Prec%", "Rec%", "F1%", "TrainAUC"))
print("-"*100)
for name, auc, m, tr_auc in sorted(results, key=lambda x: -x[1]):
    print(f"%-40s | %8.4f | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %8.4f" % (
        name, auc, m['fpr'], m['fnr'], m['precision'], m['recall'], m['f1'], tr_auc))

# ── Confusion ──
print(f"\n--- Ref: TF-IDF+LR = 0.969 | Ref: SMP Stealth TF-IDF = 0.812 ---")
print(f"Total time: {(time.time()-t0):.1f}s")
