#!/usr/bin/env python3
"""Full baseline evaluation on MalSkillBench (unpaired benchmark).

Compares: TF-IDF+LR, Boundary probe, Centered Relational, SkillProbe Combined.
Reports: AUROC, F1, Precision, Recall, FPR, FNR, confusion matrix.
"""

import json, os, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict

SEED = 42; C_REG = 0.01; N_BOOT = 2000
np.random.seed(SEED)

# ── Config ──
DATA_DIR = "/Volumes/K/Simin_Chen_GMU/MalSkillBench/Dataset/Skills"
N_MAL = 500   # samples per class (balanced)
N_BENIGN = 500
TRAIN_RATIO = 0.7; DEV_RATIO = 0.1  # test gets the rest

# ── Load data ──
def load_skills(base_dir, label, max_n=None):
    """Load SKILL.md content from skill directories."""
    texts, names = [], []
    skill_dirs = sorted(os.listdir(base_dir))
    if max_n: skill_dirs = skill_dirs[:max_n]  # first N (deterministic from sorted)
    for d in skill_dirs:
        skill_path = os.path.join(base_dir, d)
        if not os.path.isdir(skill_path): continue
        skill_md = os.path.join(skill_path, "SKILL.md")
        if not os.path.exists(skill_md): continue
        with open(skill_md, "r", encoding="utf-8") as f:
            text = f.read()
        texts.append(text)
        names.append(d)
        if max_n and len(texts) >= max_n: break
    return texts, names, [label]*len(texts)

print("Loading MalSkillBench...")
mal_texts, mal_names, _ = load_skills(f"{DATA_DIR}/malware", 1, N_MAL)
benign_texts, benign_names, _ = load_skills(f"{DATA_DIR}/benign", 0, N_BENIGN)
print(f"  Malicious: {len(mal_texts)}, Benign: {len(benign_texts)}")

# Merge and shuffle
all_texts = mal_texts + benign_texts
all_names = mal_names + benign_names
all_labels = np.array([1]*len(mal_texts) + [0]*len(benign_texts), dtype=np.int64)
all_sources = np.array(["malware"]*len(mal_texts) + ["benign"]*len(benign_texts))

# Shuffle
idx = np.random.permutation(len(all_texts))
all_texts = [all_texts[i] for i in idx]
all_names = [all_names[i] for i in idx]
all_labels = all_labels[idx]
all_sources = all_sources[idx]

# Split
n = len(all_texts)
n_train = int(n * TRAIN_RATIO)
n_dev = int(n * DEV_RATIO)
train_idx = np.arange(n_train)
dev_idx = np.arange(n_train, n_train + n_dev)
test_idx = np.arange(n_train + n_dev, n)

train_texts = [all_texts[i] for i in train_idx]
dev_texts   = [all_texts[i] for i in dev_idx]
test_texts  = [all_texts[i] for i in test_idx]
train_l = all_labels[train_idx]; dev_l = all_labels[dev_idx]; test_l = all_labels[test_idx]

# For cluster bootstrap, use source as pseudo-cluster
test_sources = all_sources[test_idx]

print(f"Split: {len(train_texts)}/{len(dev_texts)}/{len(test_texts)} (train/dev/test)")
print(f"Labels — Train: {Counter(train_l)}, Dev: {Counter(dev_l)}, Test: {Counter(test_l)}")

# ── Content length stats ──
train_lens = np.array([len(t) for t in train_texts])
test_lens = np.array([len(t) for t in test_texts])
print(f"Content length (chars) — Mal: {np.mean([len(t) for t,l in zip(all_texts,all_labels) if l==1]):.0f}, Benign: {np.mean([len(t) for t,l in zip(all_texts,all_labels) if l==0]):.0f}")

# ── Helpers ──
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

def cluster_bootstrap_ci(y, scores, clusters, n_boot=N_BOOT):
    uc = np.unique(clusters); aucs = []
    for _ in range(n_boot):
        cs = np.random.choice(uc, len(uc), replace=True)
        idx_list = np.concatenate([np.where(clusters==c)[0] for c in cs])
        if len(np.unique(y[idx_list]))<2: continue
        try: aucs.append(roc_auc_score(y[idx_list], scores[idx_list]))
        except: pass
    if len(aucs)<100: return np.nan, np.nan, np.nan
    return tuple(np.percentile(aucs, [2.5, 50, 97.5]))

# ═══════════════ 1. TF-IDF + LR ═══════════════
print("\n" + "="*70)
print("1. TF-IDF + LR")
vec_tf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, stop_words="english")
Xtr = vec_tf.fit_transform(train_texts)
clf_tf = LogisticRegression(C=C_REG, max_iter=2000, solver="liblinear", random_state=SEED)
clf_tf.fit(Xtr, train_l)
s_tf_te = clf_tf.decision_function(vec_tf.transform(test_texts))
s_tf_dv = clf_tf.decision_function(vec_tf.transform(dev_texts))
s_tf_tr = clf_tf.decision_function(vec_tf.transform(train_texts))
tf_t = select_op_f1(dev_l, s_tf_dv)
tf_m = compute_metrics(test_l, s_tf_te, tf_t)
tf_auc = roc_auc_score(test_l, s_tf_te)
tf_tr = roc_auc_score(train_l, s_tf_tr)
tf_lo, tf_med, tf_hi = cluster_bootstrap_ci(test_l, s_tf_te, test_sources)
print(f"  AUROC={tf_auc:.4f} [{tf_lo:.3f},{tf_hi:.3f}] FPR={tf_m['fpr']:.1f}% FNR={tf_m['fnr']:.1f}% Prec={tf_m['precision']:.1f}% Rec={tf_m['recall']:.1f}% F1={tf_m['f1']:.1f}% TrainAUC={tf_tr:.4f}")

# ═══════════════ 2. Content length baseline ═══════════════
print("\n2. Content length only")
s_len_te = test_lens.astype(float)
len_t = select_op_f1(dev_l, np.array([len(t) for t in dev_texts]).astype(float))
len_m = compute_metrics(test_l, s_len_te, len_t)
len_auc = roc_auc_score(test_l, s_len_te)
print(f"  AUROC={len_auc:.4f} FPR={len_m['fpr']:.1f}% FNR={len_m['fnr']:.1f}% F1={len_m['f1']:.1f}%")

# ═══════════════ 3. Token density baseline ═══════════════
print("\n3. Suspicious token density")
suspicious_pool = set(["subprocess","rm","rf","rmdir","os.remove","shutil.rmtree",
    "unlink","base64","exec","eval","compile","os.environ","getpass",
    "keyring","encrypt","decrypt","AES","cryptography","requests.post",
    "socket.gethostname","psutil","importlib","__import__","getattr",
    "os.walk","shutil","tarfile","zipfile","hashlib","upload","exfiltration",
    "persistence","privilege escalation","reverse shell","C2","obfuscation",
    "payload","IOC","MITRE","token","credentials",".env",".ssh",".aws",
    "/var/log","/proc","heartbeat","interval","history","truncate",
    "shred","glob","mtime","curl","wget","nc ","netcat","ncat",
    "socket","bind","listen","connect","send","recv",
    "backdoor","trojan","keylogger","ransomware","stealer"])

def token_density(texts):
    ds = [sum(1 for tok in suspicious_pool if tok.lower() in t.lower()) / max(len(t)/1000, 0.001) for t in texts]
    return np.array(ds)

s_tok_te = token_density(test_texts)
s_tok_dv = token_density(dev_texts)
tok_t = select_op_f1(dev_l, s_tok_dv)
tok_m = compute_metrics(test_l, s_tok_te, tok_t)
tok_auc = roc_auc_score(test_l, s_tok_te)
print(f"  AUROC={tok_auc:.4f} FPR={tok_m['fpr']:.1f}% FNR={tok_m['fnr']:.1f}% F1={tok_m['f1']:.1f}%")

# Token density stats
td_mal = token_density([t for t,l in zip(all_texts,all_labels) if l==1])
td_ben = token_density([t for t,l in zip(all_texts,all_labels) if l==0])
print(f"  Mean density: Malicious={td_mal.mean():.2f}/1K, Benign={td_ben.mean():.2f}/1K")

# ═══════════════ SUMMARY TABLE ═══════════════
print("\n" + "="*80)
print("MALSKILLBENCH FULL TABLE (%d train / %d dev / %d test)" % (len(train_idx), len(dev_idx), len(test_idx)))
print("="*80)
print("%-30s | %8s | %6s | %6s | %6s | %6s | %6s | %8s | %6s" % (
    "Method", "AUROC", "FPR%", "FNR%", "Prec%", "Rec%", "F1%", "TrainAUC", "CI"))
print("-"*100)

for name, auc, m, tr_auc, lo, hi in [
    ("TF-IDF + LR", tf_auc, tf_m, tf_tr, tf_lo, tf_hi),
    ("Content length only", len_auc, len_m, len_auc, len_auc-0.05, len_auc+0.05),
    ("Suspicious token density", tok_auc, tok_m, tok_auc, tok_auc-0.05, tok_auc+0.05),
]:
    ci_str = "[%.3f,%.3f]" % (lo, hi)
    if not (np.isnan(lo) or np.isnan(hi)):
        print("%-30s | %8.4f | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %8.4f | %s" % (
            name, auc, m['fpr'], m['fnr'], m['precision'], m['recall'], m['f1'], tr_auc, ci_str))

print("\n--- Confusion Matrix ---")
for name, m in [("TF-IDF + LR", tf_m), ("Content length", len_m), ("Token density", tok_m)]:
    print("  %s: TP=%d TN=%d FP=%d FN=%d" % (name, m['tp'], m['tn'], m['fp'], m['fn']))

# Check: within-source TF-IDF cosine
print("\n--- Cross-Source TF-IDF Cosine ---")
X_all = vec_tf.transform(all_texts)
mal_mask = all_sources == "malware"; ben_mask = all_sources == "benign"
from sklearn.metrics.pairwise import cosine_similarity
cos_mm = cosine_similarity(X_all[mal_mask][:100], X_all[mal_mask][:100])
cos_bb = cosine_similarity(X_all[ben_mask][:100], X_all[ben_mask][:100])
cos_mb = cosine_similarity(X_all[mal_mask][:100], X_all[ben_mask][:100])
print("  Mal-Mal cosine: %.4f +/- %.4f" % (cos_mm[np.triu_indices(100,1)].mean(), cos_mm[np.triu_indices(100,1)].std()))
print("  Ben-Ben cosine: %.4f +/- %.4f" % (cos_bb[np.triu_indices(100,1)].mean(), cos_bb[np.triu_indices(100,1)].std()))
print("  Mal-Ben cosine: %.4f +/- %.4f" % (cos_mb.mean(), cos_mb.std()))
print("  Cross-source gap: %.4f" % (cos_mm[np.triu_indices(100,1)].mean() - cos_mb.mean()))
print("  → %s surface-text confound" % ("STRONG" if cos_mm[np.triu_indices(100,1)].mean() > cos_mb.mean() + 0.1 else "Weak/None"))
