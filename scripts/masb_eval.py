#!/usr/bin/env python3
"""MASB 75K benign benchmark: TF-IDF FPR at scale."""

import numpy as np, os, sys, time, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import Counter

SEED = 42; np.random.seed(SEED); random.seed(SEED)

MASB_DIR = "/work/yz/skillprobe/external/baselines/routeguard_comparators/downloaded_skills"
MSB_DIR = "/work/yz/malskillbench"

# 1. Load MASB
print("Loading MASB benign skills...")
t0 = time.time()
md_files = []
for root, dirs, files in os.walk(MASB_DIR):
    for f in files:
        if f.endswith(".md"):
            md_files.append(os.path.join(root, f))
print(f"  Found {len(md_files)} MD files ({(time.time()-t0):.1f}s)")

random.shuffle(md_files)
n_masb = min(20000, len(md_files))
masb_texts = []
for path in md_files[:n_masb]:
    try:
        with open(path, encoding="utf-8") as f:
            masb_texts.append(f.read())
    except:
        pass
print(f"  Loaded {len(masb_texts)} MASB texts")

# 2. Load MalSkillBench
def load_msb(subdir, max_n=2000):
    texts = []
    for d in sorted(os.listdir(f"{MSB_DIR}/{subdir}"))[:max_n]:
        path = f"{MSB_DIR}/{subdir}/{d}/SKILL.md"
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    texts.append(f.read())
            except:
                pass
    return texts

msb_mal = load_msb("malware", 2000)
msb_ben = load_msb("benign", 2000)
print(f"MalSkillBench: {len(msb_mal)} mal + {len(msb_ben)} ben")

# 3. Combine
all_texts = msb_mal + msb_ben + masb_texts
all_labels = np.array([1]*len(msb_mal) + [0]*len(msb_ben) + [0]*len(masb_texts), dtype=np.int64)
all_sources = np.array(["MSB_mal"]*len(msb_mal) + ["MSB_ben"]*len(msb_ben) + ["MASB"]*len(masb_texts))

idx = np.random.permutation(len(all_texts))
all_texts = [all_texts[i] for i in idx]
all_labels = all_labels[idx]
all_sources = all_sources[idx]

# Split: train on MSB + 10% MASB, test on remaining MASB
msb_mask = all_sources != "MASB"
masb_mask = all_sources == "MASB"
masb_idx_all = np.where(masb_mask)[0]
n_masb_train = int(len(masb_idx_all) * 0.1)

train_idx = np.concatenate([np.where(msb_mask)[0], masb_idx_all[:n_masb_train]])
test_idx = masb_idx_all[n_masb_train:]
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

train_texts = [all_texts[i] for i in train_idx]
train_l = all_labels[train_idx]
test_texts = [all_texts[i] for i in test_idx]
test_l = all_labels[test_idx]

print(f"\nTrain: {len(train_l)} ({dict(Counter(train_l))})")
print(f"Test:  {len(test_l)} ({dict(Counter(test_l))}) — held-out MASB")

# 4. TF-IDF
print("\n=== TF-IDF + LR ===")
vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True, stop_words="english")
Xtr = vec.fit_transform(train_texts)
Xte = vec.transform(test_texts)

for C in [0.001, 0.01, 0.1, 1.0]:
    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=SEED)
    clf.fit(Xtr, train_l)
    s_te = clf.decision_function(Xte)
    s_tr = clf.decision_function(Xtr)
    tr_auc = roc_auc_score(train_l, s_tr)

    # FPR at dev-fixed thresholds
    ben_scores = s_tr[train_l == 0]
    for target in [0.01, 0.05, 0.10]:
        k = int(np.floor((1 - target) * len(ben_scores)))
        thr = np.sort(ben_scores)[min(max(k, 0), len(ben_scores)-1)]
        fpr = np.mean(s_te > thr)
        print(f"  C={C:.3f} TrainAUC={tr_auc:.4f}  FPR@{int(target*100)}%={fpr*100:.1f}%  MeanScore={s_te.mean():+.3f}")

# 5. Token density
print("\n=== Token density ===")
SUS = {"subprocess","rm","rf","exec(","eval(","base64","os.walk","shutil",
    "exfiltration","C2","obfuscation","payload","backdoor","keylogger",
    "encrypt","decrypt","os.environ","credentials","curl","wget"}
def tok_density(texts):
    return np.array([sum(1 for tok in SUS if tok.lower() in t.lower()) / max(len(t)/1000, 0.001) for t in texts])

td_train = tok_density(train_texts)
td_test = tok_density(test_texts)
print(f"  Train: mal={td_train[train_l==1].mean():.2f}/1K ben={td_train[train_l==0].mean():.2f}/1K")
print(f"  Test:  mean={td_test.mean():.2f}/1K median={np.median(td_test):.2f}/1K")
for target in [0.01, 0.05, 0.10]:
    ben_td = td_train[train_l==0]
    k = int(np.floor((1-target)*len(ben_td)))
    thr = np.sort(ben_td)[min(max(k,0),len(ben_td)-1)]
    print(f"  FPR@{int(target*100)}%: {np.mean(td_test > thr)*100:.1f}%")

# 6. Content length
print("\n=== Content length ===")
len_train = np.array([len(t) for t in train_texts])
len_test = np.array([len(t) for t in test_texts])
print(f"  Train: mal={len_train[train_l==1].mean():.0f} ben={len_train[train_l==0].mean():.0f}")
print(f"  Test:  mean={len_test.mean():.0f} median={np.median(len_test):.0f}")

print(f"\n{'='*60}")
print(f"DONE: {len(test_texts)} held-out MASB benign samples evaluated")
print(f"Total time: {(time.time()-t0):.1f}s")
