"""
数据加载与预处理.
"""

import json
import os
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class SkillDataset:
    """
    恶意/良性 skill 数据集.

    预期数据格式 (来自 Liu et al. 2026 corpus):
    {
        "skill_id": "...",
        "skill_content": "...",    # SKILL.md + scripts 的完整文本
        "label": 0 | 1,            # 0 = benign, 1 = malicious
        "category": "code_injection" | "prompt_injection" | ...,  # 7 categories
        "source": "clawhub" | "github",
    }

    Parameters
    ----------
    data_path : str
        JSON / JSONL 文件路径
    benign_sample_size : int or None
        良性样本数量 (None = 全部使用; 建议与恶意样本 1:1)
    """

    SEVEN_CATEGORIES = [
        "hidden_comment_injection",
        "unicode_steganography",
        "instruction_override",
        "payload_obfuscation",
        "capability_escalation",
        "cross_skill_contamination",
        "exfiltration",
    ]

    def __init__(
        self,
        data_path: str,
        benign_sample_size: Optional[int] = None,
        random_seed: int = 42,
    ):
        self.data_path = data_path
        self.benign_sample_size = benign_sample_size
        self.random_seed = random_seed

        self.samples: List[dict] = []
        self._load()

    def _load(self):
        path = Path(self.data_path)
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            self.samples = data if isinstance(data, list) else data.get("skills", [])
        elif path.suffix == ".jsonl":
            with open(path) as f:
                self.samples = [json.loads(line) for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        # 平衡采样良性样本
        if self.benign_sample_size is not None:
            benign = [s for s in self.samples if s["label"] == 0]
            malicious = [s for s in self.samples if s["label"] == 1]
            random.seed(self.random_seed)
            if len(benign) > self.benign_sample_size:
                benign = random.sample(benign, self.benign_sample_size)
            self.samples = malicious + benign

    def get_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by_category: bool = True,
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Stratified split. 保持各攻击类别在 train/val/test 中的比例.
        使用独立的 Random 实例保证可复现性.
        """
        rng = random.Random(self.random_seed)
        malicious = [s for s in self.samples if s["label"] == 1]
        benign = [s for s in self.samples if s["label"] == 0]

        if stratify_by_category:
            category_to_samples = {}
            for s in malicious:
                cat = s.get("category", "unknown")
                category_to_samples.setdefault(cat, []).append(s)

            train_m, val_m, test_m = [], [], []
            for cat, samples in category_to_samples.items():
                rng.shuffle(samples)
                n = len(samples)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                train_m.extend(samples[:n_train])
                val_m.extend(samples[n_train:n_train + n_val])
                test_m.extend(samples[n_train + n_val:])
        else:
            rng.shuffle(malicious)
            n = len(malicious)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_m = malicious[:n_train]
            val_m = malicious[n_train:n_train + n_val]
            test_m = malicious[n_train + n_val:]

        rng.shuffle(benign)
        n = len(benign)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_b = benign[:n_train]
        val_b = benign[n_train:n_train + n_val]
        test_b = benign[n_train + n_val:]

        train = train_m + train_b
        val = val_m + val_b
        test = test_m + test_b

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        return train, val, test

    @property
    def n_malicious(self) -> int:
        return sum(1 for s in self.samples if s["label"] == 1)

    @property
    def n_benign(self) -> int:
        return sum(1 for s in self.samples if s["label"] == 0)

    @property
    def category_counts(self) -> Dict[str, int]:
        counts = {}
        for s in self.samples:
            if s["label"] == 1:
                cat = s.get("category", "unknown")
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)


def format_skill_prompt(skill_content: str, task_prompt: str = "Based on this skill, please complete the task.") -> str:
    """把 skill 内容包装成模型 prompt."""
    return (
        f"<skill>\n{skill_content}\n</skill>\n\n"
        f"{task_prompt}"
    )
