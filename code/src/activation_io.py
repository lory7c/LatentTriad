"""Resolve the canonical and split-subdirectory activation bundle layouts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def activation_path(root: Path, split_name: str, layer: int) -> Path:
    filename = f"{split_name}_layer_{layer}.npy"
    candidates = (root / filename, root / split_name / filename)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing activation file: {candidates}")


def discover_layers(root: Path, split_name: str = "train") -> list[int]:
    pattern = re.compile(rf"{re.escape(split_name)}_layer_(\d+)\.npy")
    layers = []
    for directory in (root, root / split_name):
        if not directory.is_dir():
            continue
        for path in directory.glob(f"{split_name}_layer_*.npy"):
            match = pattern.fullmatch(path.name)
            if match:
                layers.append(int(match.group(1)))
    return sorted(set(layers))


def load_metadata(root: Path, split_name: str) -> list[dict]:
    candidates = (
        root / f"{split_name}_meta.json",
        root / split_name / f"{split_name}_meta.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    if split_name == "train":
        parts = sorted(root.glob("train_p*_meta.json"))
        if not parts and (root / "train").is_dir():
            parts = sorted((root / "train").glob("train_p*_meta.json"))
        if parts:
            return [row for path in parts for row in json.loads(path.read_text())]
    raise FileNotFoundError(f"Missing metadata for split {split_name}: {candidates}")


def label_path(root: Path, split_name: str) -> Path:
    filenames = (f"y_{split_name}.npy", f"{split_name}_labels.npy")
    for directory in (root, root / split_name):
        for filename in filenames:
            path = directory / filename
            if path.is_file():
                return path
    raise FileNotFoundError(f"Missing labels for split {split_name} in {root}")


def load_labels(root: Path, split_name: str, mmap_mode: str | None = None) -> np.ndarray:
    try:
        return np.load(label_path(root, split_name), mmap_mode=mmap_mode)
    except FileNotFoundError:
        if split_name != "train":
            raise
        parts = sorted(root.glob("train_p*_labels.npy"))
        if not parts and (root / "train").is_dir():
            parts = sorted((root / "train").glob("train_p*_labels.npy"))
        if not parts:
            raise
        return np.concatenate([np.load(path) for path in parts])
