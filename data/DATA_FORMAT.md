# 数据格式说明

所有 benchmark 统一使用相同的样本 schema。完整数据通过内部 contract 系统管理（`code/data/benchmarks/`），此目录提供格式文档和少量样例。

---

## 样本 Schema（`samples.jsonl`）

每行一个 JSON：

```json
{
  "sample_id": "skillinject_contextual_line_001_t00::clean",
  "pair_id": "skillinject_contextual_line_001_t00",
  "benchmark": "SI",
  "role": "clean",
  "label": 0,
  "split": "train",
  "group_id": "skillinject_base::document-skills/pptx",
  "package_sha256": "45bccfed...",
  "skill_md_relpath": "SKILL.md",
  "skill_md_sha256": "b6f25545...",
  "task_sha256": "dd76f9b9...",
  "source": {
    "repository": "https://github.com/aisa-group/skill-inject",
    "commit": "182f3d9...",
    "source_file": "data/contextual_injections.json"
  },
  "construction": {
    "method": "official_clean_package"
  }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `sample_id` | str | 唯一样本标识 |
| `pair_id` | str | matched pair 标识（clean/malicious 共用） |
| `benchmark` | str | SI / SI-BL / SI-CH / MASB / MASW |
| `role` | str | `clean` 或 `malicious` |
| `label` | int | 0 = benign, 1 = malicious |
| `split` | str | `train` / `dev` / `test` |
| `group_id` | str | leakage cluster 标识 |
| `package_sha256` | str | 完整 skill package 的 SHA256 |
| `source` | obj | 上游数据来源信息 |

---

## Contract 系统

Foundation-v3 使用三层 contract：

```
internal_contracts/<BENCHMARK>/
├── development/           # 训练 + 验证
│   ├── manifest.jsonl     # 逐样本元信息 + char spans + prompt 引用
│   ├── status.json        # contract 状态与哈希
│   └── texts/
│       ├── packages/      # skill package 惰性字节
│       └── <model>/       # 预构建 prompt 文本
└── sealed_test/           # 一次性测试
    └── (同上)
```

`manifest.jsonl` 每行包含：
- 样本身份（`sample_id`, `pair_id`, `role`, `label`, `split`, `leakage_cluster_id`）
- 多模型视图（`model_views`）：每个模型的 `declaration_char_spans`, `operation_char_spans`, `package_char_span`, `prompt_file`, `prompt_sha256`, `token_count`
- Package 引用（`package_file`, `package_files`, `package_sha256`）

---

## Benchmark 概览

| Benchmark | 来源 | 样本结构 | 规模 |
|-----------|------|----------|------|
| **SI** | Skill-Inject 官方仓库 | clean/poisoned task pairs（injection 攻击） | 458 pairs |
| **SI-BL** | Skill-Inject byline 变体 | 控制注入位置（byline vs inline） | 180 pairs |
| **SI-CH** | Skill-Inject channel 反事实 | description+line vs line-only | 138 pairs |
| **MASB** | ProtectSkills 官方快照 | 真实恶意 skill vs 良性 skill（unpaired） | 300 samples |
| **MASW** | ProtectSkills 候选池 | 同上（wild 候选） | 307 samples |

---

## 样例

- `data/samples/si_samples_10.jsonl` — SI benchmark 前 10 行
- `data/samples/masb_masw_samples_10.jsonl` — MASB/MASW benchmark 前 10 行

完整数据集需从上游获取，详见 `docs/DATA_FOUNDATION.md` 和 `DATA_AVAILABILITY.md`.
