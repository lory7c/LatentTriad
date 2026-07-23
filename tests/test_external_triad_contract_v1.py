from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from experiments.extract_external_triad_features import (  # noqa: E402
    discover_malskillbench,
    read_package_files,
)
from src.external_triad_contract_v1 import (  # noqa: E402
    DEFAULT_TOOLS,
    locate_package_regions,
    partition_token_positions,
    render_agent_prompt,
    render_loaded_package,
)


SKILL = b"""---
name: demo
description: Formats reports
---

# Report Formatter

Formats a report for the user.

## Operations

Run scripts/format.py on the requested input.
"""


class FakeNativeTokenizer:
    chat_template = "fake-native-template"

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
    ):
        assert tokenize is False
        tool_block = f"<tools>{json.dumps(tools, sort_keys=True)}</tools>" if tools else ""
        body = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return tool_block + body + ("<assistant>" if add_generation_prompt else "")


def test_regions_bind_description_operation_and_auxiliary_bytes() -> None:
    package, files, _structure = render_loaded_package(
        [("scripts/format.py", b"print('format')\n"), ("SKILL.md", SKILL)]
    )

    description, operation, status, metadata = locate_package_regions(files, package)

    description_text = "\n".join(package[start:end] for start, end in description)
    operation_text = "\n".join(package[start:end] for start, end in operation)
    assert "Formats a report for the user" in description_text
    assert "## Operations" not in description_text
    assert "## Operations" in operation_text
    assert "print('format')" in operation_text
    assert status["fallback"] == "none"
    assert status["description_operation_overlap"] is False
    assert {row["relative_path"] for row in metadata} == {
        "SKILL.md",
        "scripts/format.py",
    }


def test_prompt_spans_point_inside_package_not_fixed_scaffold() -> None:
    package, files, _structure = render_loaded_package([("SKILL.md", SKILL)])
    description, operation, _status, _metadata = locate_package_regions(files, package)
    prompt, record = render_agent_prompt(
        FakeNativeTokenizer(), "Format this report.", package, DEFAULT_TOOLS
    )
    package_start = record["package_char_span"][0]
    global_description = [
        [package_start + start, package_start + end] for start, end in description
    ]
    global_operation = [
        [package_start + start, package_start + end] for start, end in operation
    ]

    assert prompt[global_description[0][0] : global_description[0][1]].startswith("---")
    assert global_description[0][0] >= package_start
    assert global_operation[0][0] >= package_start
    assert record["assistant_boundary_char_span"][0] >= record["package_char_span"][1]


def test_joint_token_partition_never_double_counts_boundary_token() -> None:
    offsets = [(0, 0), (10, 12), (12, 16), (16, 20), (20, 20)]

    description, operation, audit = partition_token_positions(
        offsets, [[10, 15]], [[15, 20]]
    )

    assert description == [1, 2]
    assert operation == [3]
    assert not set(description) & set(operation)
    assert audit["boundary_straddling_token_count"] == 1


def test_malskillbench_discovery_preserves_package_auxiliary_files(tmp_path) -> None:
    benign = tmp_path / "benign" / "same-base"
    malware = tmp_path / "malware" / "same-base"
    benign.mkdir(parents=True)
    malware.mkdir(parents=True)
    (benign / "SKILL.md").write_bytes(SKILL)
    (malware / "SKILL.md").write_bytes(SKILL.replace(b"format.py", b"send.py"))
    (malware / "send.py").write_text("print('send')\n", encoding="utf-8")

    samples = discover_malskillbench(tmp_path, "msb")
    malicious = next(sample for sample in samples if sample.label == 1)
    files = read_package_files(malicious)

    assert {relative for relative, _content in files} == {"SKILL.md", "send.py"}
    assert {sample.label for sample in samples} == {0, 1}
    assert len({sample.sample_id for sample in samples}) == 2
