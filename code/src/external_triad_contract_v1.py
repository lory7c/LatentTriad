"""Deterministic input and region contract for external TRIAD benchmarks.

This module deliberately contains no model code.  It turns inert package bytes
into one model-native prompt and records the exact description, operation, and
pre-action boundary locations consumed by the feature extractor.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Protocol, Sequence


SCHEMA_VERSION = "skillprobe-external-triad-contract-v1.0"
HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
STANDARD_AGENT_SYSTEM = (
    "You are an AI agent. Use the available tools when needed to complete the "
    "user's task. Use the loaded skill as task-specific workflow guidance and "
    "resources while carrying out the task."
)
DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]


class NativeTokenizerLike(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str: ...


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return sha256_bytes(payload)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(payload)


def decode_text(content: bytes) -> str | None:
    if b"\0" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or '"' in normalized
        or "\n" in normalized
        or "\r" in normalized
    ):
        raise ValueError(f"Unsafe package path: {value!r}")
    return path.as_posix()


def render_loaded_package(
    files: Iterable[tuple[str, bytes]],
) -> tuple[str, list[tuple[str, bytes]], dict[str, int]]:
    """Serialize package bytes without executing or interpreting file content."""
    normalized = [(_safe_relative_path(path), content) for path, content in files]
    normalized.sort(key=lambda row: row[0])
    paths = [path for path, _content in normalized]
    if not normalized:
        raise ValueError("Loaded package is empty")
    if len(paths) != len(set(paths)):
        raise ValueError("Loaded package contains duplicate relative paths")
    if "SKILL.md" not in paths:
        raise ValueError("Loaded package lacks a root SKILL.md")

    parts: list[str] = []
    text_files = 0
    binary_files = 0
    package_size = 0
    for relative, content in normalized:
        package_size += len(content)
        text = decode_text(content)
        if text is None:
            binary_files += 1
            parts.append(
                f'<binary_file path="{relative}" sha256="{sha256_bytes(content)}"/>'
            )
        else:
            text_files += 1
            parts.append(f'<skill_file path="{relative}">\n{text}\n</skill_file>')
    return (
        "\n".join(parts),
        normalized,
        {
            "file_count": len(normalized),
            "text_file_count": text_files,
            "binary_file_count": binary_files,
            "package_size_bytes": package_size,
        },
    )


def description_cut(markdown: str) -> int:
    """End description after frontmatter, title, and the first prose section."""
    if len(markdown) <= 1:
        return len(markdown)
    headings = list(HEADING.finditer(markdown))
    if len(headings) >= 2:
        cut = headings[1].start()
    else:
        frontmatter_end = 0
        if markdown.startswith("---\n"):
            closing = markdown.find("\n---", 4)
            if closing >= 0:
                frontmatter_end = closing + len("\n---")
        search_start = headings[0].end() if headings else frontmatter_end
        paragraph_end = markdown.find("\n\n", max(search_start, 1))
        if paragraph_end >= 0:
            cut = paragraph_end
        elif frontmatter_end:
            cut = frontmatter_end
        else:
            cut = min(max(128, len(markdown) // 3), 1024)
    return min(max(int(cut), 1), len(markdown) - 1)


def locate_package_regions(
    files: Sequence[tuple[str, bytes]], package_text: str
) -> tuple[list[list[int]], list[list[int]], dict[str, Any], list[dict[str, Any]]]:
    """Locate description and operation character spans inside serialized bytes."""
    cursor = 0
    text_spans: dict[str, tuple[int, int]] = {}
    metadata: list[dict[str, Any]] = []
    root_text: str | None = None
    for index, (relative, content) in enumerate(files):
        if index:
            if package_text[cursor : cursor + 1] != "\n":
                raise ValueError("Loaded-package separator mismatch")
            cursor += 1
        text = decode_text(content)
        if text is None:
            part = (
                f'<binary_file path="{relative}" sha256="{sha256_bytes(content)}"/>'
            )
            if package_text[cursor : cursor + len(part)] != part:
                raise ValueError("Rendered binary package mismatch")
            cursor += len(part)
        else:
            prefix = f'<skill_file path="{relative}">\n'
            suffix = "\n</skill_file>"
            part = prefix + text + suffix
            if package_text[cursor : cursor + len(part)] != part:
                raise ValueError("Rendered text package mismatch")
            start = cursor + len(prefix)
            text_spans[relative] = (start, start + len(text))
            if relative == "SKILL.md":
                root_text = text
            cursor += len(part)
        metadata.append(
            {
                "relative_path": relative,
                "size_bytes": len(content),
                "sha256": sha256_bytes(content),
                "is_text": text is not None,
                "suffix": PurePosixPath(relative).suffix.casefold(),
            }
        )
    if cursor != len(package_text):
        raise ValueError("Loaded-package region reconstruction is incomplete")

    root_span = text_spans.get("SKILL.md")
    if root_span is None or root_text is None or not root_text:
        raise ValueError("Loaded package lacks a non-empty textual root SKILL.md")
    cut = description_cut(root_text)
    description = [[root_span[0], root_span[0] + cut]]
    operation: list[list[int]] = []
    if cut < len(root_text):
        operation.append([root_span[0] + cut, root_span[1]])
    operation.extend(
        [start, end]
        for relative, (start, end) in sorted(text_spans.items())
        if relative != "SKILL.md" and end > start
    )
    fallback = "none"
    if not operation:
        operation = [[root_span[0], root_span[1]]]
        fallback = "operation_overlaps_root_skill_due_to_no_remaining_text"
    overlap = any(
        max(left[0], right[0]) < min(left[1], right[1])
        for left in description
        for right in operation
    )
    return (
        description,
        operation,
        {
            "strategy": "root_frontmatter_title_first_section_vs_remaining_and_auxiliary",
            "description_cut_char": cut,
            "description_operation_overlap": overlap,
            "fallback": fallback,
        },
        metadata,
    )


def render_agent_prompt(
    tokenizer: NativeTokenizerLike,
    task: str,
    package_text: str,
    tools: list[dict],
) -> tuple[str, dict[str, Any]]:
    """Render one native pre-action prompt and bind semantic character spans."""
    if not task.strip() or not package_text:
        raise ValueError("Task and package must be non-empty")
    if not tools:
        raise ValueError("Tool schema must be non-empty")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no native chat template")
    user_content = (
        f"<trusted_user_task>\n{task}\n</trusted_user_task>\n\n"
        '<loaded_skill_package input_unit="owning_skill_subtree" '
        'serialization="path_sorted_inert_static">\n'
        f"{package_text}\n</loaded_skill_package>"
    )
    messages = [
        {"role": "system", "content": STANDARD_AGENT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    without_generation = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    without_tools = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Native chat template returned an empty prompt")
    if prompt == without_tools:
        raise ValueError("Native chat template ignored the frozen tool schema")
    if prompt == without_generation:
        boundary_kind = "implicit_prompt_end"
        boundary_span = [len(prompt), len(prompt)]
    elif prompt.startswith(without_generation):
        boundary_kind = "explicit_generation_suffix"
        boundary_span = [len(without_generation), len(prompt)]
    else:
        raise ValueError("Cannot derive native assistant generation boundary")
    if prompt.count(package_text) != 1:
        raise ValueError("Rendered prompt does not preserve package bytes exactly once")
    package_start = prompt.index(package_text)
    task_block = f"<trusted_user_task>\n{task}\n</trusted_user_task>"
    if prompt.count(task_block) != 1:
        raise ValueError("Rendered prompt does not preserve the trusted task exactly once")
    task_start = prompt.index(task_block) + len("<trusted_user_task>\n")
    return (
        prompt,
        {
            "prompt_sha256": sha256_text(prompt),
            "package_char_span": [package_start, package_start + len(package_text)],
            "task_char_span": [task_start, task_start + len(task)],
            "assistant_boundary_kind": boundary_kind,
            "assistant_boundary_char_span": boundary_span,
            "tool_schema_sha256": canonical_json_sha256(tools),
        },
    )


def _overlap_length(left: int, right: int, span: Sequence[int]) -> int:
    return max(0, min(right, int(span[1])) - max(left, int(span[0])))


def partition_token_positions(
    offsets: Sequence[Sequence[int]],
    description_spans: Sequence[Sequence[int]],
    operation_spans: Sequence[Sequence[int]],
) -> tuple[list[int], list[int], dict[str, int]]:
    """Jointly map spans so a tokenizer boundary token cannot enter both regions."""
    description: list[int] = []
    operation: list[int] = []
    ambiguous = 0
    for index, values in enumerate(offsets):
        left, right = int(values[0]), int(values[1])
        if right <= left:
            continue
        desc_overlap = sum(
            _overlap_length(left, right, span) for span in description_spans
        )
        oper_overlap = sum(
            _overlap_length(left, right, span) for span in operation_spans
        )
        if not desc_overlap and not oper_overlap:
            continue
        if desc_overlap and oper_overlap:
            ambiguous += 1
        if desc_overlap > oper_overlap:
            description.append(index)
        elif oper_overlap > desc_overlap:
            operation.append(index)
        elif any(int(span[0]) <= left < int(span[1]) for span in description_spans):
            description.append(index)
        else:
            operation.append(index)
    if not description or not operation:
        raise ValueError("Description and operation must both map to non-empty tokens")
    if set(description) & set(operation):
        raise AssertionError("Joint token partition produced overlapping regions")
    return (
        description,
        operation,
        {
            "description_token_count": len(description),
            "operation_token_count": len(operation),
            "boundary_straddling_token_count": ambiguous,
        },
    )
