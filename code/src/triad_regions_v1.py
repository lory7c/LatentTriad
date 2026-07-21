"""Deterministic region parsing for the TRIAD pre-action detector."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

import yaml


FILE_OPEN = re.compile(r'<skill_file path="([^"]+)">\r?\n')
FILE_CLOSE = re.compile(r"\r?\n</skill_file>")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")

CODE_SUFFIXES = frozenset(
    {
        ".bash",
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".ipynb",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".lua",
        ".m",
        ".php",
        ".pl",
        ".ps1",
        ".py",
        ".r",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".swift",
        ".ts",
        ".tsx",
        ".zsh",
    }
)


@dataclass(frozen=True)
class FileBlock:
    path: str
    content: str
    complete: bool
    path_safe: bool


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def is_safe_relative_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    return bool(normalized) and not candidate.is_absolute() and ".." not in candidate.parts


def parse_file_blocks(serialized: str) -> list[FileBlock]:
    """Parse complete blocks and retain a final block cut by token truncation."""
    openings = list(FILE_OPEN.finditer(serialized))
    blocks: list[FileBlock] = []
    for index, opening in enumerate(openings):
        path = opening.group(1)
        content_start = opening.end()
        limit = openings[index + 1].start() if index + 1 < len(openings) else len(serialized)
        closing = FILE_CLOSE.search(serialized, content_start, limit)
        content_end = closing.start() if closing else limit
        content = serialized[content_start:content_end].rstrip("\r\n")
        blocks.append(
            FileBlock(
                path=path,
                content=content,
                complete=closing is not None,
                path_safe=is_safe_relative_path(path),
            )
        )
    return blocks


def path_kind(path: str) -> str:
    normalized = path.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.name.casefold() == "skill.md":
        return "skill_instruction"
    if candidate.suffix.casefold() in CODE_SUFFIXES:
        return "code_file"
    return "auxiliary_file"


def root_skill_block(blocks: list[FileBlock]) -> Optional[FileBlock]:
    skills = [block for block in blocks if path_kind(block.path) == "skill_instruction"]
    if not skills:
        return None
    roots = [block for block in skills if block.path.replace("\\", "/").casefold() == "skill.md"]
    candidates = roots or sorted(
        skills,
        key=lambda block: (
            len(PurePosixPath(block.path.replace("\\", "/")).parts),
            block.path.casefold(),
        ),
    )
    return candidates[0]


def split_markdown_regions(markdown: str) -> tuple[str, str, bool]:
    """Separate Markdown prose from fenced implementation without rendering it."""
    prose: list[str] = []
    code: list[str] = []
    fence_char: Optional[str] = None
    fence_width = 0
    for line in markdown.splitlines(keepends=True):
        match = FENCE.match(line.rstrip("\r\n"))
        marker = match.group(1) if match else ""
        if fence_char is None and marker:
            fence_char = marker[0]
            fence_width = len(marker)
            continue
        if (
            fence_char is not None
            and marker
            and marker[0] == fence_char
            and len(marker) >= fence_width
        ):
            fence_char = None
            fence_width = 0
            continue
        (code if fence_char is not None else prose).append(line)
    return "".join(prose).strip(), "".join(code).strip(), fence_char is None


def split_frontmatter(markdown: str) -> tuple[dict, str, str, bool, Optional[str]]:
    """Parse root YAML frontmatter with a safe loader and preserve the body."""
    lines = markdown.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, markdown, "", True, None
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() in {"---", "..."}
        ),
        None,
    )
    if closing_index is None:
        return {}, markdown, "", False, "unterminated_frontmatter"
    raw = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1 :])
    try:
        loaded = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}, body, raw, True, "invalid_yaml"
    if not isinstance(loaded, dict):
        return {}, body, raw, True, "frontmatter_not_mapping"
    return loaded, body, raw, True, None


def declared_intent_text(frontmatter: dict) -> str:
    """Use only registry-facing declaration fields, never operational body text."""
    values: list[str] = []
    for key in ("description", "summary", "purpose"):
        value = frontmatter.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return "\n".join(values)


def extract_tioc_region_texts(visible: str) -> dict[str, str]:
    """Return normalized declaration and actual-behavior views from frozen bytes."""
    blocks = parse_file_blocks(visible)
    root = root_skill_block(blocks)
    if root is None:
        raise ValueError("Visible package has no SKILL.md block")
    frontmatter, body, frontmatter_raw, complete, error = split_frontmatter(root.content)
    if not complete or error is not None:
        raise ValueError(f"Visible SKILL.md frontmatter is not usable: {error}")
    declaration = declared_intent_text(frontmatter)
    operational, inline_code, fences_complete = split_markdown_regions(body)
    if not fences_complete:
        raise ValueError("Visible SKILL.md has an unterminated code fence")

    code_files: list[str] = []
    auxiliary_files: list[str] = []
    for block in blocks:
        kind = path_kind(block.path)
        if kind == "skill_instruction":
            continue
        rendered = f'<implementation_file path="{block.path}">\n{block.content}\n</implementation_file>'
        (code_files if kind == "code_file" else auxiliary_files).append(rendered)

    actual_parts = []
    if operational:
        actual_parts.append(
            f"<operational_instructions>\n{operational}\n</operational_instructions>"
        )
    if inline_code:
        actual_parts.append(f"<inline_implementation>\n{inline_code}\n</inline_implementation>")
    actual_parts.extend(code_files)
    actual_parts.extend(auxiliary_files)
    return {
        "metadata_raw": frontmatter_raw.strip(),
        "declaration": declaration,
        "operational_instructions": operational,
        "inline_implementation": inline_code,
        "code_files": "\n".join(code_files),
        "auxiliary_files": "\n".join(auxiliary_files),
        "actual": "\n".join(actual_parts),
    }


def _region_stats(contents: list[str]) -> dict:
    text = "\n".join(contents)
    return {
        "file_count": len(contents),
        "char_count": sum(len(content) for content in contents),
        "sha256": sha256_text(text),
    }


def summarize_package_regions(original: str, visible: str, package_truncated: bool) -> dict:
    original_blocks = parse_file_blocks(original)
    visible_blocks = parse_file_blocks(visible)
    visible_root = root_skill_block(visible_blocks)
    original_root = root_skill_block(original_blocks)
    declared_prose = ""
    operational_prose = ""
    inline_code = ""
    markdown_fences_complete = True
    frontmatter_complete = True
    frontmatter_error = None
    frontmatter_raw = ""
    if visible_root is not None:
        frontmatter, body, frontmatter_raw, frontmatter_complete, frontmatter_error = (
            split_frontmatter(visible_root.content)
        )
        declared_prose = declared_intent_text(frontmatter)
        operational_prose, inline_code, markdown_fences_complete = (
            split_markdown_regions(body)
        )

    def contents(blocks: list[FileBlock], kind: str) -> list[str]:
        return [block.content for block in blocks if path_kind(block.path) == kind]

    original_code = contents(original_blocks, "code_file")
    original_aux = contents(original_blocks, "auxiliary_file")
    visible_code = contents(visible_blocks, "code_file")
    visible_aux = contents(visible_blocks, "auxiliary_file")

    original_code_chars = sum(map(len, original_code))
    original_aux_chars = sum(map(len, original_aux))
    visible_code_chars = sum(map(len, visible_code))
    visible_aux_chars = sum(map(len, visible_aux))

    def ratio(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 8) if denominator else None

    return {
        "original_block_count": len(original_blocks),
        "visible_block_count": len(visible_blocks),
        "visible_incomplete_block_count": sum(not block.complete for block in visible_blocks),
        "visible_unsafe_path_count": sum(not block.path_safe for block in visible_blocks),
        "package_truncated": bool(package_truncated),
        "root_skill_present_original": original_root is not None,
        "root_skill_present_visible": visible_root is not None,
        "root_skill_complete_visible": bool(visible_root and visible_root.complete),
        "frontmatter_complete_visible": frontmatter_complete,
        "frontmatter_parse_error": frontmatter_error,
        "frontmatter_sha256": sha256_text(frontmatter_raw),
        "markdown_fences_complete_visible": markdown_fences_complete,
        "metadata_declaration": _region_stats(
            [declared_prose] if declared_prose else []
        ),
        "operational_instructions": _region_stats(
            [operational_prose] if operational_prose else []
        ),
        "inline_code": _region_stats([inline_code] if inline_code else []),
        "code_files_original": _region_stats(original_code),
        "code_files_visible": _region_stats(visible_code),
        "auxiliary_files_original": _region_stats(original_aux),
        "auxiliary_files_visible": _region_stats(visible_aux),
        "code_char_visibility_ratio": ratio(visible_code_chars, original_code_chars),
        "auxiliary_char_visibility_ratio": ratio(visible_aux_chars, original_aux_chars),
        "alignment_region_ready": bool(
            declared_prose
            and (operational_prose or inline_code or visible_code_chars or visible_aux_chars)
            and visible_root
            and visible_root.path_safe
            and frontmatter_complete
            and frontmatter_error is None
            and markdown_fences_complete
        ),
        "instruction_alignment_region_ready": bool(
            declared_prose
            and operational_prose
            and visible_root
            and visible_root.path_safe
            and frontmatter_complete
            and frontmatter_error is None
            and markdown_fences_complete
        ),
        "cross_artifact_region_ready": bool(
            declared_prose
            and (visible_code_chars or visible_aux_chars)
            and visible_root
            and visible_root.path_safe
            and frontmatter_complete
            and frontmatter_error is None
            and markdown_fences_complete
        ),
        "visible_paths": [
            {
                "path": block.path,
                "complete": block.complete,
                "path_safe": block.path_safe,
            }
            for block in visible_blocks
        ],
    }
