"""Strict first-action parser for the Qwen2.5-Instruct chat-template format."""

from __future__ import annotations

import json


OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def parse_first_action(text: str) -> dict:
    """Parse only Qwen's first native tool-call block; never dispatch it."""
    if not isinstance(text, str):
        return {"status": "malformed", "reason": "generated_output_is_not_text"}
    first = len(text) - len(text.lstrip())
    remainder = text[first:]
    if not remainder or OPEN_TAG.startswith(remainder):
        return {"status": "incomplete", "reason": "awaiting_qwen_tool_call_open"}
    if not remainder.startswith(OPEN_TAG):
        return {
            "status": "malformed",
            "reason": "first_output_is_not_qwen_tool_call",
        }
    payload_start = first + len(OPEN_TAG)
    payload_end = text.find(CLOSE_TAG, payload_start)
    if payload_end < 0:
        return {
            "status": "incomplete",
            "reason": "awaiting_qwen_tool_call_close",
        }
    payload_text = text[payload_start:payload_end].strip()
    try:
        payload = json.loads(
            payload_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"status": "malformed", "reason": "tool_call_json_invalid"}
    if not isinstance(payload, dict):
        return {
            "status": "malformed",
            "reason": "tool_call_payload_is_not_object",
        }
    if set(payload) != {"name", "arguments"}:
        return {"status": "malformed", "reason": "tool_call_keys_mismatch"}
    if not isinstance(payload["name"], str) or not payload["name"]:
        return {"status": "malformed", "reason": "tool_call_name_is_invalid"}
    if not isinstance(payload["arguments"], dict):
        return {
            "status": "malformed",
            "reason": "tool_call_arguments_are_not_object",
        }
    return {
        "status": "tool_call",
        "tool_name": payload["name"],
        "arguments": payload["arguments"],
        "consumed_character_end": payload_end + len(CLOSE_TAG),
    }
