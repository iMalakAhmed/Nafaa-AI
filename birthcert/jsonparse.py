"""Robust JSON extraction from vision-model output.

Self-contained (no heavy deps) so it can run locally and on Modal. Handles the
usual VL mistakes: code fences, smart/fullwidth punctuation, trailing commas,
invalid escapes, and truncated objects.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL)


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def _sanitize_escapes(text: str) -> str:
    def decode_unicode(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return ""

    text = re.sub(r"\\u([0-9a-fA-F]{4})", decode_unicode, text)
    text = re.sub(r"\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])", "", text)
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)
    return text


def repair_json_text(text: str) -> str:
    """Fix common vision-model JSON mistakes before parsing."""
    candidate = _strip_code_fences(text.strip()) if text.strip().startswith("```") else text.strip()
    candidate = _strip_control_chars(candidate)
    candidate = _sanitize_escapes(candidate)
    candidate = (
        candidate.replace("\uff0c", ",")
        .replace("\uff1a", ":")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    candidate = re.sub(r"//[^\n]*", "", candidate)
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
    candidate = re.sub(r"\{\s*\}", "{}", candidate)
    return candidate


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object out of (possibly messy) model text."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = _strip_code_fences(candidate)

    for attempt in (candidate, repair_json_text(candidate)):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                fragment = candidate[start : index + 1]
                for fix in (fragment, repair_json_text(fragment)):
                    try:
                        return json.loads(fix)
                    except json.JSONDecodeError:
                        continue

    # Truncated object: try to close it by trimming to the last complete field.
    fragment = candidate[start:]
    trimmed = re.sub(r",\s*\"[^\"]*\"\s*:\s*[^,}\]]*$", "", fragment).rstrip().rstrip(",")
    for closer in ("}", "}}", "}}}", "}}}}"):
        try:
            return json.loads(repair_json_text(trimmed + closer))
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse a complete JSON object from model output.")
