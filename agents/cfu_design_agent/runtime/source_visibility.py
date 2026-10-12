"""Agent-only views: hide the benchmark oracle, never alter host source bytes.

This is source disclosure control, not a semantic/anti-cheat proof. Result logs,
algorithm constants outside the verifier, and measured evidence remain visible.
"""
from __future__ import annotations

import json
import re
from typing import Any

VERSION = 1
MARKER = "/* verify_benchmark body hidden from agent */"
_NUMBERED = re.compile(r"(?m)^[ \t]*\d+\t")
_NAME = re.compile(r"\bverify_benchmark\s*\(")
_LLVM = re.compile(r'(?m)^\s*define\b[^\n]*@(?:verify_benchmark(?:\.[\w.]+)?|"verify_benchmark(?:\.[\w.]+)?")\s*\(')
_XML = re.compile(r'''<function\b[^>]*\b(?:name|function)\s*=\s*(['"])verify_benchmark\1[^>]*>''')
_ASM = re.compile(r"(?m)^[ \t]*verify_benchmark(?:\.[\w.]+)?:[ \t]*(?:[#;][^\n]*)?$")
_DISASM = re.compile(r"(?m)^[ \t]*(?:[\da-fA-F]+[ \t]+)?<verify_benchmark(?:\.[\w.]+)?>:[ \t]*$")


def _lexical(text: str) -> str:
    """Blank C comments/literals and read_file numbering; preserve offsets."""
    chars = list(text)
    for match in _NUMBERED.finditer(text):
        chars[match.start():match.end()] = " " * (match.end() - match.start())
    index = 0
    while index < len(text):
        start = index
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
        elif text[index] in "\"'":
            quote = text[index]
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == quote:
                    index += 1
                    break
                elif text[index] == "\n":
                    break  # C literals cannot cross an unescaped newline.
                else:
                    index += 1
        else:
            index += 1
            continue
        for pos in range(start, min(index, len(text))):
            if chars[pos] not in "\r\n":
                chars[pos] = " "
    return "".join(chars)


def _end(code: str, opening: int, left: str, right: str) -> int:
    depth = 0
    for pos in range(opening, len(code)):
        if code[pos] == left:
            depth += 1
        elif code[pos] == right:
            depth -= 1
            if depth == 0:
                return pos
    return len(code)


def verifier_body_ranges(text: str) -> list[tuple[int, int]]:
    """C/LLVM body interiors; an incomplete body is hidden through EOF."""
    code = _lexical(text)
    ranges = []
    for match in _NAME.finditer(code):
        opening = code.find("(", match.start(), match.end())
        pos = _end(code, opening, "(", ")") + 1
        # GNU attributes may sit between the declarator and body.
        while pos < len(code):
            while pos < len(code) and code[pos].isspace():
                pos += 1
            attribute = re.match(r"__(?:attribute__|attribute|declspec)\s*\(", code[pos:])
            if not attribute:
                break
            attr_open = code.find("(", pos, pos + attribute.end())
            pos = _end(code, attr_open, "(", ")") + 1
        if pos < len(code) and code[pos] == "{":
            ranges.append((pos + 1, _end(code, pos, "{", "}")))
    # IR views retain a recognizable verifier symbol, but not its instructions.
    for match in _LLVM.finditer(text):
        pos = code.find("{", match.end())
        if pos >= 0:
            ranges.append((pos + 1, _end(code, pos, "{", "}")))
    for match in _ASM.finditer(text):
        following = text[match.end():]
        end = re.search(r"(?m)^\s*\.size\s+verify_benchmark\b|^\s*\.type\s+\w+\s*,\s*[@%]function", following)
        ranges.append((match.end(), match.end() + end.start() if end else len(text)))
    for match in _DISASM.finditer(text):
        following = text[match.end():]
        end = re.search(r"(?m)^[ \t]*(?:[\da-fA-F]+[ \t]+)?<[\w.$]+>:", following)
        ranges.append((match.end(), match.end() + end.start() if end else len(text)))
    for match in _XML.finditer(text):
        end = text.find("</function>", match.end())
        ranges.append((match.end(), len(text) if end < 0 else end))
    outer = []
    for start, end in sorted(set(ranges)):
        if not outer or start >= outer[-1][1]:
            outer.append((start, end))
    return outer


def mask_verifier_bodies(text: str) -> str:
    """Keep the signature, braces, surrounding code and original line count.

    Call on a complete source *before* slicing, searching or truncating it; a
    body-only fragment cannot identify its owning function on its own.
    """
    if not isinstance(text, str) or "verify_benchmark" not in text:
        return text
    for start, end in reversed(verifier_body_ranges(text)):
        body = text[start:end]
        if MARKER in body and not _NUMBERED.sub("", body).replace(MARKER, "").strip():
            continue
        hidden = "".join(char if char in "\r\n" else " " for char in body)
        # Numbered code copied into prompts should retain its visible line labels.
        hidden_chars = list(hidden)
        for match in _NUMBERED.finditer(body):
            hidden_chars[match.start():match.end()] = body[match.start():match.end()]
        text = text[:start] + MARKER + "".join(hidden_chars) + text[end:]
    return text


def mask_file_view(text: str, path: str = "") -> str:
    """Also hide code stored in escaped JSON/YAML audit strings, without rewriting files."""
    if "verify_benchmark" not in text:
        return text
    if re.search(r"\.(?:jsonl?|ya?ml|txt|md|log)(?:\.[0-9a-fA-F]{64})?$", path):
        def replace_string(match):
            raw = match.group(0)
            if "verify_benchmark" not in raw:
                return raw
            try:
                decoded = json.loads(raw)
            except ValueError:
                import yaml
                try:
                    decoded = yaml.safe_load(raw)
                except yaml.YAMLError:
                    return raw
            if not isinstance(decoded, str):
                return raw
            visible = mask_verifier_bodies(decoded)
            if visible == decoded:
                return raw
            encoded = json.dumps(visible, ensure_ascii=False)
            # YAML double-quoted continuation lines preserve physical line numbers.
            return encoded[:-1] + (chr(92) + "\n") * raw.count("\n") + encoded[-1:]
        text = re.sub(r'''"(?:\\.|[^"\\])*"|'(?:''|[^'])*' ''', replace_string,
                      text, flags=re.S | re.X)
    return mask_verifier_bodies(text)


def mask_model_payload(value: Any) -> Any:
    """Copy a request payload, masking code blocks including JSON-encoded strings."""
    if isinstance(value, dict):
        return {key: mask_model_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_model_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_model_payload(item) for item in value)
    if not isinstance(value, str) or "verify_benchmark" not in value:
        return value
    if value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            pass
        else:
            masked = mask_model_payload(decoded)
            if masked != decoded:
                return json.dumps(masked, ensure_ascii=False)
    return mask_file_view(value, "model_prompt.txt")
