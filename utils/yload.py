"""Minimal YAML-like loader supporting a subset used by this repo.

If PyYAML is available, uses it; otherwise parses a limited subset:
- key: value pairs (str in quotes, int/float, true/false/null)
- nested mappings via indentation (2 spaces per level)
- inline dicts: key: {a: 1, b: 2}
- inline lists: key: [1, 2, 3]
- lists of dict items with leading '-'

This is not a general YAML parser; it is sufficient for the provided configs.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _parse_scalar(tok: str) -> Any:
    t = tok.strip()
    if t.lower() in {"true", "false"}:
        return t.lower() == "true"
    if t.lower() == "null":
        return None
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        return t[1:-1]
    # int/float
    try:
        if "." in t:
            return float(t)
        return int(t)
    except ValueError:
        return t


def _parse_inline_dict(s: str) -> Dict[str, Any]:
    inner = s.strip().strip("{}").strip()
    if not inner:
        return {}
    parts = [p.strip() for p in inner.split(",")]
    out: Dict[str, Any] = {}
    for p in parts:
        k, v = p.split(":", 1)
        out[k.strip()] = _parse_scalar(v)
    return out


def _parse_inline_list(s: str) -> List[Any]:
    inner = s.strip().strip("[]").strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [_parse_scalar(p) for p in parts]


def load(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        pass

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]  # (indent_level, container)
    current_key_stack: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
            if current_key_stack:
                current_key_stack.pop()
        container = stack[-1][1]
        s = line.strip()
        if s.startswith("-"):
            # list item; ensure container is a list
            if not isinstance(container, list):
                # create a new list for the last key in parent dict
                if isinstance(stack[-1][1], dict):
                    parent = stack[-1][1]
                    key = current_key_stack[-1] if current_key_stack else None
                    if key is not None and not isinstance(parent.get(key), list):
                        parent[key] = []
                    container = parent[key]
                    stack[-1] = (stack[-1][0], container)
            item = s[1:].strip()
            if ":" in item:
                # dict starting inline for this item
                key, val = item.split(":", 1)
                d: Dict[str, Any] = {key.strip(): _parse_scalar(val) if val.strip() else {}}
                container.append(d)
                if val.strip() == "":
                    stack.append((indent, d))
                    current_key_stack.append(key.strip())
                continue
            else:
                container.append(_parse_scalar(item))
                continue

        # key: value
        if ":" in s:
            key, val = s.split(":", 1)
            key = key.strip()
            val = val.strip()
            target = container
            if isinstance(target, list):
                # convert to dict item in list
                d: Dict[str, Any] = {}
                target.append(d)
                target = d
            if not val:
                # start of nested mapping
                target[key] = {}
                stack.append((indent, target[key]))
                current_key_stack.append(key)
            elif val.startswith("{"):
                target[key] = _parse_inline_dict(val)
            elif val.startswith("["):
                target[key] = _parse_inline_list(val)
            else:
                target[key] = _parse_scalar(val)
        else:
            # Bare value not expected in our configs
            continue

    return root

