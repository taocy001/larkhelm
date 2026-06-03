"""check_config_sync.py — verify that every config.py setdefault key is present
in larkhelm_config.example.json, and optionally fix missing keys.

Usage:
    python3 scripts/check_config_sync.py              # report only, exit 1 if drift
    python3 scripts/check_config_sync.py --fix        # append missing keys to JSON
"""
from __future__ import annotations

import ast
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PY = _REPO_ROOT / "larkhelm" / "config.py"
_EXAMPLE_JSON = _REPO_ROOT / "larkhelm_config.example.json"


def extract_setdefault_keys(config_py_path: Path) -> set[str]:
    """Parse *config_py_path* with ``ast`` and return every string key passed to
    ``config.setdefault(KEY, ...)`` calls."""
    return {k: v for k, v in extract_setdefault_pairs(config_py_path).items()}.keys()  # type: ignore[return-value]


def extract_setdefault_pairs(config_py_path: Path) -> dict[str, object]:
    """Parse *config_py_path* with ``ast`` and return ``{key: default}`` for every
    ``config.setdefault(KEY, DEFAULT)`` call.  Default values that cannot be
    evaluated with ``ast.literal_eval`` fall back to ``None``."""
    source = config_py_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(config_py_path))
    pairs: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "setdefault"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        key = first.value
        default: object = None
        if len(node.args) >= 2:
            try:
                default = ast.literal_eval(node.args[1])
            except (ValueError, TypeError):
                default = None
        pairs[key] = default
    return pairs


def load_example_keys(example_json_path: Path) -> set[str]:
    """Return top-level keys from *example_json_path*, excluding ``_comment_*``."""
    data = json.loads(example_json_path.read_text(encoding="utf-8"))
    return {k for k in data if not k.startswith("_comment_")}


def fix_missing_keys(example_json_path: Path, missing: set[str],
                     defaults: dict[str, object]) -> None:
    """Append *missing* keys with their actual default values to the JSON file."""
    data = json.loads(example_json_path.read_text(encoding="utf-8"))
    for key in sorted(missing):
        data[key] = defaults.get(key)
    example_json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify config.py / example.json key sync")
    parser.add_argument("--fix", action="store_true",
                        help="append missing keys to larkhelm_config.example.json")
    parser.add_argument("--config-py", default=str(_CONFIG_PY),
                        help="path to config.py (default: auto-detected)")
    parser.add_argument("--example-json", default=str(_EXAMPLE_JSON),
                        help="path to larkhelm_config.example.json (default: auto-detected)")
    args = parser.parse_args()

    config_py_path  = Path(args.config_py)
    example_json_path = Path(args.example_json)

    pairs        = extract_setdefault_pairs(config_py_path)
    example_keys = load_example_keys(example_json_path)
    missing      = set(pairs.keys()) - example_keys

    if not missing:
        print("config-sync: OK — all setdefault keys present in example.json")
        sys.exit(0)

    print(f"config-sync: {len(missing)} key(s) in config.py setdefault but missing from example.json:")
    for key in sorted(missing):
        print(f"  - {key}")

    if args.fix:
        fix_missing_keys(example_json_path, missing, pairs)
        print(f"config-sync: appended {len(missing)} key(s) to {example_json_path}")
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
