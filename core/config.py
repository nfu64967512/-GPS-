"""
設定載入 / 合併。

讀取 config/default.yaml，可被使用者設定檔或 GUI 覆寫。
回傳巢狀 dict (方便 GUI 雙向綁定)，並提供取值/深層更新工具。
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "config", "default.yaml")
)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """載入設定。先讀 default.yaml；若另給 path 則深層覆寫其上。"""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if path and os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG_PATH):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user)

    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """把設定寫成 YAML。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """遞迴合併 overrides 進 base 的拷貝並回傳。"""
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """以 'flight.cruise_speed' 形式取值。"""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
