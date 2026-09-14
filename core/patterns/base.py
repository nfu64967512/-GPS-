"""Pattern 共用型別與工具 (放這裡避免套件內循環匯入)。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PatternResult:
    key: str
    display_name: str
    lap_xy: np.ndarray      # (N, 2) 單圈折線, 起點==終點
    is_smooth: bool         # True=曲線(圓/8字) -> 匯出可用 SPLINE
    description: str = ""
    repeatable: bool = True  # False = 不用 DO_JUMP 重複一圈 (例如隨機手飛: 整段就是一條不重複的路徑)


def pattern_cfg(cfg: dict, key: str) -> dict:
    """取 cfg['patterns'][key]，沒有就回空 dict。"""
    return (cfg.get("patterns", {}) or {}).get(key, {}) or {}


def close_loop(poly: np.ndarray) -> np.ndarray:
    """確保折線閉合 (終點==起點)；若否則補一個起點。"""
    poly = np.asarray(poly, dtype=float)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly
