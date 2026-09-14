"""Reciprocate — 往返。兩點一線：沿 X 軸在 A、B 兩點間來回 (去回走同一條線)。"""

from __future__ import annotations

import numpy as np

from .base import PatternResult, pattern_cfg


def generate(box, cfg, key, display_name) -> PatternResult:
    p = pattern_cfg(cfg, key)
    hx = box.half_x * float(p.get("length_fill", 1.0))
    # 去/回兩條線的 y 間距 (佔 half_y 比例)。預設 0 = 兩點一線往返；
    # >0 才會把去程/回程錯開成兩條平行線 (矩形迴圈, 舊行為)。
    lane = float(p.get("lane_offset_fill", 0.0)) * box.half_y

    if lane <= 0.0:
        # 兩點一線：A(-hx) -> B(hx) -> A(-hx)，去回同一條線，閉合單圈
        lap = np.array(
            [
                [-hx, 0.0],
                [hx, 0.0],
                [-hx, 0.0],
            ],
            dtype=float,
        )
        desc = f"往返 (兩點一線) 長度 {2 * hx:.2f} m"
    else:
        # 雙線往返：左上 -> 右上 (去) -> 右下 -> 左下 (回) -> 回左上 (閉合)
        lap = np.array(
            [
                [-hx, lane],
                [hx, lane],
                [hx, -lane],
                [-hx, -lane],
                [-hx, lane],
            ],
            dtype=float,
        )
        desc = f"往返 (雙線) 長度 {2 * hx:.2f} m，雙線間距 {2 * lane:.2f} m"

    return PatternResult(
        key=key,
        display_name=display_name,
        lap_xy=lap,
        is_smooth=False,
        description=desc,
    )
