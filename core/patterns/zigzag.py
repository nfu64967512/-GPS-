"""Zigzag — 鋸齒。沿 X 前進、Y 上下交錯成對角線；到底後原路折返閉合。"""

from __future__ import annotations

import numpy as np

from .base import PatternResult, pattern_cfg


def generate(box, cfg, key, display_name) -> PatternResult:
    p = pattern_cfg(cfg, key)
    hx = box.half_x * float(p.get("width_fill", 1.0))
    hy = box.half_y * float(p.get("height_fill", 1.0))
    teeth = int(p.get("teeth", 6))                 # 對角線段數 (越多越密)
    teeth = max(2, teeth)

    xs = np.linspace(-hx, hx, teeth + 1)
    ys = np.array([hy if i % 2 == 0 else -hy for i in range(teeth + 1)], dtype=float)
    forward = np.column_stack([xs, ys])            # 左->右 鋸齒

    # 原路折返 (去掉接縫重複點) -> 閉合且連續
    backward = forward[::-1][1:]
    lap = np.vstack([forward, backward])

    return PatternResult(
        key=key,
        display_name=display_name,
        lap_xy=lap,
        is_smooth=False,
        description=f"鋸齒 {teeth} 齒，範圍 {2*hx:.2f} m × {2*hy:.2f} m",
    )
