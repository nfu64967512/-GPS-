"""Rectangle — 矩形周界 (順時針，從左上角起)。"""

from __future__ import annotations

import numpy as np

from .base import PatternResult, pattern_cfg


def generate(box, cfg, key, display_name) -> PatternResult:
    p = pattern_cfg(cfg, key)
    hx = box.half_x * float(p.get("width_fill", 1.0))
    hy = box.half_y * float(p.get("height_fill", 1.0))

    # 左上 -> 右上 -> 右下 -> 左下 -> 回左上 (順時針)
    lap = np.array(
        [
            [-hx, hy],
            [hx, hy],
            [hx, -hy],
            [-hx, -hy],
            [-hx, hy],
        ],
        dtype=float,
    )

    return PatternResult(
        key=key,
        display_name=display_name,
        lap_xy=lap,
        is_smooth=False,
        description=f"矩形 {2*hx:.2f} m × {2*hy:.2f} m",
    )
