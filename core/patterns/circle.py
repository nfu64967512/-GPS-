"""Circle — 圓形。x=R cosθ, y=R sinθ (逆時針)。"""

from __future__ import annotations

import numpy as np

from .base import PatternResult, pattern_cfg


def generate(box, cfg, key, display_name) -> PatternResult:
    p = pattern_cfg(cfg, key)
    samples = int(p.get("samples", 720))
    R = min(box.half_x, box.half_y) * float(p.get("radius_fill", 1.0))

    theta = np.linspace(0.0, 2 * np.pi, samples + 1)  # 含終點 -> 閉合
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    lap = np.column_stack([x, y])

    return PatternResult(
        key=key,
        display_name=display_name,
        lap_xy=lap,
        is_smooth=True,
        description=f"圓形 半徑 {R:.2f} m",
    )
