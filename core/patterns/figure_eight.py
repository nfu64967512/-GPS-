"""Figure eight — 8 字 (Gerono lemniscate)。

x = A cosθ, y = half_y · sin(2θ), θ∈[0, 2π]。兩瓣在原點交叉 (水平 8 字)，與圖示一致。
"""

from __future__ import annotations

import numpy as np

from .base import PatternResult, pattern_cfg


def generate(box, cfg, key, display_name) -> PatternResult:
    p = pattern_cfg(cfg, key)
    samples = int(p.get("samples", 960))
    A = box.half_x * float(p.get("width_fill", 1.0))    # x 半幅
    B = box.half_y * float(p.get("height_fill", 1.0))   # y 半幅 (sin2θ 峰值)

    theta = np.linspace(0.0, 2 * np.pi, samples + 1)
    x = A * np.cos(theta)
    y = B * np.sin(2 * theta)
    lap = np.column_stack([x, y])

    return PatternResult(
        key=key,
        display_name=display_name,
        lap_xy=lap,
        is_smooth=True,
        description=f"8 字 寬 {2*A:.2f} m × 高 {2*B:.2f} m",
    )
