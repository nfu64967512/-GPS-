"""
軌跡圖形 (pattern) 註冊表。

每個 pattern 模組提供 generate(box, cfg, key, display_name) -> PatternResult，
回傳「單圈、起點==終點」的水平折線 (Nx2, 公尺, 已縮放到 SafeBox)。
is_smooth 供匯出時決定是否用 SPLINE 航點。
"""

from __future__ import annotations

from .base import PatternResult, close_loop, pattern_cfg  # noqa: F401
from . import circle, figure_eight, random_walk, reciprocate, rectangle, zigzag

# key -> (display_name, module)
PATTERN_REGISTRY = {
    "reciprocate": ("Reciprocate (往返)", reciprocate),
    "rectangle": ("Rectangle (矩形)", rectangle),
    "figure_eight": ("Figure eight (8 字)", figure_eight),
    "circle": ("Circle (圓形)", circle),
    "zigzag": ("Zigzag (鋸齒)", zigzag),
    "random_walk": ("Random walk (隨機手飛)", random_walk),
}


def list_patterns():
    """回傳 [(key, display_name), ...] 供 UI 列表。"""
    return [(k, v[0]) for k, v in PATTERN_REGISTRY.items()]


def generate(key: str, box, cfg) -> PatternResult:
    """產生指定 pattern 的單圈折線。"""
    if key not in PATTERN_REGISTRY:
        raise KeyError(f"未知的 pattern: {key}; 可用: {list(PATTERN_REGISTRY)}")
    display_name, module = PATTERN_REGISTRY[key]
    result = module.generate(box, cfg, key, display_name)
    # 在唯一進入點統一保證「起點==終點」不變式 (tile_closed 仰賴它)，
    # 各 pattern 不必各自確保；未來新 pattern 即使回傳未閉合折線也不會被靜默丟點。
    result.lap_xy = close_loop(result.lap_xy)
    return result
