"""
幾何 / 座標工具。

- SafeBox：由空間尺寸 + 安全邊界推導的可用飛行盒 (房間本地 ENU, 中心為原點)。
- ENU(公尺) <-> WGS84 lat/lon：繞一個可設定假原點的本地切平面近似 (室內 5 m 等級
  誤差可忽略)。給 .waypoints (AUTO) 用，搭配飛控的 SET_GPS_GLOBAL_ORIGIN。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

EARTH_RADIUS_M = 6378137.0  # WGS84 長半徑


@dataclass
class SafeBox:
    """可用飛行盒 (公尺, 房間本地座標; x=東, y=北, z=上, 中心原點在地板上方)。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def half_x(self) -> float:
        return (self.x_max - self.x_min) / 2.0

    @property
    def half_y(self) -> float:
        return (self.y_max - self.y_min) / 2.0

    @property
    def z_mid(self) -> float:
        return (self.z_min + self.z_max) / 2.0

    @property
    def z_span(self) -> float:
        return self.z_max - self.z_min

    def contains(self, x: float, y: float, z: float, eps: float = 1e-6) -> bool:
        return (
            self.x_min - eps <= x <= self.x_max + eps
            and self.y_min - eps <= y <= self.y_max + eps
            and self.z_min - eps <= z <= self.z_max + eps
        )

    def clamp(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return (
            min(max(x, self.x_min), self.x_max),
            min(max(y, self.y_min), self.y_max),
            min(max(z, self.z_min), self.z_max),
        )


def safe_box_from_config(cfg: Dict) -> SafeBox:
    """由 volume + margin 推導 SafeBox (水平置中)。"""
    v = cfg["volume"]
    m = cfg["margin"]
    half_x = max(0.0, v["size_x"] / 2.0 - m["wall"])
    half_y = max(0.0, v["size_y"] / 2.0 - m["wall"])
    z_min = m["floor"]
    z_max = max(z_min + 1e-3, v["size_z"] - m["ceiling"])
    return SafeBox(-half_x, half_x, -half_y, half_y, z_min, z_max)


DEFAULT_TAKEOFF_POINT = (0.0, 0.0)


def takeoff_point(cfg: Dict) -> Tuple[float, float]:
    """起飛點的水平座標 (房間 ENU, 公尺) —— 飛機實際停放、解鎖起飛的位置。

    設定 `waypoints.takeoff_point`：
      'origin' / None -> (0, 0) 房間原點 (預設, 舊行為)
      [x, y]          -> 指定座標。通常是動捕量到的機體 rigid body 位置
                         (GUI「從 VRPN 讀取」或 `--cli vrpn --save` 寫入)。
    這是全專案唯一的解析點：進場航點、AUTO 估時、安全檢查、匯出的 HOME 列、障礙物可達性錨點
    都要用它，任何地方再寫死 (0,0) 都會讓「規劃的第一段」與「實飛的第一段」不一致。
    格式錯誤或非有限值 (NaN / inf) 時丟 ValueError —— 寧可讓使用者看到錯誤, 也不要靜默退回原點:
    NaN 會一路傳進估時與圈數細修, 讓 plan() 空轉到記憶體耗盡。
    """
    def _finite(x: float, y: float, raw) -> Tuple[float, float]:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"waypoints.takeoff_point 必須是有限數值，得到 {raw!r}")
        return x, y

    v = (cfg.get("waypoints", {}) or {}).get("takeoff_point", "origin")
    if v is None:
        return DEFAULT_TAKEOFF_POINT
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("origin", "home", ""):
            return DEFAULT_TAKEOFF_POINT
        parts = [p for p in t.replace(",", " ").split() if p]
        try:
            if len(parts) == 2:
                return _finite(float(parts[0]), float(parts[1]), v)
        except ValueError:
            if len(parts) == 2:
                raise
        raise ValueError(f"waypoints.takeoff_point 需為 'origin' 或 [x, y]，得到 {v!r}")
    if isinstance(v, dict):
        try:
            return _finite(float(v["x"]), float(v["y"]), v)
        except (KeyError, TypeError) as e:
            raise ValueError(f"waypoints.takeoff_point 的物件需有 x / y，得到 {v!r}") from e
    try:
        seq = [float(t) for t in list(v)[:2]]
    except (TypeError, ValueError) as e:
        raise ValueError(f"waypoints.takeoff_point 需為 'origin' 或 [x, y]，得到 {v!r}") from e
    if len(seq) != 2:
        raise ValueError(f"waypoints.takeoff_point 需為兩個數字 [x, y]，得到 {v!r}")
    return _finite(seq[0], seq[1], v)


def takeoff_is_origin(cfg: Dict) -> bool:
    """起飛點是否就是房間原點 (未設定 / 設成 origin)。"""
    x, y = takeoff_point(cfg)
    return abs(x) < 1e-9 and abs(y) < 1e-9


def describe_takeoff_point(cfg: Dict) -> str:
    """給訊息用的起飛點描述。"""
    x, y = takeoff_point(cfg)
    return f"({x:+.2f}, {y:+.2f}) m" + (" [房間原點]" if takeoff_is_origin(cfg) else " [實測]")


def enu_to_latlon(
    east_m: np.ndarray | float,
    north_m: np.ndarray | float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """本地 ENU 位移 (公尺) -> WGS84 lat/lon (度)。等距切平面近似。"""
    east_m = np.asarray(east_m, dtype=float)
    north_m = np.asarray(north_m, dtype=float)
    dlat = north_m / EARTH_RADIUS_M
    dlon = east_m / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat)))
    lat = origin_lat + np.degrees(dlat)
    lon = origin_lon + np.degrees(dlon)
    return lat, lon


def enu_to_ned(x_e: np.ndarray, y_n: np.ndarray, z_u: np.ndarray):
    """ENU (東,北,上) -> NED (北,東,下)。給 MAV_FRAME_LOCAL_NED 串流用。"""
    north = np.asarray(y_n, dtype=float)
    east = np.asarray(x_e, dtype=float)
    down = -np.asarray(z_u, dtype=float)
    return north, east, down
