"""
CSV 匯出：完整時間序列軌跡，給 VIO / 預測模型當 ground truth。

欄位 (SI 單位)：
  t                時間 (s)
  x_e, y_n, z_u    本地 ENU 位置 (m); x=東, y=北, z=上(相對地板)
  vx, vy, vz       ENU 速度 (m/s)
  yaw_deg          機頭朝向 (deg, 0=+x 東, 逆時針)
  north, east, down  NED 位置 (m)
  lat, lon, rel_alt  繞假原點的 WGS84 (deg, deg, m) — 對齊 .waypoints
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from core.geometry import enu_to_latlon, enu_to_ned
from core.planner import PlanResult


def export_csv(plan: PlanResult, cfg: Dict, path: str) -> int:
    """寫出 CSV，回傳列數 (取樣點數)。"""
    t = plan.trajectory
    w = cfg["waypoints"]
    lat, lon = enu_to_latlon(t.x, t.y, float(w["origin_lat"]), float(w["origin_lon"]))
    north, east, down = enu_to_ned(t.x, t.y, t.z)
    yaw_deg = np.degrees(t.yaw)

    cols = np.column_stack([
        t.t, t.x, t.y, t.z, t.vx, t.vy, t.vz, yaw_deg,
        north, east, down, lat, lon, t.z,
    ])
    header = ("t,x_e,y_n,z_u,vx,vy,vz,yaw_deg,"
              "north,east,down,lat,lon,rel_alt")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savetxt(path, cols, delimiter=",", header=header, comments="",
               fmt=["%.3f", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f", "%.2f",
                    "%.4f", "%.4f", "%.4f", "%.8f", "%.8f", "%.4f"])
    return len(t.t)
