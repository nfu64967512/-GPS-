"""
QGC WPL 110 .waypoints 匯出 (AUTO 任務)。

欄位排列與 aeroplan-studio (mission/waypoint.py:to_qgc_line) 一致：
  seq, current, frame, command, p1..p4, lat, lon, alt, autocontinue (以 TAB 分隔)
差異：室內 5 m 等級需要更高座標精度，故 lat/lon 用 8 位小數 (~1.1 mm)，而非戶外的 6 位。

任務序列：HOME -> TAKEOFF -> DO_CHANGE_SPEED -> (NAV_WAYPOINT / NAV_SPLINE_WAYPOINT)*N
          -> LAND / RTL
          預設 (waypoints.repeat=do_jump) 只放「一圈」的航點, 後面接 DO_JUMP(回第一個航點,
          重複 圈數-1 次) 與收尾點 (回起點) 再 LAND —— 任務項數與圈數無關, 上傳快、也不吃
          飛控航點數上限。
          waypoints.turn_in_place.enabled 時, 轉角航點後面緊接 CONDITION_YAW + NAV_DELAY:
          到點停下 -> 原地轉頭朝下一段 -> NAV_DELAY 秒後再前進 (見 core.trajectory.assign_turn_stops)。
座標：把房間本地 ENU 換算成繞「假原點」的 lat/lon (搭配飛控 SET_GPS_GLOBAL_ORIGIN)，
      高度用相對高度 (frame=3)。
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from core.geometry import enu_to_latlon, takeoff_point
from core.planner import PlanResult
from core.trajectory import DEFAULT_FC_BUDGET, mission_polyline  # noqa: F401 (BUDGET 供 GUI/CLI 引用)

# MAVLink 命令 / 座標系 (用整數，輸出格式穩定)
CMD_WAYPOINT = 16
CMD_RTL = 20
CMD_LAND = 21
CMD_TAKEOFF = 22
CMD_SPLINE_WAYPOINT = 82
CMD_NAV_DELAY = 93
CMD_CONDITION_YAW = 115
CMD_DO_JUMP = 177
CMD_DO_CHANGE_SPEED = 178
FRAME_GLOBAL = 0            # 絕對高度 (home 用)
FRAME_GLOBAL_REL_ALT = 3   # 相對 home 高度


def _p(v) -> str:
    """參數格式化：整數印整數、其餘精簡浮點。"""
    return f"{v:g}"


def _line(seq, current, frame, cmd, p1, p2, p3, p4, lat, lon, alt, cont) -> str:
    return (
        f"{seq}\t{current}\t{frame}\t{cmd}\t"
        f"{_p(p1)}\t{_p(p2)}\t{_p(p3)}\t{_p(p4)}\t"
        f"{lat:.8f}\t{lon:.8f}\t{alt:.2f}\t{cont}"
    )


# 航點折線取樣邏輯在 core.trajectory.mission_polyline (與 AUTO 工時預估共用,
# 保證「估時所用折線 == 實際匯出折線」)：曲線沿弧長依 point_spacing 取樣；
# 直線型 pattern 預設以精確轉角為航點、依高度容差 z_tol 精簡 (waypoints.sparse_straight)；
# 圈數重複用 DO_JUMP (waypoints.repeat)。
# 非導航固定項數 (home + takeoff + change_speed + 結束動作) 見 core.trajectory.MISSION_OVERHEAD_ITEMS。


def trajectory_to_waypoints(
    plan: PlanResult, cfg: Dict, mode: str = "precision"
) -> Tuple[List[str], int]:
    """回傳 (QGC WPL 110 行列表, 導航航點數)。

    mode:
      'precision' — 依 point_spacing 取樣，最忠實 (點數可能很多)。
      'compact'   — 保證「總任務項 <= fc_budget」，確保寫得進飛控。
                    若 precision 本來就在預算內，結果與 precision 相同。
    """
    w = cfg["waypoints"]
    origin_lat = float(w["origin_lat"])
    origin_lon = float(w["origin_lon"])
    takeoff_alt = float(w["takeoff_alt"])
    speed = float(cfg["flight"]["cruise_speed"])
    end_action = str(w.get("end_action", "land")).lower()

    mp = mission_polyline(
        plan.trajectory, cfg, plan.pattern.is_smooth,
        lap_xy=plan.pattern.lap_xy, laps=plan.laps, mode=mode,
        repeatable=plan.pattern.repeatable, field=getattr(plan, "obstacles", None),
    )
    lats, lons = enu_to_latlon(mp.x, mp.y, origin_lat, origin_lon)
    zs = mp.z
    a_lats, a_lons = enu_to_latlon(mp.approach_x, mp.approach_y, origin_lat, origin_lon)

    # HOME 列寫「實際起飛點」的經緯度 (未設定時 = 房間原點, 與舊行為逐字元相同)。
    # 飛控真正的 home 由 mocap 橋接的 SET_HOME_POSITION / 解鎖位置決定, 這一列主要是給
    # Mission Planner 的地圖顯示對得上實際停機位置。
    home_x, home_y = takeoff_point(cfg)
    home_lat, home_lon = enu_to_latlon(home_x, home_y, origin_lat, origin_lon)

    lines = ["QGC WPL 110"]
    seq = 0

    # 0: HOME (絕對, current=1) —— Mission Planner 慣例
    lines.append(_line(seq, 1, FRAME_GLOBAL, CMD_WAYPOINT, 0, 0, 0, 0,
                       float(home_lat), float(home_lon), 0.0, 1)); seq += 1
    # 1: TAKEOFF 到 takeoff_alt
    lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_TAKEOFF, 0, 0, 0, 0,
                       0.0, 0.0, takeoff_alt, 1)); seq += 1
    # 2: 設定巡航速度 (param1=1 groundspeed, param2=速度, param3=-1 不改油門)
    lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_DO_CHANGE_SPEED, 1, speed, -1, 0,
                       0.0, 0.0, 0.0, 1)); seq += 1

    def turn(stop):
        """原地轉頭 (waypoints.turn_in_place): 緊接在導航航點後面 -> 到點才開始轉, NAV_DELAY 讓飛機停在該點等轉完。"""
        nonlocal seq
        if stop is None:
            return
        # CONDITION_YAW: p1 絕對航向 (0=北 順時針), p2 角速度 deg/s, p3 方向 (0 最短 / 1 CW / -1 CCW), p4 0=絕對
        lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_CONDITION_YAW,
                           stop.heading_deg, stop.rate_deg_s, stop.direction, 0,
                           0.0, 0.0, 0.0, 1)); seq += 1
        # NAV_DELAY: p1 秒數 (> 0 = 相對延遲), p2..p4 = -1 (不用時:分:秒)
        lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_NAV_DELAY, stop.delay_s, -1, -1, -1,
                           0.0, 0.0, 0.0, 1)); seq += 1

    # 起飛到高度後先原地轉向第一段 (NAV_DELAY 讓飛機懸停等轉完再出發)
    turn(mp.turn_takeoff)

    # 進場段: 起飛點 -> 第一個航點之間為了繞開障礙物插入的一般航點 (只飛一次, 在 DO_JUMP block 之前)
    for i in range(mp.n_approach):
        lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_WAYPOINT, 0, 0, 0, 0,
                           float(a_lats[i]), float(a_lons[i]), float(mp.approach_z[i]), 1)); seq += 1
        turn(mp.turn_approach.get(i))

    # 導航航點
    nav_cmd = CMD_SPLINE_WAYPOINT if mp.use_spline else CMD_WAYPOINT

    def nav(i: int):
        nonlocal seq
        lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, nav_cmd, 0, 0, 0, 0,
                           float(lats[i]), float(lons[i]), float(zs[i]), 1)); seq += 1
        turn(mp.turn_nav.get(i))

    if mp.repeat == "do_jump" and mp.jump_repeat > 0:
        # 一圈 block (不含收尾點) -> DO_JUMP 回 block 第一點 (重複 圈數-1 次) -> 收尾點 (= 起點)
        first_seq = seq
        for i in range(mp.block_len):
            nav(i)
        lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_DO_JUMP, first_seq, mp.jump_repeat, 0, 0,
                           0.0, 0.0, 0.0, 1)); seq += 1
        for i in range(mp.block_len, len(lats)):
            nav(i)
    else:
        for i in range(len(lats)):
            nav(i)
    n_nav = int(len(lats)) + mp.n_approach

    # 結束動作
    if end_action == "rtl":
        lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_RTL, 0, 0, 0, 0,
                           0.0, 0.0, 0.0, 1)); seq += 1
    else:  # land 在最後一點
        lines.append(_line(seq, 0, FRAME_GLOBAL_REL_ALT, CMD_LAND, 0, 0, 0, 0,
                           float(lats[-1]), float(lons[-1]), 0.0, 1)); seq += 1

    return lines, n_nav


def export_waypoints_dual(plan: PlanResult, cfg: Dict, base_path: str) -> Dict:
    """雙通道輸出。

    base_path 例: 'output/circle.waypoints'
      -> 'output/circle.waypoints'          (高精度)
      -> 'output/circle_compact.waypoints'  (精簡, 總任務項 <= fc_budget)

    回傳 {'precision': {...}, 'compact': {...}}，含路徑、nav 數、總任務項數、是否在預算內。
    """
    root, ext = os.path.splitext(base_path)
    budget = int(cfg["waypoints"].get("fc_budget", DEFAULT_FC_BUDGET))
    out = {}
    for mode, p in (("precision", base_path), ("compact", f"{root}_compact{ext}")):
        lines, n_nav = trajectory_to_waypoints(plan, cfg, mode=mode)
        total = len(lines) - 1  # 扣掉 'QGC WPL 110' 標頭
        os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        out[mode] = {
            "path": p, "nav": n_nav, "total_items": total,
            "within_budget": total <= budget,
        }
    out["budget"] = budget
    return out
