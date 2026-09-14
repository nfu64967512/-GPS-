"""
工時規劃與預估。

預估精度層級：
  * naive_duration —— 水平弧長 / 巡航速度 (舊模型)。忽略加減速、轉角減速與
    jerk，一律低估；保留做對照與 speed_profile=constant 模式。
  * 動態速度剖面 (core/speed_profile) —— 加減速 + 硬轉角 (5 cm 接受半徑) +
    曲率 + S 曲線 jerk。GUIDED 串流的軌跡即依此重取時，因此
    「預估工時 == 串流時長 == 實飛時長 (追蹤正常時)」。
  * estimate_auto_duration —— AUTO 任務：對「實際會匯出的航點折線 (3D)」跑
    同一套剖面，並加上起飛/降落，對應上傳 .waypoints 切 AUTO 的整段任務時間。

decide_laps: 依單圈折線幾何與目標工時決定圈數，使預估工時盡量落在
[min_duration, max_duration]、貼近 target_duration (GUIDED 基準的粗估；
flight.duration_basis=auto 時 planner 會再以 AUTO 航線時間細修)。

工時基準 (flight.duration_basis): auto (預設, 只飛 AUTO 的使用者) 以 AUTO 航線時間
判定 200~300 s 需求；guided 沿用 GUIDED 軌跡時間。見 duration_basis / basis_duration。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .speed_profile import ProfileParams, compute_profile
from .trajectory import arclength, mission_polyline, tile_closed

# AUTO 工時預估的固定餘裕 (s)：起飛含起轉/爬升暫態、降落含觸地判定
_TAKEOFF_MARGIN_S = 2.0
_LAND_MARGIN_S = 2.0


def duration_basis(cfg: Dict) -> str:
    """工時基準：'auto' (預設) 或 'guided'。"""
    b = str(cfg.get("flight", {}).get("duration_basis", "auto")).strip().lower()
    return "guided" if b == "guided" else "auto"


def auto_nav_seconds(traj) -> Optional[float]:
    """AUTO (compact / 上飛控版) 航線時間 (s, 含進場、不含起飛/降落)；沒有估時資料時回 None。"""
    est = (traj.meta.get("auto_estimate") or {}).get("compact")
    return float(est["nav_s"]) if est else None


def basis_duration(traj, cfg: Dict) -> float:
    """依 duration_basis 回傳「拿來判定工時需求」的秒數 (auto -> AUTO 航線; guided -> 軌跡)。"""
    if duration_basis(cfg) == "auto":
        nav = auto_nav_seconds(traj)
        if nav is not None:
            return nav
    return traj.duration


def naive_duration(lap_length_m: float, laps: int, speed: float) -> float:
    """舊模型：純飛行工時 = 水平弧長 / 定速 (低估, 對照用)。"""
    if speed <= 0:
        return 0.0
    return lap_length_m * laps / speed


# 向後相容舊名
predicted_duration = naive_duration


def decide_laps(lap_xy: np.ndarray, cfg: Dict) -> Tuple[int, float]:
    """回傳 (laps, 預估純飛行工時 s)。

    cfg.flight.laps = 'auto' 時自動求圈數；給整數則直接採用。
    speed_profile=dynamic (預設) 時以動態速度剖面預估 (含轉角減速)，
    constant 時沿用舊定速公式。
    """
    f = cfg["flight"]
    lap_xy = np.asarray(lap_xy, dtype=float)
    lap_len = arclength(lap_xy)
    speed = float(f["cruise_speed"])
    target = float(f["target_duration"])
    laps_cfg = f.get("laps", "auto")
    dynamic = str(f.get("speed_profile", "dynamic")).lower() != "constant"
    prm = ProfileParams.from_config(cfg)

    def dur(n: int) -> float:
        if dynamic:
            return compute_profile(tile_closed(lap_xy, n), prm).duration
        return naive_duration(lap_len, n, speed)

    if isinstance(laps_cfg, int) or (isinstance(laps_cfg, str) and laps_cfg.isdigit()):
        laps = max(1, int(laps_cfg))
        return laps, dur(laps)

    if lap_len <= 0 or speed <= 0:
        return 1, 0.0

    if dynamic:
        # T(n) ≈ T1 + (n-1)·T_lap；T_lap 由 2 圈與 1 圈之差求得 (含接縫轉角)
        t1 = dur(1)
        per_lap = max(dur(2) - t1, 1e-6)
        n = max(1, round((target - (t1 - per_lap)) / per_lap))
        # 鄰域內取最貼近 target 者 (剖面非線性, 粗估後細修)
        n = min({max(1, n - 1), n, n + 1},
                key=lambda k: abs(dur(k) - target))
    else:
        n = max(1, round(target * speed / lap_len))

    min_dur = float(f.get("min_duration", 200))
    while dur(n) < min_dur:
        n += 1
    return n, dur(n)


def estimate_auto_duration(
    traj, cfg: Dict, is_smooth: bool, mode: str = "compact",
    lap_xy=None, laps: int = 1, repeatable: bool = True, field=None,
) -> Dict:
    """預估 AUTO (.waypoints 上傳) 任務時間。

    用與 io_export.waypoints 相同的取樣規則 (core.trajectory.mission_polyline) 重建
    航點折線 (3D)：曲線沿弧長取樣、直線型 pattern 以精確轉角為航點 (需給 lap_xy/laps)。
    前面接上起飛點 (waypoints.takeoff_point, takeoff_alt) 當作進場段 —— 由 MissionPath.home 帶入,
    因此估的是「實際會飛的第一段」；再跑動態速度剖面 (接受半徑 = 5 cm 級
    waypoints.accept_radius)，再加簡化的起飛/降落時間。

    回傳 {mode, n_wp (寫進任務的導航航點數, 含進場), n_approach, n_items (航點 + DO_JUMP), repeat,
          jump_repeat, takeoff_s, nav_s, land_s, total_s, hard_corners}。nav_s 以「實際會飛的完整折線」
    (do_jump = 一圈 block 重複 圈數 次) 估算；field (障礙物) 給了就含繞障的進場航點。
    """
    w = cfg["waypoints"]
    mp = mission_polyline(traj, cfg, is_smooth, lap_xy=lap_xy, laps=laps, mode=mode,
                          repeatable=repeatable, field=field)
    takeoff_alt = float(w.get("takeoff_alt", 1.0))
    xs, ys, zs = mp.flown_with_approach(takeoff_alt)
    poly = np.column_stack([xs, ys, zs])
    prm = ProfileParams.from_config(cfg)
    # 原地轉頭 (waypoints.turn_in_place): 轉頭點後面接 NAV_DELAY -> 飛控不做 fast waypoint, 在該點
    # 減速到 0、懸停 NAV_DELAY 秒 (轉頭) 再從靜止出發 -> 在這些點把折線切開各自估 (靜止->靜止) + 懸停時間
    stops = mp.flown_turn_stops()
    cuts = [0] + [f for f, _ in stops] + [len(poly) - 1]
    fly_s, hard = 0.0, 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            prof = compute_profile(poly[a:b + 1], prm)
            fly_s += prof.duration
            hard += prof.hard_corners
    turn_s = float(sum(d for _, d in stops))
    nav_s = fly_s + turn_s

    t_takeoff = takeoff_alt / max(prm.vz_up, 0.05) + _TAKEOFF_MARGIN_S
    land_speed = float(w.get("land_speed", 0.4))
    t_land = float(zs[-1]) / max(land_speed, 0.05) + _LAND_MARGIN_S
    return {
        "mode": mode,
        "n_wp": mp.n_nav,
        "n_approach": mp.n_approach,
        "n_items": mp.n_items,
        "repeat": mp.repeat,
        "jump_repeat": mp.jump_repeat,
        "block_len": mp.block_len,
        "takeoff_s": t_takeoff,
        "nav_s": nav_s,
        "land_s": t_land,
        "total_s": t_takeoff + nav_s + t_land,
        "hard_corners": hard,
        # 原地轉頭: n_turns = 任務裡的轉頭點 (各 CONDITION_YAW + NAV_DELAY 兩項), turn_stops = 實飛停下次數
        # (DO_JUMP 每圈都停), turn_s = 懸停轉頭總秒數 (已含在 nav_s), turn_points = 轉頭點位置 (預覽用)
        "n_turns": mp.n_turns,
        "turn_stops": len(stops),
        "turn_s": turn_s,
        "turn_points": mp.turn_points(takeoff_alt).tolist(),
    }
