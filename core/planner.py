"""
高階規劃入口：pattern -> 圈數 -> 3D 軌跡 -> AUTO 估時 -> 安全檢查。
CLI 與 GUI 都呼叫 plan()。

工時基準 (flight.duration_basis)：
  * auto  (預設) —— 使用者只飛 AUTO (.waypoints)。圈數以「AUTO 航線時間」(compact 版
    航點折線的動態速度剖面, 含進場、不含起飛/降落) 逼近 target_duration 並 >= min_duration；
    GUIDED 軌跡時間僅供參考。
  * guided —— 舊行為：圈數以 GUIDED 串流軌跡 (純水平剖面) 的時間為準。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import patterns
from .altitude import altitude_profile, coprime_cycles
from .config import deep_update
from .geometry import SafeBox, safe_box_from_config
from .obstacles import AvoidInfo, ObstacleField, field_from_config, over_ramp_slopes
from .patterns.base import PatternResult, close_loop
from .safety import SafetyReport, check_trajectory
from .timing import (  # noqa: F401  (basis 工具由此轉出給 GUI/CLI 用)
    auto_nav_seconds,
    basis_duration,
    decide_laps,
    duration_basis,
    estimate_auto_duration,
)
from .trajectory import Trajectory, build_trajectory, mission_polyline


# 圈數細修時「多加一圈至少要讓 AUTO 航線多這麼多秒」, 否則視為不收斂而停止 (見 _refine_laps_by_auto)
MIN_LAP_GAIN_S = 0.5


@dataclass
class PlanResult:
    key: str
    box: SafeBox
    pattern: PatternResult
    trajectory: Trajectory
    laps: int
    report: SafetyReport
    obstacles: Optional[ObstacleField] = None    # 障礙物 + 禁區 (顯示 / 匯出進場段 / GUIDED 進場繞障)
    avoid: Optional[AvoidInfo] = None            # 單圈折線避障統計 (None = 沒有障礙物 / 未啟用)


def mission_repeat(cfg: Dict, repeatable: bool = True) -> str:
    """.waypoints 的圈數重複方式：'do_jump' (預設, 一圈 + DO_JUMP) 或 'unroll' (全展開)。
    pattern 宣告 repeatable=False (例如隨機手飛) 時一律 unroll。"""
    if not repeatable:
        return "unroll"
    r = str(cfg.get("waypoints", {}).get("repeat", "do_jump")).strip().lower()
    return "unroll" if r == "unroll" else "do_jump"


def effective_cycles(cfg: Dict, laps: int, repeatable: bool = True) -> tuple[int, int]:
    """依 圈數 與 重複方式 決定實際的高度起伏次數 -> (全程總次數, 每圈次數; unroll 時每圈為 0=不適用)。

    * do_jump: 每一圈飛的是同一段航點, 高度必須每圈相同 -> 每圈整數次 k = max(1, round(cycles/laps))
      (cycles=0 則 0), 總次數 = k × laps。
    * unroll: 沿用「與圈數互質」的調整, 讓各圈在 3D 中相位錯開、不重疊。
    """
    req = int(round(float(cfg["altitude"].get("cycles", 6))))
    laps = max(1, int(laps))
    if mission_repeat(cfg, repeatable) == "do_jump":
        k = 0 if req <= 0 else max(1, int(round(req / laps)))
        return k * laps, k
    return coprime_cycles(req, laps), 0


def _build(pr: PatternResult, box: SafeBox, cfg: Dict, laps: int,
           modes: tuple = ("precision", "compact"), field: Optional[ObstacleField] = None) -> Trajectory:
    """指定圈數 -> 3D 軌跡 (含高度起伏次數調整) + AUTO 估時 meta (modes 指定要算哪些版本)。"""
    req_cycles = int(round(float(cfg["altitude"].get("cycles", 6))))
    repeat = mission_repeat(cfg, pr.repeatable)
    eff_cycles, per_lap = effective_cycles(cfg, laps, pr.repeatable)
    cfg_eff = cfg
    if eff_cycles != req_cycles:
        cfg_eff = deep_update(cfg, {"altitude": {"cycles": eff_cycles}})

    # do_jump: 高度「每圈」求值 (週期 = 1/圈數) -> 各圈高度嚴格相同 (smooth_random 也是)
    period = (1.0 / laps) if (repeat == "do_jump" and laps > 1) else None

    # 越過低矮箱子: 高度下限 z_floor(s) (平台 + 斜坡, 週期 = 單圈長) -> z = max(剖面, 下限)
    floor = None
    if field is not None and field.over:
        _, _, slope_up, slope_dn = over_ramp_slopes(cfg)
        floor = field.altitude_floor(pr.lap_xy, slope_up, slope_dn)
    profile_holder: Dict = {}

    def alt_fn(t, b, c, s):
        z = altitude_profile(t, b, c, s, period=period)
        if floor is not None:
            profile_holder["profile"] = z.copy()
            z = np.maximum(z, np.minimum(floor(s), b.z_max))
        return z

    traj = build_trajectory(pr.display_name, pr.lap_xy, laps, box, cfg_eff, alt_fn)
    if floor is not None and traj.s is not None:
        traj.meta["z_anchors_s"] = floor.anchors(traj.s, profile_holder.get("profile", traj.z)).tolist()
        traj.meta["over_plateaus"] = [(a, b, zr, i) for a, b, zr, i in floor.plateaus]
    traj.meta["altitude_cycles_requested"] = req_cycles
    traj.meta["altitude_cycles_effective"] = eff_cycles
    traj.meta["altitude_cycles_per_lap"] = per_lap          # do_jump 才有意義 (0 = 不適用)
    traj.meta["mission_repeat"] = repeat
    # AUTO (.waypoints) 任務時間預估 —— 對實際匯出的航點折線估時, 含起飛/降落
    traj.meta["auto_estimate"] = {
        m: estimate_auto_duration(traj, cfg, pr.is_smooth, mode=m,
                                  lap_xy=pr.lap_xy, laps=laps, repeatable=pr.repeatable, field=field)
        for m in modes
    }
    return traj


def _ensure_estimates(traj: Trajectory, pr: PatternResult, cfg: Dict, laps: int,
                      field: Optional[ObstacleField] = None) -> Trajectory:
    """補齊 precision / compact 兩種估時 (圈數細修時候選只算 compact 以省時間)。"""
    est = traj.meta.setdefault("auto_estimate", {})
    for m in ("precision", "compact"):
        if m not in est:
            est[m] = estimate_auto_duration(traj, cfg, pr.is_smooth, mode=m,
                                            lap_xy=pr.lap_xy, laps=laps, repeatable=pr.repeatable,
                                            field=field)
    return traj


def _refine_laps_by_auto(
    pr: PatternResult, box: SafeBox, cfg: Dict, laps0: int, traj0: Trajectory,
    field: Optional[ObstacleField] = None,
) -> tuple[int, Trajectory]:
    """duration_basis=auto：以 AUTO 航線時間逼近 target_duration、且 >= min_duration。

    先用 GUIDED 基準的圈數 laps0 當起點 (AUTO 只會更長, 故最佳解通常是 laps0 或少一圈)：
    由兩個圈數的 AUTO 時間差求每圈時間 -> 線性推 n* -> 在 n* 鄰域取最貼近 target 者
    (與 timing.decide_laps 同一套「粗估後細修」邏輯)。每個候選都真的建軌跡估時，
    高度起伏的互質調整等效應一併涵蓋。
    """
    f = cfg["flight"]
    target = float(f["target_duration"])
    min_dur = float(f.get("min_duration", 200))
    cache: Dict[int, Trajectory] = {laps0: traj0}

    def nav(n: int) -> float:
        if n not in cache:
            cache[n] = _build(pr, box, cfg, n, modes=("compact",), field=field)   # 候選只算上飛控版
        return auto_nav_seconds(cache[n]) or 0.0

    n1 = laps0 - 1 if laps0 > 1 else laps0 + 1
    per_lap = max(abs(nav(laps0) - nav(n1)) / abs(laps0 - n1), 1e-6)
    intercept = nav(laps0) - laps0 * per_lap
    n_star = max(1, int(round((target - intercept) / per_lap)))
    # 只再看「target 另一側」的那個鄰居 (少建一條軌跡; 每建一次含估時約 0.1~0.4 s)
    side = n_star + 1 if nav(n_star) < target else n_star - 1
    if side >= 1:
        nav(side)

    ok = [n for n in cache if nav(n) >= min_dur - 1e-6]
    pool = ok if ok else list(cache)
    best = min(pool, key=lambda n: (abs(nav(n) - target), n))
    # 全部候選都不足 -> 往上加圈。加圈若已經無法讓 AUTO 航線變長就停 —— fc_budget 設得極小時
    # compact 折線會被壓到只剩十幾點, 航線時間對圈數幾乎不變, 這個迴圈會發散 (每一步都要建一條軌跡,
    # 介面會像當掉)。停下來後工時不足會由 safety 的工時檢查照常警告。
    while nav(best) < min_dur - 1e-6:
        prev = nav(best)
        best += 1
        if nav(best) - prev < MIN_LAP_GAIN_S:
            break
    return best, _ensure_estimates(cache[best], pr, cfg, best, field)


def apply_obstacles(pr: PatternResult, field: ObstacleField) -> Optional[AvoidInfo]:
    """對 pattern 單圈折線做水平避障 (就地改 pr.lap_xy)。沒有障礙物 / 未啟用時回 None。"""
    if not field.active:
        return None
    lap, info = field.avoid_lap(pr.lap_xy, pr.is_smooth)
    pr.lap_xy = close_loop(lap)
    return info


def plan(key: str, cfg: Dict) -> PlanResult:
    box = safe_box_from_config(cfg)
    pr = patterns.generate(key, box, cfg)
    field = field_from_config(cfg, box)             # 障礙物 (箱子) -> 禁區; 沒設就是空的
    avoid = apply_obstacles(pr, field)              # 單圈折線先繞開障礙物, 之後圈數/高度/匯出流程不變
    laps, _ = decide_laps(pr.lap_xy, cfg)          # GUIDED 基準 (純水平剖面) 的圈數
    laps_cfg = cfg["flight"].get("laps", "auto")
    laps_is_auto = not (isinstance(laps_cfg, int)
                        or (isinstance(laps_cfg, str) and laps_cfg.isdigit()))
    refine = laps_is_auto and duration_basis(cfg) == "auto"

    traj = _build(pr, box, cfg, laps, modes=("compact",) if refine else ("precision", "compact"),
                  field=field)
    if refine:
        laps, traj = _refine_laps_by_auto(pr, box, cfg, laps, traj, field)

    traj.meta["obstacles"] = {
        "count": len(field.obstacles),
        "clearance": field.clearance,
        "enabled": field.enabled,
        "description": field.describe() if field.obstacles else "",
        "avoid": avoid.as_dict() if avoid else None,
    }
    mission = None
    if field.obstacles:
        # 安全檢查用: 實際會飛的 AUTO 航點折線 (compact 版, 含起飛進場段)
        mission = mission_polyline(traj, cfg, pr.is_smooth, lap_xy=pr.lap_xy, laps=laps,
                                   mode="compact", repeatable=pr.repeatable, field=field)
    report = check_trajectory(traj, box, cfg, obstacles=field, mission=mission, avoid=avoid)
    return PlanResult(key=key, box=box, pattern=pr, trajectory=traj, laps=laps, report=report,
                      obstacles=field, avoid=avoid)
