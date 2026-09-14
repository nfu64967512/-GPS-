"""
安全檢查：超界、超速、工時不足、航點數過多。

回傳 SafetyReport (ok + 訊息列表)，GUI / CLI 會顯示。errors 會擋下飛行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .geometry import SafeBox, takeoff_point
from .obstacles import (CLEARANCE_SLACK, CLIMB_OVERSHOOT, drone_radius, drone_size, marker_height,
                        pivot_above, pivot_below, points_inside, recommended_clearance,
                        recommended_vertical_clearance, segments_enter, signed_distance)
from .timing import basis_duration, duration_basis
from .trajectory import YAW_SLEW_DEFAULT_DEG_S, Trajectory, TurnInPlace

# 設定鍵缺省時的後備預設 (與 config/default.yaml 一致; 集中定義避免各檔字面量分歧)
DEFAULT_MAX_WAYPOINTS = 1000
# 註: 爬升柱的頂端 = 起飛高度 + 機身半高 + 超調餘裕 (見 core.obstacles.CLIMB_OVERSHOOT),
#     由機身尺寸推導, 不再用固定值。
# 量航點折線離障礙物距離時的取樣步長 (m)。只看航點會漏掉弦的中段; 步長太粗則會漏掉「擦過箱角」——
# 5 cm 的弦切過直角時中點可深入數 mm, 實測會被漏報, 故取 2 cm。
DISTANCE_SAMPLE_STEP = 0.02


@dataclass
class SafetyReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def as_lines(self) -> List[str]:
        out = [f"[ERROR] {m}" for m in self.errors]
        out += [f"[WARN]  {m}" for m in self.warnings]
        if not out:
            out = ["[OK] 通過所有安全檢查"]
        return out


def check_trajectory(
    traj: Trajectory, box: SafeBox, cfg: Dict, n_waypoints: int | None = None,
    obstacles=None, mission=None, avoid=None,
) -> SafetyReport:
    """對已建好的軌跡做安全檢查。

    obstacles: core.obstacles.ObstacleField (None = 不檢查障礙物)。
    mission:   core.trajectory.MissionPath (compact 版, 實際會飛的 AUTO 航點折線；含進場段) —— 也對它量
               離障礙物的距離 (曲線 pattern 的航點是弦, 可能比軌跡更靠近障礙物)。
    avoid:     core.obstacles.AvoidInfo (單圈避障統計; failed > 0 = 有線段繞不開)。
    """
    r = SafetyReport()
    f = cfg["flight"]

    # 1) 邊界
    out = ~np.array(
        [box.contains(x, y, z) for x, y, z in zip(traj.x, traj.y, traj.z)]
    )
    if out.any():
        b = traj.bounds()
        r.errors.append(
            f"{int(out.sum())} 個點超出安全盒 "
            f"(x={b['x'][0]:.2f}~{b['x'][1]:.2f}, "
            f"y={b['y'][0]:.2f}~{b['y'][1]:.2f}, "
            f"z={b['z'][0]:.2f}~{b['z'][1]:.2f})"
        )

    # stair 高度模式的垂直速度由 altitude.ramp_speed 決定 (auto 時自動低於各上限)，
    # 升降時 3D 合成速度也會明顯變大，提示需一併指向它
    alt_mode = str(cfg.get("altitude", {}).get("mode", "sine"))

    # 2) 速度 (3D 合成)
    max_speed = float(f.get("max_speed", 1.0))
    if traj.max_speed > max_speed + 1e-3:
        v_hint = (
            "請降低 cruise_speed 或 altitude.ramp_speed (階梯升降時合成速度 = √(水平²+垂直²))"
            if alt_mode == "stair" else "請降低 cruise_speed 或放大軌跡"
        )
        r.errors.append(
            f"最大速度 {traj.max_speed:.2f} m/s 超過上限 {max_speed:.2f} m/s ({v_hint})"
        )

    # 1b) 起飛點必須在房間內 (在安全盒外、房間內是合法的: 飛機常停在牆邊)。
    #     量錯單位 / axes / offset 打錯一個位數時, 幽靈般的長進場段會吃掉工時預算又不會有人發現。
    tx, ty = takeoff_point(cfg)
    v = cfg["volume"]
    hx, hy = float(v["size_x"]) / 2.0, float(v["size_y"]) / 2.0
    if abs(tx) > hx + 1e-6 or abs(ty) > hy + 1e-6:
        r.errors.append(
            f"起飛點 HOME ({tx:+.2f}, {ty:+.2f}) m 在房間外 "
            f"(房間 {float(v['size_x']):.1f}×{float(v['size_y']):.1f} m); "
            f"請檢查 waypoints.takeoff_point, 或 obstacles.vrpn 的 axes / offset / 單位"
        )

    # 2b) 垂直速度可行性 (WPNAV_SPEED_UP / _DN 等級)
    if traj.n > 1:
        vz_up = float(f.get("speed_up", 1.0))
        vz_dn = float(f.get("speed_down", 0.6))
        vzmax = float(np.max(traj.vz))
        vzmin = float(np.min(traj.vz))
        vz_hint = (
            "降低 altitude.ramp_speed，或減少 altitude.cycles/amplitude、增加工時"
            if alt_mode == "stair"
            else "減少 altitude.cycles/amplitude 或增加工時"
        )
        if vzmax > vz_up + 0.05:
            r.warnings.append(
                f"上升速度 {vzmax:.2f} m/s 超過上限 {vz_up:.2f} m/s ({vz_hint})"
            )
        if -vzmin > vz_dn + 0.05:
            r.warnings.append(
                f"下降速度 {-vzmin:.2f} m/s 超過上限 {vz_dn:.2f} m/s ({vz_hint})"
            )

    # 2c) 水平加速度可行性 (定速模式在轉角需要瞬間變向, 實機必跟不上)。
    #     轉角尖點由 5cm 接受半徑的切角吸收, 故量測「持續」加速度:
    #     先對速度做 ~0.5s 移動平均 (≈實機通過圓角的時間尺度) 再差分。
    if traj.n > 3:
        accel_lim = float(f.get("accel", 1.0))
        lat_lim = float(f.get("lat_accel", accel_lim))
        a_h = sustained_accel_p98(traj)
        a_allow = 1.3 * float(np.hypot(accel_lim, lat_lim))
        if a_h > a_allow:
            hint = (
                "flight.speed_profile 目前為 constant, 轉角處需要瞬間變向; "
                "建議改 dynamic"
                if str(f.get("speed_profile", "dynamic")).lower() == "constant"
                else "請降低 cruise_speed 或提高 flight.accel/lat_accel"
            )
            r.warnings.append(
                f"軌跡需求水平加速度 (P98) {a_h:.1f} m/s² 超過可行範圍 "
                f"{a_allow:.1f} m/s² ({hint})"
            )

    # 3) 工時 (依 flight.duration_basis: auto -> AUTO 航線時間; guided -> 軌跡時間)
    min_dur = float(f.get("min_duration", 200))
    max_dur = float(f.get("max_duration", 300))
    dur = basis_duration(traj, cfg)
    label = "AUTO 航線工時" if duration_basis(cfg) == "auto" else "純飛行工時"
    if dur < min_dur:
        r.warnings.append(
            f"{label} {dur:.0f} s 低於下限 {min_dur:.0f} s "
            f"(增加圈數或降低速度)"
        )
    elif dur > max_dur:
        r.warnings.append(
            f"{label} {dur:.0f} s 高於建議上限 {max_dur:.0f} s"
        )

    # 4) 航點數 (僅在有提供時)
    if n_waypoints is not None:
        max_wp = int(cfg.get("waypoints", {}).get("max_waypoints", DEFAULT_MAX_WAYPOINTS))
        if n_waypoints > max_wp:
            r.warnings.append(
                f"航點數 {n_waypoints} 超過建議上限 {max_wp} "
                f"(放大 point_spacing、開啟 use_spline，或改用 GUIDED 串流)"
            )

    # 4a) 原地轉頭 (waypoints.turn_in_place): CONDITION_YAW 的角速度會被飛控上限 (ATC_SLEW_YAW) 截斷
    tip = TurnInPlace.from_config(cfg)
    if tip.enabled and tip.rate_deg_s > YAW_SLEW_DEFAULT_DEG_S + 1e-9:
        r.warnings.append(
            f"原地轉頭角速度 {tip.rate_deg_s:.0f} deg/s 超過 ArduCopter 預設上限 {YAW_SLEW_DEFAULT_DEG_S:.0f} deg/s "
            f"(ATC_SLEW_YAW / 新版 ATC_RATE_WPY_MAX): 飛控會限速, NAV_DELAY 已依 "
            f"{YAW_SLEW_DEFAULT_DEG_S:.0f} deg/s 估; 要真的轉更快需一併調高飛控參數"
        )

    # 4b) 機身尺寸 vs 安全邊界 (規劃的是機身『中心』的路徑, 會撞到的是離中心最遠的角)
    check_airframe(r, cfg, obstacles)

    # 5) 障礙物 (箱子): 軌跡與 AUTO 航點折線都不可進入障礙物底面, 且應保持安全距離 (只繞不越)
    if obstacles is not None and getattr(obstacles, "obstacles", None):
        check_obstacles(r, traj, cfg, obstacles, mission, avoid)

    return r


def check_airframe(r: SafetyReport, cfg: Dict, field=None) -> None:
    """機身裝不裝得下：離牆邊界與障礙物安全距離都必須大於機身半徑。

    規劃出來的是機身中心的路徑；離牆 0.5 m 對 50×50 cm 的機身 (半對角 0.354 m) 只剩 15 cm,
    對 80×80 cm 的機身 (半對角 0.566 m) 就已經是機身穿牆了。這個檢查把「安全距離是誰算出來的」
    從使用者的腦袋搬進程式。
    """
    try:
        sx, sy, sz = drone_size(cfg)
        rad = drone_radius(cfg)
        below, above = pivot_below(cfg), pivot_above(cfg)
        mh = marker_height(cfg)
    except Exception as e:  # noqa: BLE001  (設定格式錯誤)
        r.errors.append(f"機身尺寸設定錯誤: {e}")
        return
    ref = (f"光球貼頂板, 參考點離地 {mh:.2f} m" if mh is not None else "假設參考點在機身中心")
    body = (f"機身 {sx * 100:.0f}×{sy * 100:.0f}×{sz * 100:.0f} cm "
            f"(半對角 {rad:.2f} m; {ref}: 規劃高度下方 {below:.2f} m、上方 {above:.2f} m)")

    # 垂直: 規劃的是參考點的高度, 機體在它上下各佔一段, 兩邊分別檢查
    for key, label, need in (("floor", "離地", below), ("ceiling", "離天花板", above)):
        m = float(cfg.get("margin", {}).get(key, 0.0))
        if m < need - 1e-9:
            r.errors.append(
                f"{body}: margin.{key} ({label}) {m:.2f} m 小於規劃高度{'下' if key == 'floor' else '上'}方的 "
                f"{need:.2f} m; 軌跡貼到安全盒上下界時機身已經穿出去, 請把 margin.{key} 加大"
            )
        elif m < need + 0.10:
            r.warnings.append(
                f"{body}: 貼到安全盒{label}邊界時只剩 {m - need:.2f} m; "
                f"margin.{key} 建議 >= {need + 0.10:.2f} m"
            )

    wall = float(cfg.get("margin", {}).get("wall", 0.0))
    if wall < rad - 1e-9:
        r.errors.append(
            f"{body} 比離牆邊界 {wall:.2f} m 還寬: 軌跡貼到安全盒邊時機身已經在牆外; "
            f"請把 margin.wall 加到 {rad:.2f} m 以上"
        )
    elif wall < rad + 0.05:
        r.warnings.append(
            f"{body} 貼到安全盒邊時離牆只剩 {wall - rad:.2f} m; margin.wall 建議 >= {rad + 0.05:.2f} m"
        )

    if field is None or not getattr(field, "obstacles", None):
        return
    # 越過箱子的垂直安全距離
    if getattr(field, "over", None):
        vclr = float(field.vertical_clearance)
        vrec = recommended_vertical_clearance(cfg)
        if vclr < below - 1e-9:
            r.errors.append(
                f"垂直安全距離 {vclr:.2f} m 小於規劃高度下方的機體 {below:.2f} m: "
                f"越過箱子時機身底部已經碰到箱頂; 請把 obstacles.vertical_clearance 加大"
            )
        elif vclr < vrec - CLEARANCE_SLACK:
            r.warnings.append(
                f"垂直安全距離 {vclr:.2f} m 低於建議的 {vrec:.2f} m "
                f"(= 規劃高度下方機體 {below:.2f} + 下洗/追蹤餘裕 0.15); 越過輕紙箱時容易把它吹動"
            )
    clr = float(field.clearance)
    rec = recommended_clearance(cfg)
    if clr < rad - 1e-9:
        r.errors.append(
            f"障礙物安全距離 {clr:.2f} m 小於{body}: 路徑中心離箱子 {clr:.2f} m 時機身已經碰到了; "
            f"請把 obstacles.clearance 加到 {rec:.2f} m 以上 (或設成 auto)"
        )
    elif clr < rec - CLEARANCE_SLACK:
        r.warnings.append(
            f"障礙物安全距離 {clr:.2f} m 低於建議的 {rec:.2f} m "
            f"(= {body} + 航點切角 {float(cfg.get('waypoints', {}).get('accept_radius', 0.05)):.2f} "
            f"+ 追蹤餘裕 0.10); 只剩 {clr - rad:.2f} m 吸收定位誤差"
        )


def check_obstacles(r: SafetyReport, traj: Trajectory, cfg: Dict, field, mission=None, avoid=None) -> None:
    """障礙物距離檢查 (以『原始』底面量; 規劃用的禁區已外擴 clearance)。

    起飛點取自 waypoints.takeoff_point (實際停機位置)，不是房間原點 —— 進場段、起飛爬升與 RTL
    的檢查都以它為準。
    """
    clr = float(field.clearance)
    tol = 0.02                                   # 禁區圓角以外接多邊形近似, 留 2 cm 容差
    names = [o.name for o in field.obstacles]
    over_idx = {z.index for z in getattr(field, "over", [])}
    tx, ty = takeoff_point(cfg)
    d_traj = field.distances(np.column_stack([traj.x, traj.y]))
    d_home = field.distances(np.array([[tx, ty]]))
    k_home = int(np.argmin(d_home))
    home_inside = d_home[k_home] < 0
    d_mis = None
    mis_xyz = None
    mis_xy = None
    if mission is not None:
        takeoff_alt = float(cfg.get("waypoints", {}).get("takeoff_alt", 1.0))
        mx, my, mz = mission.flown_with_approach(takeoff_alt)
        # 不能因為起飛點在某個障礙物裡就把整條第一段丟掉 —— 那一段可能還穿過『別的』障礙物。
        # 起飛點所在的那個障礙物由 check_takeoff_climb 負責報告 (下面的迴圈會跳過它)。
        mis_xy = np.column_stack([mx, my])
        mis_xyz = _densify_xyz(mx, my, mz, DISTANCE_SAMPLE_STEP)
        d_mis = field.distances(mis_xyz[:, :2])

    def enters(poly) -> bool:
        """軌跡或航點折線是否有『線段』真的切進這個底面 —— 精確判定, 不靠取樣密度。
        擦過箱角只切進幾 mm 時, 取樣可能整段跳過, 但這種路徑實飛就是會碰到。"""
        for pts in (np.column_stack([traj.x, traj.y]), mis_xy):
            if pts is None or len(pts) < 2:
                continue
            if segments_enter(pts[:-1], pts[1:], poly).any():
                return True
        return False
    dis_hint = "" if field.enabled else " (避障已停用: obstacles.enabled)"
    # 越過的低矮箱子: 水平可以穿過其禁區, 但禁區內高度必須 >= 箱頂 + 垂直安全距離
    for z in getattr(field, "over", []):
        name = names[z.index]
        z_top = field.obstacles[z.index].z_top
        # 軌跡容差 3 cm; AUTO 航點折線是弦, 曲線處會略切進禁區邊緣 (斜坡高度), 容差放寬到 10 cm
        for label, xyz, tol_z in (("軌跡", np.column_stack([traj.x, traj.y, traj.z]), 0.03),
                                  ("AUTO 航點折線 (含起飛進場段)", mis_xyz, 0.10)):
            if xyz is None or len(xyz) == 0:
                continue
            inside = points_inside(xyz[:, :2], z.poly)
            if not inside.any():
                continue
            zmin = float(xyz[inside, 2].min())
            if zmin < z_top + pivot_below(cfg) + CLIMB_OVERSHOOT:
                r.errors.append(
                    f"{label}越過障礙物「{name}」時離箱頂只有 {zmin - z_top:.2f} m (箱頂 {z_top:.2f} m); "
                    f"請加大垂直安全距離或改為繞開 (obstacles.low_mode: around)"
                )
            elif zmin < z.z_req - tol_z:
                r.warnings.append(
                    f"{label}越過障礙物「{name}」時最低 {zmin:.2f} m < 箱頂+垂直安全距離 {z.z_req:.2f} m"
                )
    for i, name in enumerate(names):
        if i in over_idx:
            continue
        if home_inside and i == k_home:
            continue                             # 起飛點就在它裡面 -> 由 check_takeoff_climb 報告
        dt = d_traj[i]
        dm = d_mis[i] if d_mis is not None else np.inf
        worst = min(dt, dm)
        where = "軌跡" if dt <= dm else "AUTO 航點折線 (含起飛進場段)"
        if worst >= 0 and enters(field.obstacles[i].footprint):
            worst = 0.0                          # 精確判定說有切進去, 但取樣點都落在外面 (擦角)
        if worst <= 0:
            depth = f"最深 {-worst:.2f} m" if worst < 0 else "擦過箱角"
            r.errors.append(
                f"{where}穿過障礙物「{name}」({depth}){dis_hint}"
                + ("" if field.enabled else "；請開啟避障或移動障礙物")
            )
        elif worst < clr - tol:
            if where.startswith("AUTO") and mission is not None and mission.use_spline:
                hint = ("SPLINE 航點間的弦會切角, 實飛樣條介於弦與規劃路徑之間; "
                        "要保證距離請加大安全距離或縮小航點間距 / spline 倍率")
            elif where.startswith("AUTO"):
                hint = "航點間直線會切角; 請加大安全距離或縮小航點間距"
            else:
                hint = "請加大安全距離或檢查障礙物設定"
            r.warnings.append(
                f"{where}離障礙物「{name}」最近 {worst:.2f} m < 安全距離 {clr:.2f} m ({hint})"
            )
    if avoid is not None and getattr(avoid, "unreachable", 0):
        r.warnings.append(
            f"{avoid.unreachable} 個路徑點落在障礙物與牆之間、從 HOME 到不了的區域, 已移到可達的禁區邊界 "
            f"(路徑形狀改變; 若非預期請移動障礙物或縮小安全距離)"
        )
    if avoid is not None and getattr(avoid, "failed", 0):
        r.errors.append(
            f"{avoid.failed} 段路徑無法在安全盒內繞開障礙物 (障礙物擋住整個通道; "
            f"請移動障礙物、縮小安全距離 {clr:.2f} m 或離牆邊界)"
        )
    # 起飛垂直爬升 (原地爬到起飛高度)
    check_takeoff_climb(r, cfg, field, tx, ty)

    # RTL: 直線飛回 HOME, 不會避障。飛控的 HOME 可能是解鎖位置 (= 起飛點), 也可能是 mocap 橋接
    # 用 SET_HOME_POSITION 設在房間原點 —— 兩者不同時兩條都檢查, 由使用者去對飛控確認是哪一個。
    end_action = str(cfg.get("waypoints", {}).get("end_action", "land")).lower()
    if end_action == "rtl" and field.zones and traj.n:
        last = (float(traj.x[-1]), float(traj.y[-1]))
        targets = [((tx, ty), "起飛點")]
        if abs(tx) > 1e-9 or abs(ty) > 1e-9:
            targets.append(((0.0, 0.0), "房間原點"))
        for tgt, label in targets:
            if field.blocked(last, tgt):
                r.warnings.append(
                    f"結束動作 RTL 會直線飛回 HOME，若 HOME 是{label} ({tgt[0]:+.2f}, {tgt[1]:+.2f}) "
                    f"則途中有障礙物 (RTL 不避障); 建議改用 land，或確認飛控的 HOME 實際在哪"
                )


def check_takeoff_climb(r: SafetyReport, cfg: Dict, field, tx: float, ty: float) -> None:
    """起飛垂直爬升檢查：飛機在起飛點原地爬到 waypoints.takeoff_alt，途中不可撞到箱子。

    只看水平距離不夠 —— 箱子可能疊高或放在桌上 (例如底 0.92 m、頂 1.42 m)，飛機爬到 1.0 m 正好
    卡進去；反過來，掛在遠高於起飛高度的東西則不擋爬升。所以同時比對水平距離與 z 區間：
      * 水平距離 < 安全距離 且 箱底 <= 起飛高度 + CLIMB_MARGIN -> ERROR (爬升會撞到)
      * 水平距離 < 安全距離 但 箱子整個在爬升柱之上          -> WARN  (不擋爬升, 但很近)
    訊息保留「起飛點 HOME」字樣, 並印出實際座標 (操作者才知道要走到哪裡看)。
    """
    takeoff_alt = float(cfg.get("waypoints", {}).get("takeoff_alt", 1.0))
    clr = float(field.clearance)
    pt = np.array([[tx, ty]])
    for o in field.obstacles:
        d = float(signed_distance(pt, o.footprint)[0])
        if d >= clr:
            continue
        span = f"該箱佔 z {o.z_bottom:.2f}~{o.z_top:.2f} m"
        # 爬升柱頂端 = 起飛高度 + 參考點上方的機體 + 超調
        climb_top = takeoff_alt + pivot_above(cfg) + CLIMB_OVERSHOOT
        overlaps = o.z_bottom <= climb_top
        if overlaps and d < 0:
            r.errors.append(
                f"起飛點 HOME ({tx:+.2f}, {ty:+.2f}) 就在障礙物「{o.name}」的底面範圍內 "
                f"(內縮 {-d:.2f} m), 原地爬到 {takeoff_alt:.2f} m 會撞上去 ({span}); "
                f"請把飛機移到別處起飛或移開障礙物"
            )
        elif overlaps:
            r.errors.append(
                f"起飛點 HOME ({tx:+.2f}, {ty:+.2f}) 離障礙物「{o.name}」只有 {d:.2f} m, "
                f"不足安全距離 {clr:.2f} m, 而且 {span} 與爬到 {takeoff_alt:.2f} m 的爬升重疊; "
                f"請把飛機移到別處起飛、移開障礙物, 或降低 waypoints.takeoff_alt"
            )
        else:
            near = (f"水平距離 {d:.2f} m" if d >= 0 else f"就在底面範圍內 (內縮 {-d:.2f} m)")
            r.warnings.append(
                f"起飛點 HOME ({tx:+.2f}, {ty:+.2f}) 附近有障礙物「{o.name}」({near} < 安全距離 "
                f"{clr:.2f} m; {span}), 整個高過 {takeoff_alt:.2f} m 的爬升柱, 不擋起飛; "
                f"進場爬升時仍請留意"
            )


def _densify_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray, step: float) -> np.ndarray:
    """3D 折線每段 (依水平長度) 細分到 <= step 的點 (M,3)。"""
    pts = np.column_stack([x, y, z])
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts[:, :2], axis=0).T)
    n_per = np.maximum(1, np.ceil(seg / max(step, 1e-3)).astype(int))
    out = [pts[:1]]
    for i, n in enumerate(n_per):
        frac = (np.arange(1, n + 1) / n)[:, None]
        out.append(pts[i] + (pts[i + 1] - pts[i]) * frac)
    return np.vstack(out)




def sustained_accel_p98(traj: Trajectory, window_s: float = 0.5) -> float:
    """軌跡需求的持續水平加速度 (P98, m/s²)。

    速度先做 window_s 移動平均再差分：單樣本的轉角尖點會被 5cm 級接受半徑
    的切角吸收，不代表實機需要的推力；持續值才是可行性判準。
    """
    if traj.n < 4:
        return 0.0
    dt = float(np.median(np.diff(traj.t))) or 1e-3
    win = max(1, int(round(window_s / dt)))
    kernel = np.ones(win) / win
    vx = np.convolve(traj.vx, kernel, mode="same")
    vy = np.convolve(traj.vy, kernel, mode="same")
    ax = np.gradient(vx, traj.t)
    ay = np.gradient(vy, traj.t)
    return float(np.percentile(np.hypot(ax, ay), 98))


def box_summary(box: SafeBox) -> str:
    return (
        f"可用盒: x∈[{box.x_min:.2f},{box.x_max:.2f}] "
        f"y∈[{box.y_min:.2f},{box.y_max:.2f}] "
        f"z∈[{box.z_min:.2f},{box.z_max:.2f}] (m)"
    )
