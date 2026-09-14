"""核心管線 smoke test：跑所有 pattern，檢查工時/邊界/速度/動態速度剖面。
用法: python tests/smoke.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:  # Windows 主控台預設非 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

from core import patterns  # noqa: E402
from core.config import deep_update, load_config  # noqa: E402
from core.geometry import safe_box_from_config  # noqa: E402
from core.planner import plan  # noqa: E402
from core.safety import box_summary, sustained_accel_p98  # noqa: E402
from core.speed_profile import (  # noqa: E402
    ProfileParams, compute_profile, effective_accel,
)


def test_profile_analytic() -> int:
    """速度剖面對直線的解析解驗證 (梯形 + jerk 修正)。"""
    fails = 0
    line = np.array([[0.0, 0.0], [10.0, 0.0]])

    # 純梯形 (jerk=0): T = L/v + v/a
    prm = ProfileParams(cruise=0.5, accel=1.0, jerk=0.0)
    dur = compute_profile(line, prm).duration
    expect = 10.0 / 0.5 + 0.5 / 1.0
    if abs(dur - expect) > 0.05:
        print(f"!! 梯形解析解: 預期 {expect:.2f}s, 得到 {dur:.2f}s")
        fails += 1

    # S 曲線 (jerk=1): 低速由 jerk 主導 -> 用有效加速度 a_eff=½√(j·v)。
    # 對稱梯形總時 T = L/v + v/a_eff (與真 jerk 受限 0->v->0 的時間一致)。
    prm_j = ProfileParams(cruise=0.5, accel=1.0, jerk=1.0)
    dur_j = compute_profile(line, prm_j).duration
    a_eff = effective_accel(1.0, 1.0, 0.5)          # = 0.5*sqrt(0.5) ≈ 0.354
    expect_j = 10.0 / 0.5 + 0.5 / a_eff
    if abs(dur_j - expect_j) > 0.15:
        print(f"!! jerk 解析解: 預期 {expect_j:.2f}s, 得到 {dur_j:.2f}s")
        fails += 1

    # jerk 單調性: 提高 accel 不應讓工時變長 (jerk 主導區應持平, 加速主導區應變短)
    dl = [compute_profile(line, ProfileParams(cruise=0.5, accel=a, jerk=1.0)).duration
          for a in (0.3, 1.0, 3.0)]
    if not (dl[0] >= dl[1] - 1e-6 >= dl[2] - 1e-6):
        print(f"!! accel 單調性 (直線): {[round(x, 2) for x in dl]} 未非遞增")
        fails += 1

    # 90° 轉角: 5cm 接受半徑 -> fillet R=0.121m -> v_corner≈0.35 (lat_accel=1)
    corner = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]])
    prof = compute_profile(corner, ProfileParams(cruise=0.5, accel=1.0, jerk=0.0))
    if prof.hard_corners != 1:
        print(f"!! 90° 轉角未被偵測 (hard_corners={prof.hard_corners})")
        fails += 1
    v_at_corner = float(np.interp(3.0, prof.s, prof.v))
    if not (0.25 < v_at_corner < 0.45):
        print(f"!! 90° 轉角通過速度 {v_at_corner:.2f} 不在預期 0.25~0.45 m/s")
        fails += 1

    # 180° 折返: 應減速到近懸停 (corner_min_speed)
    uturn = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]])
    prof_u = compute_profile(uturn, ProfileParams(cruise=0.5, accel=1.0, jerk=0.0))
    v_turn = float(np.interp(3.0, prof_u.s, prof_u.v))
    if v_turn > 0.08:
        print(f"!! 180° 折返速度 {v_turn:.2f} 未降到近懸停")
        fails += 1

    # 取樣密度無關性 (曲率用原始頂點弧長, 非加密步長)：
    # 同一個 R=2 圓在 cruise=0.8 下, 粗取樣 (24~120 點) 與細取樣 (720) 工時應相近。
    # (修好前粗取樣會因 κ=δ/densified_ds 高估曲率而虛假變慢 ~30-60%)
    def circle_dur(n_samples):
        th = np.linspace(0.0, 2 * np.pi, n_samples + 1)
        poly = np.column_stack([2.0 * np.cos(th), 2.0 * np.sin(th)])
        return compute_profile(poly, ProfileParams(cruise=0.8, accel=1.0,
                               lat_accel=1.0, jerk=1.0)).duration
    fine = circle_dur(720)
    for n in (24, 40, 120):
        coarse = circle_dur(n)
        if abs(coarse - fine) / fine > 0.05:
            print(f"!! 圓取樣密度敏感: n={n} -> {coarse:.2f}s vs n=720 -> {fine:.2f}s "
                  f"(差 {100*abs(coarse-fine)/fine:.0f}%)")
            fails += 1

    # 轉角以水平 (XY) 幾何計算：3D 折線的高度變化不得讓 180° 折返 / 90° 角變「緩」。
    # (a) 帶爬升的往返: 3D 向量夾角 < 180°, 但水平仍是原路折返 -> 過彎速度應與 2D 相同
    prm3 = ProfileParams(cruise=0.5, accel=1.0, jerk=0.0)
    uturn3d = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.4], [0.0, 0.0, 1.8]])
    prof_u3 = compute_profile(uturn3d, prm3)
    s_mid = float(np.linalg.norm(uturn3d[1] - uturn3d[0]))     # 折返頂點的 3D 弧長位置
    v_turn3 = float(np.interp(s_mid, prof_u3.s, prof_u3.v))
    if v_turn3 > 0.08 or prof_u3.hard_corners != 1:
        print(f"!! 3D 折返未被當成 180° (v={v_turn3:.2f}, hard={prof_u3.hard_corners})")
        fails += 1
    # (b) 帶爬升的 90° 角: 過彎速度應與純 2D 直角相同 (角度只看 XY)
    corner3d = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.5], [3.0, 3.0, 2.0]])
    prof_c3 = compute_profile(corner3d, prm3)
    v_c3 = float(np.interp(float(np.linalg.norm(corner3d[1] - corner3d[0])), prof_c3.s, prof_c3.v))
    if abs(v_c3 - v_at_corner) > 0.03:
        print(f"!! 3D 直角過彎速度 {v_c3:.2f} != 2D {v_at_corner:.2f} (角度應只看 XY)")
        fails += 1
    # (c) 先純垂直升降再轉向: 轉角應在垂直段頂端被偵測 (不可被零向量吃掉)
    hop = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [3.0, 0.0, 1.5], [3.0, 3.0, 1.5]])
    prof_h = compute_profile(hop, prm3)
    if prof_h.hard_corners != 1:
        print(f"!! 垂直段後的直角未被偵測 (hard_corners={prof_h.hard_corners})")
        fails += 1

    # 3D 斜段: cruise 是水平速度上限 -> 沿斜段 3D 速度 = cruise/cos(傾角) (與 GUIDED 合成速度一致),
    #     垂直分量另受 vz_up/vz_dn 限制。10 m 水平 + 2 m 爬升的直線: 時間應 ≈ 純水平 10 m 的時間
    #     (垂直 0.2·0.5=0.1 m/s 遠低於 vz_up), 而不是 3D 長度 10.2 m / 0.5。
    prm3 = ProfileParams(cruise=0.5, accel=1.0, jerk=0.0, vz_up=1.0, vz_dn=0.6)
    d_flat = compute_profile(np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]), prm3).duration
    d_slant = compute_profile(np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 3.0]]), prm3).duration
    if abs(d_slant - d_flat) > 0.3:
        print(f"!! 斜段估時 {d_slant:.2f}s 應 ≈ 水平段 {d_flat:.2f}s (cruise 應為水平上限)")
        fails += 1
    # 陡降段: 垂直上限主導 -> 時間 ≈ 由 vz_dn 決定 (2 m 降、1 m 水平: 3D 速度 <= 0.6/(2/√5)=0.67)
    steep = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 1.0]])
    prof_st = compute_profile(steep, prm3)
    v_peak = float(np.max(prof_st.v))
    if not (0.55 <= v_peak <= 0.68):
        print(f"!! 陡降段峰值 3D 速度 {v_peak:.2f} 不在 vz_dn 主導的 0.55~0.68 m/s")
        fails += 1
    # 純垂直段: 速度 = vz_up, 不得因水平限制為 inf 而崩潰
    vert = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    prof_v = compute_profile(vert, prm3)
    if not np.isfinite(prof_v.duration) or abs(float(np.max(prof_v.v)) - 1.0) > 0.02:
        print(f"!! 純垂直段: dur={prof_v.duration:.2f}, vmax={float(np.max(prof_v.v)):.2f} (應為 vz_up=1.0)")
        fails += 1

    # 退化短段 (< densify step) 不應產生爆炸性工時 (finding: 6e7s / OOM)
    tiny = np.array([[0.0, 0.0], [0.03, 0.0]])
    dur_tiny = compute_profile(tiny, ProfileParams(cruise=0.5, accel=1.0, jerk=1.0)).duration
    if not (0.0 < dur_tiny < 5.0):
        print(f"!! 退化短段工時異常: {dur_tiny:.3g}s (應為個位數秒)")
        fails += 1

    if not fails:
        print("速度剖面解析解驗證 OK")
    return fails


def test_altitude_stair(cfg) -> int:
    """階梯高度模式：平台明顯 (vz==0 佔多數)、階層數正確、垂直速度不超過 speed_up/down。"""
    from core.altitude import STAIR_AUTO_RAMP_FRAC, altitude_profile

    fails = 0
    f = cfg["flight"]
    # auto 升降速度 = FRAC × min(垂直上限, 3D 合成速度餘裕 √(max_speed²−cruise²))
    room = math.sqrt(max(float(f["max_speed"]) ** 2 - float(f["cruise_speed"]) ** 2, 0.0))
    v_up_lim = min(float(f.get("speed_up", 1.0)), room)
    v_dn_lim = min(float(f.get("speed_down", 0.6)), room)

    for steps in (1, 3, 4):
        cfg_s = deep_update(cfg, {"altitude": {"mode": "stair", "steps": steps,
                                                "ramp_speed": "auto"}})
        pr = plan("circle", cfg_s)
        t = pr.trajectory
        vz = t.vz
        box = pr.box
        base = box.z_mid
        amp = 0.45 * box.z_span
        levels = base + amp * (2.0 * np.arange(steps + 1) / steps - 1.0)
        tag = f"stair steps={steps}"

        if not pr.report.ok or any(("上升速度" in w or "下降速度" in w) for w in pr.report.warnings):
            print(f"!! {tag}: 安全檢查有垂直速度問題: {pr.report.errors + pr.report.warnings}")
            fails += 1
        # 起點/終點在 base (與 sine/triangle 一致)
        if abs(t.z[0] - base) > 1e-6 or abs(t.z[-1] - base) > 1e-6:
            print(f"!! {tag}: 起/終點高度 {t.z[0]:.3f}/{t.z[-1]:.3f} 不在 base {base:.3f}")
            fails += 1
        # 用滿振幅
        if abs(t.z.min() - (base - amp)) > 1e-6 or abs(t.z.max() - (base + amp)) > 1e-6:
            print(f"!! {tag}: z 範圍 {t.z.min():.3f}~{t.z.max():.3f} 未達 base±amp")
            fails += 1
        # 平台: 至少一半以上時間垂直速度為 0，且平台樣本都落在某一階層上、每一階層都到過
        plateau = np.abs(vz) < 1e-9
        if plateau.mean() < 0.5:
            print(f"!! {tag}: 平飛比例 {plateau.mean():.2f} 太低, 不像階梯")
            fails += 1
        zp = t.z[plateau]
        off = np.min(np.abs(zp[:, None] - levels[None, :]), axis=1)
        if off.max() > 1e-6:
            print(f"!! {tag}: 有平台樣本不在階層上 (最大偏差 {off.max():.3g} m)")
            fails += 1
        if not all(np.any(np.abs(zp - L) < 1e-6) for L in levels):
            print(f"!! {tag}: 有階層從未到達 (levels={np.round(levels, 2)})")
            fails += 1
        # 升降速度 = auto 比例 × 上限 (不多不少)
        if abs(vz.max() - STAIR_AUTO_RAMP_FRAC * v_up_lim) > 1e-3 or \
           abs(-vz.min() - STAIR_AUTO_RAMP_FRAC * v_dn_lim) > 1e-3:
            print(f"!! {tag}: 升降速度 {vz.max():.3f}/{-vz.min():.3f} 不等於 "
                  f"{STAIR_AUTO_RAMP_FRAC}×({v_up_lim:.3f}/{v_dn_lim:.3f})")
            fails += 1

    # auto 在較快巡航 (0.8 m/s, max_speed 1.0) 下也不得自己觸發 3D 超速 ERROR / 垂直速度警告
    cfg_f = deep_update(cfg, {"flight": {"cruise_speed": 0.8},
                              "altitude": {"mode": "stair", "steps": 4, "ramp_speed": "auto"}})
    for key in ("circle", "reciprocate"):
        pr_f = plan(key, cfg_f)
        bad = pr_f.report.errors + [w for w in pr_f.report.warnings
                                    if "上升速度" in w or "下降速度" in w]
        if bad:
            print(f"!! stair auto @ cruise 0.8 ({key}): {bad}")
            fails += 1

    # 手動 ramp_speed: 升降皆用該速度
    cfg_m = deep_update(cfg, {"altitude": {"mode": "stair", "steps": 4, "ramp_speed": 0.3}})
    t = plan("circle", cfg_m).trajectory
    if abs(t.vz.max() - 0.3) > 1e-3 or abs(-t.vz.min() - 0.3) > 1e-3:
        print(f"!! stair ramp_speed=0.3: 實際 {t.vz.max():.3f}/{-t.vz.min():.3f}")
        fails += 1
    # 手動 ramp_speed 超過上限 -> 應被 safety 警告
    cfg_h = deep_update(cfg, {"altitude": {"mode": "stair", "steps": 4, "ramp_speed": 1.5}})
    rep = plan("circle", cfg_h).report
    if not any("上升速度" in w for w in rep.warnings):
        print("!! stair ramp_speed=1.5 未觸發上升速度警告")
        fails += 1

    # 退化: cycles 過多 -> 平台消失, 但不得崩潰、仍在安全帶內
    cfg_d = deep_update(cfg, {"altitude": {"mode": "stair", "steps": 4, "cycles": 60}})
    pr_d = plan("circle", cfg_d)
    b = pr_d.trajectory.bounds()["z"]
    if b[0] < pr_d.box.z_min - 1e-6 or b[1] > pr_d.box.z_max + 1e-6:
        print(f"!! stair cycles=60: z {b} 超出安全帶")
        fails += 1

    # cycles=0 -> 平飛於 base
    box = safe_box_from_config(cfg)
    tt = np.linspace(0, 100, 2001)
    z0 = altitude_profile(tt, box, deep_update(cfg, {"altitude": {"mode": "stair", "cycles": 0}}),
                          s=np.linspace(0, 50, len(tt)))
    if np.ptp(z0) > 1e-9:
        print("!! stair cycles=0 應為平飛")
        fails += 1

    if not fails:
        print("階梯 (stair) 高度模式驗證 OK")
    return fails


def test_sparse_straight(cfg) -> int:
    """直線型軌跡的 .waypoints：平飛只留端點 (振幅 0 -> 只剩精確轉角)、曲線不受影響、
    高度容差 z_tol 精簡後折線仍在容差內、旗標關閉回到密集取樣，且 AUTO 估時折線 == 匯出折線。"""
    from core.trajectory import mission_polyline, resample_by_spacing
    from io_export.waypoints import trajectory_to_waypoints

    fails = 0
    flat = deep_update(cfg, {"altitude": {"amplitude": 0.0}})
    # (pattern, 每圈頂點數) —— 往返 2 點一線 A,B,A / 矩形 4 角 / 鋸齒 2*teeth
    teeth = int(cfg["patterns"]["zigzag"]["teeth"])
    for key, per_lap in (("reciprocate", 2), ("rectangle", 4), ("zigzag", 2 * teeth)):
        pr = plan(key, flat)
        t = pr.trajectory
        mp = mission_polyline(t, flat, pr.pattern.is_smooth, pr.pattern.lap_xy, pr.laps, "precision")
        # 預設 do_jump: 一圈 block = 每圈頂點數 (+ 收尾點); 實飛折線 = 每圈頂點數 × 圈數 + 1
        if mp.repeat != "do_jump" or mp.block_len != per_lap or mp.n_nav != per_lap + 1:
            print(f"!! sparse {key} (振幅 0): block {mp.block_len}/n_nav {mp.n_nav} != 每圈 {per_lap} 頂點 (+收尾)")
            fails += 1
        if len(mp.flown_x) != per_lap * pr.laps + 1:
            print(f"!! sparse {key} (振幅 0): 實飛折線 {len(mp.flown_x)} != {per_lap} × {pr.laps} + 1")
            fails += 1
        if np.ptp(mp.flown_z) > 1e-9:
            print(f"!! sparse {key} (振幅 0): 高度應全等於 base, ptp={np.ptp(mp.flown_z):.3g}")
            fails += 1
        # 每個航點都是精確頂點 (在 lap_xy 裡)
        verts = np.asarray(pr.pattern.lap_xy)
        far = [i for i in range(mp.n_nav)
               if np.min(np.hypot(verts[:, 0] - mp.x[i], verts[:, 1] - mp.y[i])) > 1e-9]
        if far:
            print(f"!! sparse {key}: {len(far)} 個航點不是精確頂點")
            fails += 1
        # 匯出檔的 nav 數與 AUTO 估時所用點數一致
        _, n_nav = trajectory_to_waypoints(pr, flat, "precision")
        auto = t.meta["auto_estimate"]["precision"]
        if n_nav != mp.n_nav or auto["n_wp"] != mp.n_nav:
            print(f"!! sparse {key}: 匯出 nav {n_nav} / 估時 n_wp {auto['n_wp']} / 折線 {mp.n_nav} 不一致")
            fails += 1
        # AUTO 航線時間應 ≈ 軌跡工時 + 進場 (同一條頂點折線)
        if not (t.duration - 1.0 <= auto["nav_s"] <= t.duration + 15.0):
            print(f"!! sparse {key}: AUTO 航線 {auto['nav_s']:.0f}s 與軌跡工時 {t.duration:.0f}s 差太多")
            fails += 1

    # 往返 (振幅 0) 就是 A,B,A,B... 兩點一直線 (實飛折線)
    pr = plan("reciprocate", flat)
    mp = mission_polyline(pr.trajectory, flat, False, pr.pattern.lap_xy, pr.laps)
    hx = pr.pattern.lap_xy[1, 0]
    fx = mp.flown_x
    if not (np.allclose(fx[0::2], -hx) and np.allclose(fx[1::2], hx) and np.allclose(mp.flown_y, 0.0)):
        print("!! sparse reciprocate: 不是 A,B,A,B... 兩點一線")
        fails += 1

    # 曲線 (圓/8字) 不受直線精簡影響：unroll 時與直接沿弧長取樣完全相同
    unroll = deep_update(flat, {"waypoints": {"repeat": "unroll"}})
    for key in ("circle", "figure_eight"):
        pr = plan(key, unroll)
        t = pr.trajectory
        mp = mission_polyline(t, unroll, True, pr.pattern.lap_xy, pr.laps, "precision")
        w = unroll["waypoints"]
        spacing = w["point_spacing"] * (w["spline_spacing_mult"] if mp.use_spline else 1.0)
        xo, yo, zo = resample_by_spacing(t.x, t.y, t.z, spacing)
        if mp.repeat != "unroll" or mp.n_nav != len(xo) or not np.allclose(mp.x, xo):
            print(f"!! sparse 不應影響曲線 {key} ({mp.n_nav} vs {len(xo)})")
            fails += 1

    # 高度有變化 (sine) 的直線軌跡: z_tol 精簡後航點變少、段變長, 但折線高度與剖面差 <= z_tol
    n_dense0 = None
    for z_tol in (0.0, 0.02, 0.05):
        c = deep_update(cfg, {"waypoints": {"z_tol": z_tol, "repeat": "unroll"}})
        pr = plan("rectangle", c)
        t = pr.trajectory
        mp = mission_polyline(t, c, False, pr.pattern.lap_xy, pr.laps)
        # 折線在每 5 cm 處的高度 vs 軌跡 z(s)
        d = np.hypot(np.diff(mp.x), np.diff(mp.y))
        s_mp = np.concatenate([[0.0], np.cumsum(d)])
        s_q = np.arange(0.0, s_mp[-1], 0.05)
        z_poly = np.interp(s_q, s_mp, mp.z)
        z_true = np.interp(s_q, t.s, t.z)
        dev = float(np.max(np.abs(z_poly - z_true)))
        if dev > z_tol + 2e-3:
            print(f"!! z_tol={z_tol}: 折線高度偏差 {dev:.3f} m 超過容差")
            fails += 1
        if z_tol == 0.0:
            n_dense0 = mp.n_nav
            if np.ptp(mp.z) < 1.0:
                print("!! sine rectangle (z_tol=0) 航點高度起伏遺失")
                fails += 1
        else:
            if not (mp.n_nav < 0.5 * n_dense0):
                print(f"!! z_tol={z_tol}: 航點 {mp.n_nav} 未明顯少於不精簡的 {n_dense0}")
                fails += 1
            legs = np.sqrt(np.diff(mp.x) ** 2 + np.diff(mp.y) ** 2 + np.diff(mp.z) ** 2)
            if np.median(legs) < 0.5:
                print(f"!! z_tol={z_tol}: 中位段長 {np.median(legs):.2f} m 仍太短")
                fails += 1

    # stair (unroll): 平台變疏但升降段有點 -> 介於「只剩轉角」與「密集」之間
    st = deep_update(cfg, {"altitude": {"mode": "stair", "steps": 4}, "waypoints": {"repeat": "unroll"}})
    pr = plan("reciprocate", st)
    mp = mission_polyline(pr.trajectory, st, False, pr.pattern.lap_xy, pr.laps)
    dense_n = len(resample_by_spacing(pr.trajectory.x, pr.trajectory.y, pr.trajectory.z,
                                      cfg["waypoints"]["point_spacing"])[0])
    if not (2 * pr.laps + 1 < mp.n_nav < 0.7 * dense_n):
        print(f"!! stair reciprocate 航點 {mp.n_nav} 未落在 (轉角數, 0.7×密集) 之間")
        fails += 1
    if abs(mp.z.max() - pr.trajectory.z.max()) > 0.02 or abs(mp.z.min() - pr.trajectory.z.min()) > 0.02:
        print("!! stair reciprocate 航點高度範圍與軌跡不符")
        fails += 1

    # 旗標關閉 (+ unroll) -> 舊行為 (沿弧長密集取樣)
    off = deep_update(flat, {"waypoints": {"sparse_straight": False, "repeat": "unroll"}})
    pr = plan("rectangle", off)
    mp = mission_polyline(pr.trajectory, off, False, pr.pattern.lap_xy, pr.laps)
    xo, yo, zo = resample_by_spacing(pr.trajectory.x, pr.trajectory.y, pr.trajectory.z,
                                     off["waypoints"]["point_spacing"])
    if mp.n_nav != len(xo) or not np.allclose(mp.x, xo):
        print(f"!! sparse_straight=false 應回到密集取樣 ({mp.n_nav} vs {len(xo)})")
        fails += 1

    # compact 預算仍然守住 (縮小預算強迫觸發; unroll 才會超過)
    tiny = deep_update(cfg, {"waypoints": {"fc_budget": 60, "repeat": "unroll", "z_tol": 0.0}})
    for key in ("rectangle", "circle"):
        pr = plan(key, tiny)
        lines, n_nav = trajectory_to_waypoints(pr, tiny, "compact")
        if len(lines) - 1 > 60:
            print(f"!! compact {key}: 總任務項 {len(lines) - 1} 超過預算 60")
            fails += 1

    if not fails:
        print("直線平飛只留端點 (sparse_straight / z_tol) 驗證 OK")
    return fails


def test_do_jump(cfg) -> int:
    """DO_JUMP 重複圈數：一圈 block + DO_JUMP + 收尾；任務項數與圈數無關；每圈高度相同；
    實飛折線 = block × 圈數；unroll 為舊行為。"""
    from core.trajectory import mission_polyline
    from io_export.waypoints import CMD_DO_JUMP, CMD_LAND, trajectory_to_waypoints

    fails = 0
    for key, _ in patterns.list_patterns():
        pr = plan(key, cfg)
        t = pr.trajectory
        mp = mission_polyline(t, cfg, pr.pattern.is_smooth, pr.pattern.lap_xy, pr.laps,
                              repeatable=pr.pattern.repeatable)
        lines, n_nav = trajectory_to_waypoints(pr, cfg, "compact")
        rows = [ln.split("\t") for ln in lines[1:]]
        jumps = [(i, r) for i, r in enumerate(rows) if int(r[3]) == CMD_DO_JUMP]
        if not pr.pattern.repeatable:
            # 宣告不重複的 pattern (隨機手飛): 一律 unroll、不寫 DO_JUMP
            if jumps or mp.repeat != "unroll" or n_nav != len(mp.flown_x):
                print(f"!! {key}: repeatable=False 卻有 DO_JUMP / 非 unroll")
                fails += 1
            continue
        if pr.laps > 1:
            if len(jumps) != 1:
                print(f"!! do_jump {key}: DO_JUMP 數 {len(jumps)} != 1")
                fails += 1
                continue
            i, r = jumps[0]
            first_nav_seq = 3                       # HOME, TAKEOFF, SPEED 之後
            if int(float(r[4])) != first_nav_seq or int(float(r[5])) != pr.laps - 1:
                print(f"!! do_jump {key}: DO_JUMP 參數 ({r[4]}, {r[5]}) != ({first_nav_seq}, {pr.laps - 1})")
                fails += 1
            # block 在 DO_JUMP 前, 收尾點在後, 最後 LAND
            if i != first_nav_seq + mp.block_len or int(rows[-1][3]) != CMD_LAND:
                print(f"!! do_jump {key}: DO_JUMP 位置 {i} != {first_nav_seq + mp.block_len} 或結尾非 LAND")
                fails += 1
            # 收尾點 = 起點 (閉合)
            if abs(mp.x[-1] - mp.x[0]) > 1e-9 or abs(mp.y[-1] - mp.y[0]) > 1e-9 or abs(mp.z[-1] - mp.z[0]) > 1e-6:
                print(f"!! do_jump {key}: 收尾點不等於起點")
                fails += 1
        # 實飛折線 = block 重複 laps 圈
        exp_flown = mp.block_len * pr.laps + 1
        if len(mp.flown_x) != exp_flown:
            print(f"!! do_jump {key}: 實飛折線 {len(mp.flown_x)} != block {mp.block_len} × {pr.laps} + 1")
            fails += 1
        # 每圈高度相同 (軌跡 z 為弧長的週期函數, 週期 = 一圈長)
        L = t.meta["single_lap_length_m"]
        if pr.laps > 1:
            sq = np.linspace(0.0, L, 200)
            z1 = np.interp(sq, t.s, t.z)
            z2 = np.interp(sq + L, t.s, t.z)
            if np.max(np.abs(z1 - z2)) > 2e-3:
                print(f"!! do_jump {key}: 各圈高度不同 (max diff {np.max(np.abs(z1 - z2)):.3g} m)")
                fails += 1
        # 每圈起伏次數 = 整數 k >= 1, 全程 = k × 圈數
        k = t.meta["altitude_cycles_per_lap"]
        if k < 1 or t.meta["altitude_cycles_effective"] != k * pr.laps:
            print(f"!! do_jump {key}: 每圈起伏 {k}, 全程 {t.meta['altitude_cycles_effective']} 不一致")
            fails += 1

    # smooth_random 在 do_jump 下也必須每圈相同 (每圈同一段隨機形狀), 且匯出 block 高度 == 軌跡第 2 圈
    for key in ("circle", "rectangle", "random_walk"):
        c = deep_update(cfg, {"altitude": {"mode": "smooth_random"}})
        pr = plan(key, c)
        t = pr.trajectory
        L = t.meta["single_lap_length_m"]
        if pr.laps > 1:
            sq = np.linspace(0.0, L, 300)
            z1 = np.interp(sq, t.s, t.z)
            z2 = np.interp(sq + L * (pr.laps - 1), t.s, t.z)
            if np.max(np.abs(z1 - z2)) > 5e-3:
                print(f"!! do_jump smooth_random {key}: 首圈與末圈高度不同 (max diff {np.max(np.abs(z1 - z2)):.3g} m)")
                fails += 1
        if np.ptp(t.z) < 0.3:
            print(f"!! do_jump smooth_random {key}: 高度幾乎沒變化")
            fails += 1

    # 任務項數與圈數無關 (手動 3 圈 vs 9 圈; 起伏固定每圈 1 次)
    n_items = []
    for laps in (3, 9):
        c = deep_update(cfg, {"flight": {"laps": laps}, "altitude": {"cycles": laps}})
        pr = plan("circle", c)
        lines, _ = trajectory_to_waypoints(pr, c, "compact")
        n_items.append(len(lines) - 1)
    if n_items[0] != n_items[1]:
        print(f"!! do_jump: 3 圈 {n_items[0]} 項 vs 9 圈 {n_items[1]} 項 (應相同)")
        fails += 1

    # 1 圈: 不寫 DO_JUMP
    pr = plan("circle", deep_update(cfg, {"flight": {"laps": 1}}))
    lines, _ = trajectory_to_waypoints(pr, cfg, "compact")
    if any(int(ln.split("\t")[3]) == CMD_DO_JUMP for ln in lines[1:]):
        print("!! do_jump: 1 圈不應有 DO_JUMP")
        fails += 1

    # unroll: 無 DO_JUMP, 導航點 = 實飛折線, 起伏次數走互質調整
    un = deep_update(cfg, {"waypoints": {"repeat": "unroll"}})
    pr = plan("rectangle", un)
    mp = mission_polyline(pr.trajectory, un, False, pr.pattern.lap_xy, pr.laps)
    lines, n_nav = trajectory_to_waypoints(pr, un, "compact")
    if mp.repeat != "unroll" or n_nav != len(mp.flown_x) or \
       any(int(ln.split("\t")[3]) == CMD_DO_JUMP for ln in lines[1:]):
        print("!! unroll: 不應有 DO_JUMP 且導航點應 = 實飛折線")
        fails += 1
    if pr.trajectory.meta["altitude_cycles_per_lap"] != 0:
        print("!! unroll: 不應套用每圈整數起伏")
        fails += 1

    if not fails:
        print("DO_JUMP 重複圈數驗證 OK")
    return fails


def test_random_walk(cfg) -> int:
    """隨機手飛路徑：同 seed 決定性、換 seed 不同、精確閉合、全在盒內、平滑 (無硬折角 >90°)、
    轉折點數影響一圈長度、DO_JUMP 一圈 block 收尾 == 起點。"""
    from core.trajectory import mission_polyline

    fails = 0
    pr_a = plan("random_walk", cfg)
    pr_b = plan("random_walk", cfg)
    if not np.array_equal(pr_a.pattern.lap_xy, pr_b.pattern.lap_xy):
        print("!! random_walk: 同 seed 兩次結果不同 (必須決定性)")
        fails += 1
    pr_c = plan("random_walk", deep_update(cfg, {"patterns": {"random_walk": {"seed": 7}}}))
    if pr_c.pattern.lap_xy.shape == pr_a.pattern.lap_xy.shape and \
       np.allclose(pr_c.pattern.lap_xy, pr_a.pattern.lap_xy):
        print("!! random_walk: 換 seed 路徑竟相同")
        fails += 1
    if not pr_a.pattern.is_smooth:
        print("!! random_walk 應為曲線 (is_smooth)")
        fails += 1

    for seed in range(4):
        c = deep_update(cfg, {"patterns": {"random_walk": {"seed": seed}}})
        pr = plan("random_walk", c)
        lap = pr.pattern.lap_xy
        box = pr.box
        if not np.allclose(lap[0], lap[-1]):
            print(f"!! random_walk seed {seed}: 未精確閉合")
            fails += 1
        if lap[:, 0].min() < box.x_min - 1e-9 or lap[:, 0].max() > box.x_max + 1e-9 or \
           lap[:, 1].min() < box.y_min - 1e-9 or lap[:, 1].max() > box.y_max + 1e-9:
            print(f"!! random_walk seed {seed}: 超出可用盒")
            fails += 1
        # 用到大部分的盒子 (兩軸範圍 >= 85%; 只縮不放, 不強求碰到牆)
        if np.ptp(lap[:, 0]) < 0.85 * (box.x_max - box.x_min) or np.ptp(lap[:, 1]) < 0.85 * (box.y_max - box.y_min):
            print(f"!! random_walk seed {seed}: 路徑範圍太小 (x {np.ptp(lap[:, 0]):.2f}, y {np.ptp(lap[:, 1]):.2f} m)")
            fails += 1
        # 平滑、無髮夾彎: 相鄰取樣段夾角 <= 60° (轉向上限 + 平順回起點)
        d = np.diff(lap, axis=0)
        n = np.linalg.norm(d, axis=1)
        ok = (n[:-1] > 1e-9) & (n[1:] > 1e-9)
        cosang = np.einsum("ij,ij->i", d[:-1], d[1:])[ok] / (n[:-1] * n[1:])[ok]
        if np.min(cosang) < np.cos(np.radians(60.0)):
            print(f"!! random_walk seed {seed}: 有 > 60° 的折角 (髮夾彎; 最大 "
                  f"{np.degrees(np.arccos(np.clip(np.min(cosang), -1, 1))):.0f}°)")
            fails += 1
        if not pr.report.ok:
            print(f"!! random_walk seed {seed}: 安全檢查未過 {pr.report.errors}")
            fails += 1
        mp = mission_polyline(pr.trajectory, c, True, lap, pr.laps)
        if mp.repeat == "do_jump" and (abs(mp.x[-1] - mp.x[0]) > 1e-9 or abs(mp.y[-1] - mp.y[0]) > 1e-9):
            print(f"!! random_walk seed {seed}: DO_JUMP block 收尾點 != 起點")
            fails += 1

    # 預設 n_points=auto: 整段一條不重複 -> 圈數 1、無 DO_JUMP、AUTO 航線落在 [min, max]
    from io_export.waypoints import CMD_DO_JUMP, trajectory_to_waypoints
    f = cfg["flight"]
    for seed in range(4):
        c = deep_update(cfg, {"patterns": {"random_walk": {"seed": seed}}})
        pr = plan("random_walk", c)
        nav = pr.trajectory.meta["auto_estimate"]["compact"]["nav_s"]
        lines, _ = trajectory_to_waypoints(pr, c, "compact")
        if pr.laps != 1:
            print(f"!! random_walk auto seed {seed}: 圈數 {pr.laps} != 1 (整段應為一條路徑)")
            fails += 1
        if any(int(ln.split("\t")[3]) == CMD_DO_JUMP for ln in lines[1:]):
            print(f"!! random_walk auto seed {seed}: 不應有 DO_JUMP")
            fails += 1
        if not (float(f["min_duration"]) - 1 <= nav <= float(f["max_duration"]) + 1):
            print(f"!! random_walk auto seed {seed}: AUTO 航線 {nav:.0f}s 不在 [{f['min_duration']},{f['max_duration']}]")
            fails += 1
        if not pr.pattern.description.startswith("隨機手飛"):
            print(f"!! random_walk: description 異常 {pr.pattern.description}")
            fails += 1
    # 手動轉折點數 (短路徑) -> 圈數 >1 但仍 unroll (各圈高度互質錯開)
    c = deep_update(cfg, {"patterns": {"random_walk": {"n_points": 12}}})
    pr = plan("random_walk", c)
    lines, _ = trajectory_to_waypoints(pr, c, "compact")
    if pr.laps < 2 or any(int(ln.split("\t")[3]) == CMD_DO_JUMP for ln in lines[1:]) or \
       pr.trajectory.meta["altitude_cycles_per_lap"] != 0:
        print(f"!! random_walk n_points=12: 應多圈全展開且無 DO_JUMP (laps={pr.laps})")
        fails += 1

    # 轉折點越多一圈越長
    l_small = plan("random_walk", deep_update(cfg, {"patterns": {"random_walk": {"n_points": 6}}}))
    l_big = plan("random_walk", deep_update(cfg, {"patterns": {"random_walk": {"n_points": 30}}}))
    if l_big.trajectory.meta["single_lap_length_m"] <= l_small.trajectory.meta["single_lap_length_m"]:
        print("!! random_walk: 30 個轉折點的一圈不比 6 個長")
        fails += 1

    if not fails:
        print("隨機手飛路徑 (random_walk) 驗證 OK")
    return fails


def test_duration_basis(cfg) -> int:
    """工時基準：auto (預設) 以 AUTO 航線時間決定圈數與 200~300 s 判定；guided 為舊行為。"""
    from core.planner import auto_nav_seconds, basis_duration, duration_basis
    from core.timing import decide_laps

    fails = 0
    f = cfg["flight"]
    target, min_dur, max_dur = float(f["target_duration"]), float(f["min_duration"]), float(f["max_duration"])
    if duration_basis(cfg) != "auto":
        print("!! 預設 duration_basis 應為 auto")
        fails += 1

    cfg_g = deep_update(cfg, {"flight": {"duration_basis": "guided"}})
    for key, _ in patterns.list_patterns():
        # guided: 圈數 == decide_laps 的純水平剖面結果; 判定用軌跡時間
        pr_g = plan(key, cfg_g)
        laps_dec, _ = decide_laps(pr_g.pattern.lap_xy, cfg_g)
        if pr_g.laps != laps_dec:
            print(f"!! basis=guided {key}: 圈數 {pr_g.laps} != decide_laps {laps_dec}")
            fails += 1
        if abs(basis_duration(pr_g.trajectory, cfg_g) - pr_g.trajectory.duration) > 1e-9:
            print(f"!! basis=guided {key}: basis_duration 應為軌跡時間")
            fails += 1

        # auto: 判定用 AUTO 航線時間, 落在 [min, max]、且比相鄰圈數更貼近 target
        pr_a = plan(key, cfg)
        nav = auto_nav_seconds(pr_a.trajectory)
        if nav is None or abs(basis_duration(pr_a.trajectory, cfg) - nav) > 1e-9:
            print(f"!! basis=auto {key}: basis_duration 應為 AUTO 航線時間")
            fails += 1
            continue
        if not (min_dur - 1 <= nav <= max_dur + 1):
            print(f"!! basis=auto {key}: AUTO 航線 {nav:.0f}s 不在 [{min_dur:.0f},{max_dur:.0f}]")
            fails += 1
        for n in (pr_a.laps - 1, pr_a.laps + 1):
            if n < 1:
                continue
            pr_n = plan(key, deep_update(cfg, {"flight": {"laps": n}}))
            nav_n = auto_nav_seconds(pr_n.trajectory)
            if nav_n >= min_dur and abs(nav_n - target) < abs(nav - target) - 1e-6:
                print(f"!! basis=auto {key}: {n} 圈 (AUTO {nav_n:.0f}s) 比 {pr_a.laps} 圈 "
                      f"(AUTO {nav:.0f}s) 更貼近 target {target:.0f}s")
                fails += 1
        # 手動圈數: 兩種基準都照用
        pr_m = plan(key, deep_update(cfg, {"flight": {"laps": 3}}))
        if pr_m.laps != 3:
            print(f"!! {key}: 手動 laps=3 未被採用 ({pr_m.laps})")
            fails += 1

    # AUTO 一定不短於 GUIDED (含進場) -> auto 基準的圈數 <= guided 基準的圈數
    for key, _ in patterns.list_patterns():
        la = plan(key, cfg).laps
        lg = plan(key, cfg_g).laps
        if la > lg:
            print(f"!! {key}: auto 基準圈數 {la} > guided 基準 {lg}")
            fails += 1

    # 工時警告訊息隨基準改變 (把 min_duration 拉高強迫觸發)
    hi = deep_update(cfg, {"flight": {"min_duration": 900, "laps": 2}})
    w_a = plan("circle", hi).report.warnings
    w_g = plan("circle", deep_update(hi, {"flight": {"duration_basis": "guided"}})).report.warnings
    if not any("AUTO 航線工時" in w for w in w_a) or not any("純飛行工時" in w for w in w_g):
        print(f"!! 工時警告未依基準標示: auto={w_a} guided={w_g}")
        fails += 1

    if not fails:
        print("工時基準 (duration_basis) 驗證 OK")
    return fails


def test_turn_in_place(cfg) -> int:
    """到航點停下原地轉頭 (waypoints.turn_in_place)：預設關閉不影響任務；開啟時 CONDITION_YAW + NAV_DELAY
    緊接在轉角航點後、朝向下一段、NAV_DELAY p1 > 0 且 p2..p4 = -1；DO_JUMP 重複經過時方向不繞遠路；
    估時含停下與懸停；精簡版仍守預算。"""
    from core.trajectory import (MissionPath, TurnInPlace, assign_turn_stops, enu_to_compass_deg,
                                 mission_polyline)
    from io_export.waypoints import (CMD_CONDITION_YAW, CMD_DO_CHANGE_SPEED, CMD_DO_JUMP, CMD_NAV_DELAY,
                                     CMD_SPLINE_WAYPOINT, CMD_WAYPOINT, trajectory_to_waypoints)

    fails = 0
    NAV = (CMD_WAYPOINT, CMD_SPLINE_WAYPOINT)
    on = deep_update(cfg, {"waypoints": {"turn_in_place": {"enabled": True}}})

    def rows_of(lines):
        return [ln.split("\t") for ln in lines[1:]]

    # 1) 預設關閉: 沒有 CONDITION_YAW / NAV_DELAY
    if TurnInPlace.from_config(cfg).enabled:
        print("!! turn_in_place 預設應關閉")
        fails += 1
    for key in ("reciprocate", "rectangle"):
        rows = rows_of(trajectory_to_waypoints(plan(key, cfg), cfg, "compact")[0])
        if any(int(r[3]) in (CMD_CONDITION_YAW, CMD_NAV_DELAY) for r in rows):
            print(f"!! {key}: 關閉時不應有 CONDITION_YAW / NAV_DELAY")
            fails += 1

    # 2a) 合成折線: 起飛點 (0,0) -> 進場 (1,0) 左轉 -> (1,1) 右轉 -> (2,1) 原路折返 -> (1,1) 收尾
    tip = TurnInPlace.from_config(on)
    if abs(tip.delay_s(90.0) - 3.0) > 1e-9 or abs(tip.delay_s(180.0) - 5.0) > 1e-9:
        print(f"!! 預設延遲 (45 deg/s + 1 s): 90° {tip.delay_s(90.0)} / 180° {tip.delay_s(180.0)}")
        fails += 1
    xs, ys, z3 = np.array([1.0, 2.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.ones(3)
    mp = MissionPath(xs, ys, z3, xs, ys, z3, use_spline=False, repeat="unroll", block_len=0, jump_repeat=0,
                     approach_x=np.array([1.0]), approach_y=np.array([0.0]), approach_z=np.ones(1),
                     home=(0.0, 0.0))
    assign_turn_stops(mp, on, is_smooth=False)
    expect = {  # (羅盤航向, CONDITION_YAW 方向, NAV_DELAY)
        "takeoff": (mp.turn_takeoff, 90.0, 0, tip.delay_s(180.0)),
        "approach0": (mp.turn_approach.get(0), 0.0, 0, tip.delay_s(90.0)),
        "nav0": (mp.turn_nav.get(0), 90.0, 0, tip.delay_s(90.0)),
        "nav1 (折返)": (mp.turn_nav.get(1), 270.0, 1, tip.delay_s(180.0)),
    }
    for k, (s, h, d, dl) in expect.items():
        if s is None or abs(s.heading_deg - h) > 1e-6 or s.direction != d or abs(s.delay_s - dl) > 1e-9:
            print(f"!! 合成折線 {k}: 得到 {s}, 預期 航向 {h} 方向 {d} 延遲 {dl}")
            fails += 1
    if 2 in mp.turn_nav:
        print("!! 最後一個導航點 (接 LAND) 不應轉頭")
        fails += 1

    # 2b) DO_JUMP block A,B: A 第一次從起飛點來 (右轉 90°), 之後每圈原路折返 (180°)。
    #     指定方向不得讓任何一次繞遠路: u_turn_dir=ccw 會讓第一次轉 270° -> 必須退回 CW
    for pref in ("cw", "ccw"):
        c = deep_update(on, {"waypoints": {"turn_in_place": {"u_turn_dir": pref}}})
        fx = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        mpj = MissionPath(np.array([0.0, 1.0, 0.0]), np.zeros(3), np.ones(3), fx, np.zeros(7), np.ones(7),
                          use_spline=False, repeat="do_jump", block_len=2, jump_repeat=2, home=(0.0, -1.0))
        assign_turn_stops(mpj, c, is_smooth=False)
        s0 = mpj.turn_nav.get(0)
        if s0 is None or s0.direction != 1 or abs(s0.turn_deg - 180.0) > 1e-6:
            print(f"!! DO_JUMP 第一點 (u_turn_dir={pref}): {s0} (應為 CW、最多轉 180°)")
            fails += 1
        if len(mpj.flown_turn_stops()) != 1 + 3 + 3:          # 起飛後 1 + A 每圈 1 + B 每圈 1
            print(f"!! DO_JUMP 實飛停下 {len(mpj.flown_turn_stops())} 次 != 7")
            fails += 1

    # 3) 實際 pattern (振幅 0, 固定 4 圈): 匯出格式、航向、DO_JUMP 目標、估時
    flat = deep_update(on, {"altitude": {"amplitude": 0.0}, "flight": {"laps": 4}})
    flat_off = deep_update(flat, {"waypoints": {"turn_in_place": {"enabled": False}}})
    for key, per_lap in (("reciprocate", 2), ("rectangle", 4)):
        pr = plan(key, flat)
        mp = mission_polyline(pr.trajectory, flat, pr.pattern.is_smooth, pr.pattern.lap_xy, pr.laps,
                              "compact", repeatable=pr.pattern.repeatable)
        rows = rows_of(trajectory_to_waypoints(pr, flat, "compact")[0])
        cmds = [int(r[3]) for r in rows]
        est = pr.trajectory.meta["auto_estimate"]["compact"]
        if len(mp.turn_nav) != per_lap or mp.turn_takeoff is None:
            print(f"!! {key}: 轉頭點 {len(mp.turn_nav)} != 每圈轉角 {per_lap} (或缺起飛後轉向)")
            fails += 1
        if cmds.count(CMD_CONDITION_YAW) != mp.n_turns or cmds.count(CMD_NAV_DELAY) != mp.n_turns \
           or len(rows) != 4 + mp.n_items or est["n_items"] != mp.n_items:
            print(f"!! {key}: CONDITION_YAW {cmds.count(CMD_CONDITION_YAW)} / NAV_DELAY {cmds.count(CMD_NAV_DELAY)}"
                  f" / 總項 {len(rows)} 與 n_turns {mp.n_turns} / n_items {mp.n_items} 不一致")
            fails += 1
        for i, cmd in enumerate(cmds):
            if cmd != CMD_CONDITION_YAW:
                continue
            r, prv, nxt = rows[i], rows[i - 1], rows[i + 1]
            if int(prv[3]) not in NAV + (CMD_DO_CHANGE_SPEED,) or int(nxt[3]) != CMD_NAV_DELAY:
                print(f"!! {key} seq {i}: CONDITION_YAW 必須緊接在導航航點 (或起飛設速) 後、NAV_DELAY 前")
                fails += 1
            if not (0.0 <= float(r[4]) < 360.0) or float(r[5]) <= 0 or int(float(r[6])) not in (-1, 0, 1) \
               or float(r[7]) != 0.0:
                print(f"!! {key} seq {i}: CONDITION_YAW 參數異常 {r[4:8]}")
                fails += 1
            if float(nxt[4]) < 0.5 or [float(v) for v in nxt[5:8]] != [-1.0, -1.0, -1.0]:
                print(f"!! {key} seq {i + 1}: NAV_DELAY 參數 {nxt[4:8]} (p1 必須 > 0, p2..p4 必須 -1)")
                fails += 1
        for i, s in mp.turn_nav.items():                       # 航向 = 從該點出發那段的水平方向
            h = enu_to_compass_deg(math.atan2(mp.y[i + 1] - mp.y[i], mp.x[i + 1] - mp.x[i]))
            if abs((s.heading_deg - h + 180.0) % 360.0 - 180.0) > 0.01:
                print(f"!! {key} nav {i}: 航向 {s.heading_deg} != 下一段方向 {h:.2f}")
                fails += 1
        jumps = [r for r in rows if int(r[3]) == CMD_DO_JUMP]
        if len(jumps) != 1 or cmds[int(float(jumps[0][4]))] not in NAV:
            print(f"!! {key}: DO_JUMP 目標不是導航航點")
            fails += 1
        est_off = plan(key, flat_off).trajectory.meta["auto_estimate"]["compact"]
        if est["turn_stops"] != per_lap * pr.laps + 1 or \
           abs(est["turn_s"] - sum(d for _, d in mp.flown_turn_stops())) > 1e-9:
            print(f"!! {key}: 實飛停下 {est['turn_stops']} 次 != {per_lap} × {pr.laps} + 1")
            fails += 1
        if est["nav_s"] < est_off["nav_s"] + est["turn_s"] - 0.05:
            print(f"!! {key}: 開啟後航線 {est['nav_s']:.1f}s 應 >= 關閉 {est_off['nav_s']:.1f}s"
                  f" + 懸停 {est['turn_s']:.1f}s")
            fails += 1

    # 4) 曲線 pattern: 預設只有起飛後轉向; curves=true + 低門檻才在曲線航點轉
    est = plan("circle", on).trajectory.meta["auto_estimate"]["compact"]
    if est["n_turns"] != 1:
        print(f"!! circle (curves=false): 轉頭點 {est['n_turns']} 應只有起飛後 1 個")
        fails += 1
    c = deep_update(on, {"waypoints": {"turn_in_place": {"curves": True, "min_turn_deg": 10}},
                         "flight": {"laps": 2}})
    if plan("circle", c).trajectory.meta["auto_estimate"]["compact"]["n_turns"] <= 1:
        print("!! circle curves=true 門檻 10°: 應有曲線航點轉頭")
        fails += 1

    # 5) after_takeoff=false: DO_CHANGE_SPEED 後直接是導航航點
    c = deep_update(flat, {"waypoints": {"turn_in_place": {"after_takeoff": False}}})
    rows = rows_of(trajectory_to_waypoints(plan("rectangle", c), c, "compact")[0])
    if int(rows[3][3]) not in NAV:
        print("!! after_takeoff=false: 第 3 項應為導航航點")
        fails += 1

    # 6) 角速度超過飛控上限 -> 延遲依 60 deg/s 估 + 警告; settle 0 / 門檻 0 -> 下限生效 (NAV_DELAY p1 <= 0 危險)
    c = deep_update(flat, {"waypoints": {"turn_in_place": {"rate_deg_s": 90}}})
    if abs(TurnInPlace.from_config(c).delay_s(180.0) - 4.0) > 1e-9:
        print(f"!! rate 90: 180° 延遲 {TurnInPlace.from_config(c).delay_s(180.0)} 應依 60 deg/s 估 = 4.0")
        fails += 1
    if not any("原地轉頭角速度" in w for w in plan("rectangle", c).report.warnings):
        print("!! rate 90 應警告超過 ATC_SLEW_YAW")
        fails += 1
    t0 = TurnInPlace.from_config(deep_update(on, {"waypoints": {"turn_in_place": {
        "settle_s": 0, "min_turn_deg": 0, "rate_deg_s": 180}}}))
    if t0.delay_s(0.0) < 0.5 or t0.min_turn_deg < 5.0:
        print(f"!! NAV_DELAY 下限 ({t0.delay_s(0.0)}) / min_turn_deg 下限 ({t0.min_turn_deg}) 未生效")
        fails += 1

    # 7) 精簡版預算: 轉頭項也算進總任務項
    tiny = deep_update(on, {"waypoints": {"fc_budget": 40, "repeat": "unroll", "z_tol": 0.0},
                            "altitude": {"amplitude": 0.0}, "flight": {"laps": 6}})
    for key in ("rectangle", "zigzag", "reciprocate"):
        lines, _ = trajectory_to_waypoints(plan(key, tiny), tiny, "compact")
        if len(lines) - 1 > 40:
            print(f"!! compact {key} (轉頭開啟): 總任務項 {len(lines) - 1} 超過預算 40")
            fails += 1

    # 8) 自動圈數以含懸停的 AUTO 航線逼近目標 (往返每圈 2 次 180° 折返 -> 圈數變少)
    la_off, la_on = plan("reciprocate", cfg).laps, plan("reciprocate", on).laps
    if not la_on < la_off:
        print(f"!! reciprocate: 開啟轉頭後自動圈數 {la_on} 應少於關閉時 {la_off}")
        fails += 1

    if not fails:
        print("到航點原地轉頭 (turn_in_place: CONDITION_YAW + NAV_DELAY) 驗證 OK")
    return fails


def main() -> int:
    cfg = load_config()
    print(box_summary(safe_box_from_config(cfg)))
    print(f"目標工時 {cfg['flight']['target_duration']}s, 速度 {cfg['flight']['cruise_speed']} m/s, "
          f"speed_profile={cfg['flight'].get('speed_profile', 'dynamic')}\n")

    failures = test_profile_analytic()
    failures += test_altitude_stair(cfg)
    failures += test_sparse_straight(cfg)
    failures += test_do_jump(cfg)
    failures += test_random_walk(cfg)
    failures += test_duration_basis(cfg)
    failures += test_turn_in_place(cfg)
    print()

    header = (f"{'pattern':<14}{'laps':>5}{'GUIDED':>8}{'naive':>7}{'AUTOnav':>9}{'AUTO(s)':>9}"
              f"{'len(m)':>9}{'vmax':>7}{'z range (m)':>14}{'  safety'}")
    print(header)
    print("-" * len(header))

    from core.planner import basis_duration
    cruise = float(cfg["flight"]["cruise_speed"])
    accel = float(cfg["flight"].get("accel", 1.0))
    lat = float(cfg["flight"].get("lat_accel", accel))

    for key, name in patterns.list_patterns():
        pr = plan(key, cfg)
        t = pr.trajectory
        m = t.meta
        b = t.bounds()
        zr = f"{b['z'][0]:.2f}~{b['z'][1]:.2f}"
        naive = m.get("naive_duration_s", 0.0)
        auto = (m.get("auto_estimate") or {}).get("compact", {})
        safe = "OK" if pr.report.ok else "ERR:" + "; ".join(pr.report.errors)
        warns = (" | " + "; ".join(pr.report.warnings)) if pr.report.warnings else ""
        print(f"{key:<14}{pr.laps:>5}{t.duration:>8.1f}{naive:>7.0f}"
              f"{auto.get('nav_s', 0):>9.0f}{auto.get('total_s', 0):>9.0f}{t.path_length_3d:>9.1f}"
              f"{t.max_speed:>7.2f}{zr:>14}   {safe}{warns}")

        # 斷言 (原有; 工時依 duration_basis 判定)
        if not pr.report.ok:
            failures += 1
        dur_b = basis_duration(t, cfg)
        if dur_b < cfg["flight"]["min_duration"] - 1:
            print(f"   !! {key} 工時 {dur_b:.0f}s < 下限")
            failures += 1
        box = pr.box
        inside = all(box.contains(x, y, z) for x, y, z in zip(t.x, t.y, t.z))
        if not inside:
            print(f"   !! {key} 有點超出安全盒")
            failures += 1
        if (b["z"][1] - b["z"][0]) < 0.2:
            print(f"   !! {key} z 幾乎沒變化，未達 3D")
            failures += 1
        if not (abs(t.x[0] - t.x[-1]) < 0.3 and abs(t.y[0] - t.y[-1]) < 0.3):
            print(f"   !! {key} 起點/終點 xy 未閉合")
            failures += 1

        # 斷言 (動態速度剖面)
        if t.duration < naive - 1.0:
            print(f"   !! {key} 動態工時 {t.duration:.0f}s 竟低於定速下界 {naive:.0f}s")
            failures += 1
        if m.get("avg_speed_h", 0.0) > cruise + 1e-3:
            print(f"   !! {key} 平均水平速度 {m['avg_speed_h']:.2f} 超過巡航 {cruise}")
            failures += 1
        vh = np.hypot(t.vx, t.vy)
        a98 = sustained_accel_p98(t)
        if a98 > 1.3 * math.hypot(accel, lat):
            print(f"   !! {key} 持續水平加速度 P98 {a98:.2f} m/s² 超出可行範圍")
            failures += 1
        if key == "reciprocate":
            # 折返點應近懸停
            interior = vh[t.n // 10: -t.n // 10]
            if float(interior.min()) > 0.12:
                print(f"   !! reciprocate 折返最低速 {interior.min():.2f} 未近懸停")
                failures += 1
        if not auto:
            print(f"   !! {key} 缺 AUTO 工時預估")
            failures += 1
        elif auto["total_s"] <= auto["nav_s"]:
            print(f"   !! {key} AUTO 總時間未含起降")
            failures += 1

    # 舊版定速模式迴歸: 工時應 ≈ naive 公式
    cfg_const = deep_update(cfg, {"flight": {"speed_profile": "constant"}})
    pr_c = plan("circle", cfg_const)
    naive_c = pr_c.trajectory.meta["naive_duration_s"]
    if abs(pr_c.trajectory.duration - naive_c) > 1.0:
        print(f"!! constant 模式工時 {pr_c.trajectory.duration:.1f}s != naive {naive_c:.1f}s")
        failures += 1
    else:
        print("\nconstant 模式迴歸 OK (工時 == 弧長/定速)")

    print()
    if failures:
        print(f"FAILED: {failures} 項問題")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
