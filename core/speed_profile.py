"""
動態速度剖面：沿折線的時間最佳化參數化 (梯形 + S 曲線 jerk 近似)。

室內多旋翼實飛不是全程定速，工時預估必須考慮：
  * 起步/收尾 —— 從懸停加速、減速回懸停 (加速度 ≈ WPNAV_ACCEL 等級)
  * 硬轉角 —— 矩形 90°、鋸齒、往返 180°。能通過轉角的速度由「容許切角半徑」
    決定：對 AUTO 是航點接受半徑 WPNAV_RADIUS (本專案 5 cm 級)，對 GUIDED 是
    想維持的追蹤容差。內切圓弧半徑 R = r·cos(δ/2)/(1−cos(δ/2))，δ=轉向角，
    過彎速度 v = sqrt(lat_accel·R)；180° 折返 R→0，等同短暫懸停再出發。
    δ 一律以『水平 (XY)』幾何計算 —— 飛控的轉角邏輯是水平面的，3D 折線 (AUTO 估時)
    的高度變化不該把 180° 折返「看」成較緩的轉角。
  * 平滑曲線 —— 圓/8字受向心加速度限制 v <= sqrt(lat_accel/κ)。
  * S 曲線 jerk —— ArduPilot 4.x 為 jerk 受限 (WPNAV_JERK)。室內低速下速度變化
    Δv 多半達不到加速度上限, 加/減速由 jerk 主導: 一段 Δv 的耗時 ≈ 2·sqrt(Δv/j)
    (與 a 無關)。我們把它折成「有效加速度」a_eff = min(a, ½·sqrt(j·v_scale))
    直接餵給前向/後向掃描 —— 時間與位置同步一致 (不用事後拉長時間, 避免高 a
    時斜坡距離被壓成近乎瞬間、把過多距離丟給巡航段而反讓工時變長的假象)。
    v_scale 取巡航速度 (最大的一次速度變化)。
  * 3D 折線 (AUTO 估時) —— cruise 與過彎上限是『水平』速度 (WPNAV_SPEED)，沿斜段的
    3D 速度 = 水平速度 / cos(傾角)；垂直分量另受上升/下降速度上限 (WPNAV_SPEED_UP/DN)
    限制。與 GUIDED 軌跡 (水平定巡航 + 疊加 z(t)) 的合成速度一致。

流程：在『原折線』上算各頂點過彎上限 (轉角切角 ∧ 曲率, 曲率用原始頂點間弧長, 不隨
      取樣密度漂移) -> 折線加密 (densify) 並把上限貼回對應點 -> 疊加垂直上限 ->
      以 a_eff 前向/後向掃描 (jerk 一致的加減速可達性) -> 梯形積分時間。

使用者：core/trajectory.py (GUIDED 重取時)、core/timing.py (圈數決策與
AUTO 任務工時預估)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# 折線加密解析度 (m)：太粗會低估轉角減速區間，太細只是變慢
DENSIFY_STEP = 0.05

# 轉角分類門檻：單頂點轉向角小於此視為平滑曲率 (密取樣曲線)，之上視為硬轉角
CORNER_ANGLE_MIN = math.radians(15.0)

_EPS = 1e-9


@dataclass
class ProfileParams:
    """速度剖面的動力學參數 (單位 m / s)。預設對應保守的室內 ArduCopter 設定。"""

    cruise: float                 # 巡航速度上限 (DO_CHANGE_SPEED / WPNAV_SPEED)
    accel: float = 1.0            # 沿軌加/減速上限 (m/s²); 1.0 = WPNAV_ACCEL 100 cm/s²
    lat_accel: float = 1.0        # 轉彎向心加速度上限 (m/s²); 保守設 = accel。
    #                               實機 AUTO 轉角加速度 WPNAV_ACCEL_C 內定 2×WPNAV_ACCEL
    jerk: float = 1.0             # S 曲線急動度 (m/s³) = WPNAV_JERK; <=0 不修正
    accept_radius: float = 0.05   # 轉角容許切角半徑 (m) = 航點接受半徑 (5 cm 級)
    v_floor: float = 0.05         # 硬轉角最低通過速度 (m/s); 避免時間積分發散
    vz_up: float = 1.0            # 上升速度上限 (m/s), 僅 3D 折線使用
    vz_dn: float = 0.6            # 下降速度上限 (m/s), 僅 3D 折線使用

    @classmethod
    def from_config(cls, cfg: Dict) -> "ProfileParams":
        f = cfg.get("flight", {})
        w = cfg.get("waypoints", {})
        accel = float(f.get("accel", 1.0))
        return cls(
            cruise=float(f.get("cruise_speed", 0.5)),
            accel=accel,
            lat_accel=float(f.get("lat_accel", accel)),
            jerk=float(f.get("jerk", 1.0)),
            accept_radius=float(w.get("accept_radius", 0.05)),
            v_floor=float(f.get("corner_min_speed", 0.05)),
            vz_up=float(f.get("speed_up", 1.0)),
            vz_dn=float(f.get("speed_down", 0.6)),
        )


@dataclass
class SpeedProfile:
    """折線的動態速度剖面 (各欄位對齊加密後折線頂點)。"""

    points: np.ndarray        # (N, 2|3) 加密後折線
    s: np.ndarray             # (N,) 累積弧長 (m)
    v: np.ndarray             # (N,) 沿軌速度 (m/s)
    t: np.ndarray             # (N,) 時間 (s), 由 0 起
    hard_corners: int         # 硬轉角數 (含每圈重複)
    min_corner_speed: float   # 硬轉角最低通過速度 (無硬轉角時 = cruise)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


# ----------------------------------------------------------------------
# 幾何前處理
# ----------------------------------------------------------------------
def _dedupe(poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """移除連續重複點 (避免零長線段)。"""
    if len(poly) <= 1:
        return poly
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    keep = np.concatenate([[True], d > tol])
    return poly[keep]


def _densify(poly: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """把每段細分到 <= step，保留原頂點 (轉角位置/角度不變)。

    回傳 (dense, orig_idx)：orig_idx[i] 是第 i 個原頂點在 dense 中的索引。
    轉角/曲率速度上限在『原折線』上計算 (見 _corner_caps) 後再靠 orig_idx 貼回
    dense，避免用『加密步長』當弧長而讓曲率上限隨取樣密度漂移。
    """
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly.copy(), np.arange(len(poly))
    seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    n_per = np.maximum(1, np.ceil(seg_len / max(step, 1e-4)).astype(int))   # 每段細分數
    # 向量化: 第 i 段產生 k=1..n_i 的點 a + (b-a)*k/n_i (與逐段迴圈結果完全相同)
    seg_idx = np.repeat(np.arange(len(seg_len)), n_per)
    starts = np.cumsum(n_per) - n_per
    k = np.arange(int(n_per.sum())) - np.repeat(starts, n_per) + 1
    frac = k / n_per[seg_idx]
    new_pts = poly[seg_idx] + (poly[seg_idx + 1] - poly[seg_idx]) * frac[:, None]
    dense = np.vstack([poly[:1], new_pts])
    orig_idx = np.concatenate([[0], np.cumsum(n_per)]).astype(int)
    return dense, orig_idx


def _xy_dirs(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """各段的『水平』方向向量與長度 (只取 XY 分量)。

    純垂直段 (XY 零長) 沿用最近一段的有效水平方向：一個「先垂直升降、再轉向」的頂點，
    轉角才會在正確的頂點被偵測 (否則會被零向量吃掉)。
    """
    u = np.diff(poly[:, :2], axis=0)
    n = np.linalg.norm(u, axis=1)
    ok = n > _EPS
    if ok.any() and not ok.all():
        idx = np.arange(len(ok))
        last = np.maximum.accumulate(np.where(ok, idx, 0))     # 最近的有效段 (往前找)
        first = int(np.argmax(ok))
        last = np.where(idx < first, first, last)              # 開頭的零長段沿用第一個有效段
        u, n = u[last], n[last]
    return u, n


def _turn_angles(poly: np.ndarray) -> np.ndarray:
    """各內部頂點的『水平』轉向角 δ (rad, 0=直行, π=原路折返)。首尾為 0。

    只用 XY 分量：飛控的轉角/切角邏輯是水平面的 (WPNAV 接受半徑與過彎速度)，3D 折線
    的高度變化不該讓 180° 折返「看起來」變成較緩的轉角而被高估過彎速度。
    """
    u, n = _xy_dirs(poly)
    dot = np.einsum("ij,ij->i", u[:-1], u[1:])
    denom = n[:-1] * n[1:]
    cosd = np.ones_like(dot)
    ok = denom > _EPS
    cosd[ok] = np.clip(dot[ok] / denom[ok], -1.0, 1.0)
    delta = np.zeros(len(poly))
    delta[1:-1] = np.arccos(cosd)
    return delta


# ----------------------------------------------------------------------
# 逐點速度上限
# ----------------------------------------------------------------------
def _corner_caps(poly: np.ndarray, prm: ProfileParams) -> Tuple[np.ndarray, int, float]:
    """在『原始』折線上算各頂點過彎速度上限。回傳 (caps, 硬轉角數, 硬轉角最低速)。

    轉向角 δ 與曲率都以『水平 (XY)』幾何計算 (飛控過彎邏輯是水平面的; 3D 折線的高度
    變化由 _apply_vertical_limits 另行限制)。每個頂點取兩種上限的較小值：
      * 切角圓弧 (5cm 接受半徑決定): R = r·cos(δ/2)/(1−cos(δ/2)), v = sqrt(lat·R)。
        僅由轉向角 δ 決定, 與取樣密度無關 → 直角/折返等真實硬轉角。
      * 連續曲率: κ = δ / 相鄰『原始』段平均水平弧長, v = sqrt(lat/κ)。用原始頂點間距
        (非加密步長) 才會收斂到真實 1/R, 不隨取樣密度漂移 (密取樣曲線由此綁定)。
    取 min 讓兩種情況都自然涵蓋, 免去對取樣密度敏感的硬/平滑門檻切換。
    """
    _, seg = _xy_dirs(poly)          # 水平段長 (純垂直段沿用鄰段, 僅供曲率的弧長尺度)
    delta = _turn_angles(poly)
    caps = np.full(len(poly), prm.cruise, dtype=float)
    hard = 0
    vmin_corner = prm.cruise

    for i in range(1, len(poly) - 1):
        d = float(delta[i])
        if d < 1e-4:
            continue
        c = math.cos(0.5 * d)
        fillet_r = prm.accept_radius * c / max(1.0 - c, _EPS)
        v_fillet = math.sqrt(prm.lat_accel * fillet_r)
        ds = 0.5 * (seg[i - 1] + seg[i])
        kappa = d / max(ds, _EPS)
        v_curv = math.sqrt(prm.lat_accel / kappa) if kappa > _EPS else prm.cruise
        v = max(min(v_fillet, v_curv), prm.v_floor)
        caps[i] = min(caps[i], v)
        if d >= CORNER_ANGLE_MIN:            # 僅用於「硬轉角」統計 (顯示用)
            hard += 1
            vmin_corner = min(vmin_corner, v)

    return caps, hard, vmin_corner


def _apply_vertical_limits(
    vlim: np.ndarray, dense: np.ndarray, seg: np.ndarray, prm: ProfileParams
) -> np.ndarray:
    """3D：垂直分量限制 v·(|dz|/ds) <= vz_up (上升) / vz_dn (下降)。

    seg 為 dense 的 3D 段長 -> slope ∈ [0,1]，slope=1 (純垂直) 時上限即 vz_cap。
    在加密後折線上逐段套用 (與曲率上限不同, 垂直上限本就是逐段幾何量)。
    """
    if dense.shape[1] < 3:
        return vlim
    dz = np.diff(dense[:, 2])
    slope = np.abs(dz) / np.maximum(seg, _EPS)
    vz_cap = np.where(dz >= 0, prm.vz_up, prm.vz_dn)
    vseg = np.where(slope > _EPS, vz_cap / np.maximum(slope, _EPS), np.inf)
    vseg = np.maximum(vseg, prm.v_floor)
    out = vlim.copy()
    out[:-1] = np.minimum(out[:-1], vseg)
    out[1:] = np.minimum(out[1:], vseg)
    return out


# ----------------------------------------------------------------------
# 前向/後向掃描 + 時間積分
# ----------------------------------------------------------------------
def _forward_backward(
    s: np.ndarray, vlim: np.ndarray, accel: float, v_start: float, v_end: float
) -> np.ndarray:
    v = vlim.copy()
    v[0] = min(v[0], max(v_start, 0.0))
    for i in range(1, len(v)):
        ds = s[i] - s[i - 1]
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * accel * ds))
    v[-1] = min(v[-1], max(v_end, 0.0))
    for i in range(len(v) - 2, -1, -1):
        ds = s[i + 1] - s[i]
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * accel * ds))
    return v


def _integrate_time(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """梯形積分各段時間 (等加速度段下為精確解)。"""
    ds = np.diff(s)
    vm = np.maximum(v[:-1] + v[1:], _EPS)
    return 2.0 * ds / vm


def effective_accel(accel: float, jerk: float, v_scale: float) -> float:
    """把 jerk 上限折成「有效加速度」(m/s²)。

    速度變化 v_scale 的 jerk 受限耗時為 2·sqrt(v_scale/j)，對應平均加速度
    ½·sqrt(j·v_scale)。若它小於原加速度上限 a，代表這段變速根本達不到 a、
    由 jerk 主導 -> 用較小的 a_eff。jerk<=0 表示不限 jerk -> 回傳原 a。

    以單一 a_eff (取 v_scale = 巡航速度) 做整段掃描: 對主導的巡航級變速精確,
    對較小的轉角變速略樂觀 (次要項), 且對 a 單調 (a 越大工時越短或持平)。

    注意: 只在 jerk 主導區 (cruise <= accel²/jerk) 為精確解; 若 cruise > accel²/jerk
    (變速其實達得到加速度上限, 例如高 cruise+低 jerk) 會略微『樂觀』(低估斜坡時間)。
    室內預設 (cruise≤max_speed=1.0, accel=jerk=1.0) 恆在 jerk 主導區, 誤差為 0。
    """
    if jerk <= 0 or accel <= 0:
        return accel
    return min(accel, 0.5 * math.sqrt(jerk * max(v_scale, _EPS)))


# ----------------------------------------------------------------------
# 對外 API
# ----------------------------------------------------------------------
def compute_profile(
    poly: np.ndarray,
    prm: ProfileParams,
    v_start: float = 0.0,
    v_end: float = 0.0,
    step: float = DENSIFY_STEP,
) -> SpeedProfile:
    """對折線 (Nx2 或 Nx3, 公尺) 計算動態速度剖面。

    v_start / v_end: 起點/終點速度 (預設懸停出發、懸停結束)。
    """
    poly = _dedupe(np.asarray(poly, dtype=float))
    if len(poly) < 2:
        pts = poly if len(poly) else np.zeros((1, 2))
        zero = np.zeros(len(pts))
        return SpeedProfile(points=pts, s=zero, v=zero.copy(), t=zero.copy(),
                            hard_corners=0, min_corner_speed=prm.cruise)

    caps_orig, hard, vmin_corner = _corner_caps(poly, prm)

    dense, orig_idx = _densify(poly, step)
    if len(dense) < 3:
        # 單一短段 (全長 <= step)：補一個中點，否則兩端皆靜止 -> v 全零 -> dt 發散
        mid = 0.5 * (dense[0] + dense[-1])
        dense = np.vstack([dense[0], mid, dense[-1]])
        orig_idx = np.array([0, len(dense) - 1])

    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    vlim = np.full(len(dense), prm.cruise, dtype=float)
    vlim[orig_idx] = caps_orig                       # 原頂點的過彎上限貼回 dense
    if dense.shape[1] >= 3:
        # 巡航 (WPNAV_SPEED) 與過彎上限都是『水平』速度；沿 3D 路徑的速度上限
        # = v_h · ds3d/dsxy (斜段可以走得比 cruise 快, 垂直分量另由下面的上升/下降上限管)。
        # 頂點取相鄰兩段中較平的一側 (保守)；純垂直段沒有水平限制 (inf), 交給垂直上限。
        seg_xy = np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1)
        ratio = np.where(seg_xy > _EPS, seg / np.maximum(seg_xy, _EPS), np.inf)
        r_pt = np.full(len(dense), np.inf)
        r_pt[:-1] = np.minimum(r_pt[:-1], ratio)
        r_pt[1:] = np.minimum(r_pt[1:], ratio)
        vlim = vlim * r_pt
    vlim = _apply_vertical_limits(vlim, dense, seg, prm)

    a_eff = effective_accel(prm.accel, prm.jerk, prm.cruise)
    v = _forward_backward(s, vlim, a_eff, v_start, v_end)
    dt = _integrate_time(s, v)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return SpeedProfile(points=dense, s=s, v=v, t=t,
                        hard_corners=hard, min_corner_speed=vmin_corner)


def predict_duration(poly: np.ndarray, prm: ProfileParams, **kw) -> float:
    """折線的預估飛行時間 (s, 懸停出發/結束)。"""
    return compute_profile(poly, prm, **kw).duration


def resample_profile(prof: SpeedProfile, dt: float):
    """以固定 dt 沿剖面重取樣 -> (t, x, y[, z]) 均勻時間格點。"""
    duration = prof.duration
    n = max(2, int(round(duration / max(dt, 1e-4))) + 1)
    tg = np.linspace(0.0, duration, n)
    sg = np.interp(tg, prof.t, prof.s)
    cols = tuple(
        np.interp(sg, prof.s, prof.points[:, k])
        for k in range(prof.points.shape[1])
    )
    return (tg,) + cols
