"""
軌跡資料結構與建構流程。

設計：每個 pattern 只負責產生「單圈、起點==終點」的水平折線 (Nx2, 公尺, 已縮放到
SafeBox)。本模組負責：
  1. 依 timing 決定的圈數把單圈接成連續折線
  2. 時間參數化 —— 預設用動態速度剖面 (core/speed_profile: 加減速 + 轉角減速
     + 曲率 + jerk, 貼近室內多旋翼實飛)；flight.speed_profile=constant 則退回
     舊版定速取樣 (對照用)
  3. 疊加垂直高度調變 z(t) 使其成為 3D
  4. 計算速度與機頭朝向 yaw (含最大角速度限制, 轉角處 yaw 才追得上)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import SafeBox, takeoff_point
from .speed_profile import ProfileParams, compute_profile, resample_profile


@dataclass
class Trajectory:
    """時間參數化的 3D 軌跡 (房間本地 ENU; x=東, y=北, z=上)。"""

    name: str
    t: np.ndarray          # (N,) 秒, 由 0 起
    x: np.ndarray          # (N,) 東 (m)
    y: np.ndarray          # (N,) 北 (m)
    z: np.ndarray          # (N,) 上 (m, 相對地板)
    yaw: np.ndarray        # (N,) 弧度 (ENU: 0=+x 東, 逆時針為正)
    vx: np.ndarray         # (N,) m/s
    vy: np.ndarray
    vz: np.ndarray
    meta: Dict = field(default_factory=dict)
    s: np.ndarray | None = None   # (N,) 沿水平頂點折線的累積弧長 (m); 高度 z = f(s/S)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def path_length_3d(self) -> float:
        d = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2 + np.diff(self.z) ** 2)
        return float(np.sum(d))

    @property
    def speed_profile(self) -> np.ndarray:
        return np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)

    @property
    def max_speed(self) -> float:
        return float(np.max(self.speed_profile)) if self.n else 0.0

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "x": (float(self.x.min()), float(self.x.max())),
            "y": (float(self.y.min()), float(self.y.max())),
            "z": (float(self.z.min()), float(self.z.max())),
        }


# ----------------------------------------------------------------------
# 折線工具
# ----------------------------------------------------------------------
def _dedupe(poly: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """移除連續重複點 (避免零長線段)。"""
    if len(poly) <= 1:
        return poly
    keep = [0]
    for i in range(1, len(poly)):
        if np.hypot(*(poly[i] - poly[keep[-1]])) > tol:
            keep.append(i)
    return poly[keep]


def arclength(poly: np.ndarray) -> float:
    """折線總長 (公尺)。"""
    poly = _dedupe(poly)
    if len(poly) < 2:
        return 0.0
    d = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    return float(np.sum(d))


def tile_closed(lap_xy: np.ndarray, laps: int) -> np.ndarray:
    """把「起點==終點」的單圈接成 laps 圈的連續折線 (接縫去重)。"""
    lap_xy = np.asarray(lap_xy, dtype=float)
    if laps <= 1:
        return lap_xy.copy()
    parts = [lap_xy]
    for _ in range(laps - 1):
        parts.append(lap_xy[1:])  # 丟掉與上一圈終點重複的起點
    return np.vstack(parts)


def _merge_anchors(target: np.ndarray, anchors, total: float, spacing: float) -> np.ndarray:
    """把必留的弧長位置 (anchors) 併進等分取樣點：太靠近錨點的等分點拿掉 (免得出現極短段), 首尾保留。"""
    if anchors is None:
        return target
    a = np.asarray(anchors, dtype=float)
    a = a[(a > 1e-9) & (a < total - 1e-9)]
    if len(a) == 0:
        return target
    keep = np.ones(len(target), dtype=bool)
    for v in a:
        near = np.abs(target - v) < 0.25 * spacing
        near[0] = near[-1] = False
        keep &= ~near
    return np.unique(np.concatenate([target[keep], a]))


def resample_by_spacing(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, spacing: float,
    max_points: int | None = None, anchors=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿 3D 弧長以固定間距取樣 (航點匯出與 AUTO 工時預估共用同一套取樣)。

    max_points: 若指定且依間距取樣會超過此數，改用等分 (linspace) 取剛好 max_points 點，
                以保證點數不超過上限 (精簡/上飛控版用)。
    anchors:    必須保留的 3D 弧長位置 (例如越過箱子的斜坡轉折點), 會併進取樣點。
    """
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total <= 0:
        return x[:1], y[:1], z[:1]
    # 等分 (段長 ≈ spacing, 含起終點)；不用 arange+終點, 免得最後一段只剩零頭 (DO_JUMP 一圈
    # block 的收尾段會每圈重複, 太短的段會讓飛控在那裡減速)
    n = max(1, int(round(total / max(spacing, 1e-3))))
    target = _merge_anchors(np.linspace(0.0, total, n + 1), anchors, total, spacing)
    if max_points is not None and len(target) > max_points:
        target = np.linspace(0.0, total, max(2, int(max_points)))
    return (
        np.interp(target, s, x),
        np.interp(target, s, y),
        np.interp(target, s, z),
    )


# ----------------------------------------------------------------------
# AUTO 任務 (.waypoints) 的航點折線 —— 匯出與工時預估共用同一套取樣規則，
# 保證「估時所用折線 == 實際匯出折線」
# ----------------------------------------------------------------------
DEFAULT_FC_BUDGET = 650        # 精簡版「總任務項」上限 (缺省後備; 與 config/default.yaml 一致)
MISSION_OVERHEAD_ITEMS = 4     # 非導航固定項: HOME + TAKEOFF + DO_CHANGE_SPEED + LAND/RTL
DEFAULT_Z_TOL = 0.02           # 直線段航點精簡的高度容差 (m); 與 config/default.yaml 一致

# ---- 到航點停下、原地轉頭再前進 (waypoints.turn_in_place; AUTO 任務專用) ----
# 每個轉頭點在導航航點後面多寫兩項: CONDITION_YAW(朝下一段) + NAV_DELAY(等轉完)。
# ArduCopter 的 DO/CONDITION 命令是在「前一個導航命令完成 (到點)」時才開始, 且與下一個導航命令並行；
# 下一個導航命令是 NAV_DELAY 時前一個航點會完全停下 (不做 fast waypoint), 飛機原地懸停轉頭。
TURN_ITEMS_PER_STOP = 2
_U_TURN_BAND_DEG = 5.0         # |轉向角| >= 180° − 此值 視為原路折返: 最短方向不明確, 改用 u_turn_dir 指定
YAW_SLEW_DEFAULT_DEG_S = 60.0  # CONDITION_YAW 實際角速度 = min(param2, 飛控上限); 上限 ATC_SLEW_YAW 預設 6000 cdeg/s
                               # (新版改名 ATC_RATE_WPY_MAX, 預設也是 60 deg/s) -> NAV_DELAY 以兩者較小值估
MIN_TURN_DELAY_S = 0.5         # NAV_DELAY 秒數下限: ArduCopter 的 p1 <= 0 代表「等到 UTC 時:分:秒」, 絕不能寫 0
MIN_TURN_DEG_FLOOR = 5.0       # min_turn_deg 下限 (幾乎直行的點不值得停; 也避免指定方向時的小角度誤差)


@dataclass(frozen=True)
class TurnInPlace:
    """waypoints.turn_in_place 設定 (缺省 = 關閉, 任務與舊版逐字元相同)。"""

    enabled: bool = False
    min_turn_deg: float = 30.0   # 水平轉向角 >= 此值的航點才停下轉頭
    rate_deg_s: float = 45.0     # CONDITION_YAW 轉頭角速度 (飛控另受 ATC_SLEW_YAW 限制)
    settle_s: float = 1.0        # 轉完後多停的餘裕 (s)
    curves: bool = False         # 曲線 pattern (圓/8字/隨機手飛) 的航點是否也套用
    u_turn_dir: int = 1          # 原路折返的轉向: 1 = 順時針 (cw), -1 = 逆時針 (ccw)
    after_takeoff: bool = True   # 起飛到高度後先原地轉向第一段再出發

    @classmethod
    def from_config(cls, cfg: Dict) -> "TurnInPlace":
        t = (cfg.get("waypoints", {}) or {}).get("turn_in_place") or {}
        if not isinstance(t, dict):
            t = {"enabled": bool(t)}
        d = cls()
        return cls(
            enabled=bool(t.get("enabled", d.enabled)),
            min_turn_deg=max(MIN_TURN_DEG_FLOOR, float(t.get("min_turn_deg", d.min_turn_deg))),
            rate_deg_s=max(1.0, float(t.get("rate_deg_s", d.rate_deg_s))),
            settle_s=max(0.0, float(t.get("settle_s", d.settle_s))),
            curves=bool(t.get("curves", d.curves)),
            u_turn_dir=-1 if str(t.get("u_turn_dir", "cw")).strip().lower() == "ccw" else 1,
            after_takeoff=bool(t.get("after_takeoff", d.after_takeoff)),
        )

    def delay_s(self, rotation_deg: float) -> float:
        """NAV_DELAY 秒數 = 要轉的角度 / min(角速度, 飛控上限) + settle, 進位到 0.1 s, 且 >= MIN_TURN_DELAY_S。"""
        rate = min(self.rate_deg_s, YAW_SLEW_DEFAULT_DEG_S)
        return max(MIN_TURN_DELAY_S, math.ceil((rotation_deg / rate + self.settle_s) * 10.0 - 1e-6) / 10.0)


@dataclass(frozen=True)
class TurnStop:
    """一個轉頭點 = 導航航點後面的 CONDITION_YAW + NAV_DELAY。"""

    heading_deg: float   # CONDITION_YAW param1: 絕對航向 (羅盤: 0=北/+y, 90=東/+x, 順時針) = 下一段水平方向
    rate_deg_s: float    # param2: 轉頭角速度 (deg/s)
    direction: int       # param3: 0 = 最短方向, 1 = 順時針, -1 = 逆時針
    turn_deg: float      # 實際要轉的角度 (deg; DO_JUMP 重複經過時取最大)
    delay_s: float       # NAV_DELAY 秒數 = turn_deg / rate + settle (進位到 0.1 s)


def enu_to_compass_deg(yaw_enu_rad: float) -> float:
    """ENU yaw (0=東, 逆時針) -> 羅盤航向 (0=北, 順時針, [0, 360))。"""
    return float((90.0 - np.degrees(yaw_enu_rad)) % 360.0)


@dataclass
class MissionPath:
    """AUTO 任務的航點折線 (匯出 + 估時共用)。

    x/y/z      : 要寫進 .waypoints 的導航點。repeat='do_jump' 時 = 一圈 block (block_len 點)
                 + 收尾點 (= 起點, DO_JUMP 跑完後回到它再 LAND)；'unroll' 時 = 全程展開折線。
    flown_*    : 實際會飛的完整 3D 折線 (do_jump = block 重複 laps 圈; unroll = 同 x/y/z)，估時用。
    use_spline : 曲線用 NAV_SPLINE_WAYPOINT。
    jump_repeat: DO_JUMP 的重複次數 (= 圈數 - 1; 0 = 不寫 DO_JUMP)。
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    flown_x: np.ndarray
    flown_y: np.ndarray
    flown_z: np.ndarray
    use_spline: bool
    repeat: str
    block_len: int
    jump_repeat: int
    # 進場段 (HOME 起飛點 -> 第一個航點) 為了繞開障礙物插入的航點 (只飛一次, 在 DO_JUMP block 之前);
    # 沒有障礙物或直線可達時為空
    approach_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    approach_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    approach_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # 起飛點 (房間 ENU 水平座標)。存在 MissionPath 上而不是各呼叫端各自傳入 ——
    # 這樣估時 / 安全檢查 / 匯出拿到的第一段一定是同一段, 不會有人漏傳而靜默退回原點。
    home: Tuple[float, float] = (0.0, 0.0)
    # 原地轉頭點 (waypoints.turn_in_place): 進場航點 index / 導航點 index (x 的 index) -> TurnStop。
    # 關閉時為空。由 assign_turn_stops 填入 (匯出與估時都從這裡讀, 保證一致)。
    turn_approach: Dict[int, TurnStop] = field(default_factory=dict)
    turn_nav: Dict[int, TurnStop] = field(default_factory=dict)
    turn_takeoff: Optional[TurnStop] = None     # 起飛到高度後、飛第一段前的原地轉向 (寫在 DO_CHANGE_SPEED 後面)

    @property
    def n_approach(self) -> int:
        return int(len(self.approach_x))

    @property
    def n_nav(self) -> int:
        """寫進任務的導航航點數 (含進場段)。"""
        return int(len(self.x)) + self.n_approach

    @property
    def n_turns(self) -> int:
        """寫進任務的轉頭點數 (每個 = CONDITION_YAW + NAV_DELAY 兩項; DO_JUMP 重複不另計)。"""
        return len(self.turn_approach) + len(self.turn_nav) + (1 if self.turn_takeoff is not None else 0)

    @property
    def n_items(self) -> int:
        """導航航點 + DO_JUMP + 轉頭 (CONDITION_YAW/NAV_DELAY) 的任務項數 (不含 HOME/TAKEOFF/SPEED/LAND 固定項)。"""
        return self.n_nav + (1 if self.jump_repeat > 0 else 0) + TURN_ITEMS_PER_STOP * self.n_turns

    def item_occurrences(self) -> List[Tuple[str, int, List[int]]]:
        """每個導航任務項在「實飛折線 flown_with_approach」裡出現的位置。

        回傳 [(kind, index, [flown_with_approach 的 index, ...]), ...]；kind = 'approach' | 'nav'。
        DO_JUMP block 裡的航點每圈經過一次 (多個位置)；進場點、收尾點與 unroll 各只有一個。
        flown_with_approach 的第 0 點是起飛點 (不是任務項)。
        """
        na = self.n_approach
        out: List[Tuple[str, int, List[int]]] = [("approach", i, [1 + i]) for i in range(na)]
        base = 1 + na
        if self.repeat == "do_jump" and self.jump_repeat > 0:
            bl, laps = self.block_len, self.jump_repeat + 1
            out += [("nav", j, [base + j + k * bl for k in range(laps)]) for j in range(bl)]
            out += [("nav", i, [base + bl * laps + (i - bl)]) for i in range(bl, len(self.x))]
        else:
            out += [("nav", i, [base + i]) for i in range(len(self.x))]
        return out

    def flown_turn_stops(self) -> List[Tuple[int, float]]:
        """實飛折線 (flown_with_approach) 上每一次「停下轉頭」的 (index, NAV_DELAY 秒數), 依 index 排序。
        DO_JUMP block 裡的轉頭點每圈都會停一次。估時用。"""
        if not self.n_turns:
            return []
        stops = [(0, self.turn_takeoff.delay_s)] if self.turn_takeoff is not None else []
        for kind, i, occ in self.item_occurrences():
            ts = (self.turn_approach if kind == "approach" else self.turn_nav).get(i)
            if ts is not None:
                stops += [(f, ts.delay_s) for f in occ]
        return sorted(stops)

    def turn_points(self, takeoff_alt: float) -> np.ndarray:
        """轉頭點的位置 (Nx3, 房間 ENU; 每個任務項一次, 起飛後的轉向在起飛點正上方)。預覽標示用。"""
        pts = [(self.home[0], self.home[1], takeoff_alt)] if self.turn_takeoff is not None else []
        pts += [(self.approach_x[i], self.approach_y[i], self.approach_z[i]) for i in sorted(self.turn_approach)]
        pts += [(self.x[i], self.y[i], self.z[i]) for i in sorted(self.turn_nav)]
        return np.asarray(pts, dtype=float).reshape(-1, 3)

    def flown_with_approach(self, takeoff_alt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """起飛點 (home, 起飛高) -> 進場航點 -> 完整實飛折線 (估時 / 安全檢查用)。"""
        return (
            np.concatenate([[float(self.home[0])], self.approach_x, self.flown_x]),
            np.concatenate([[float(self.home[1])], self.approach_y, self.flown_y]),
            np.concatenate([[float(takeoff_alt)], self.approach_z, self.flown_z]),
        )


def thin_polyline(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, max_points: int | None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """點數超過 max_points 時沿 3D 弧長等分重取樣 (精簡/上飛控版用)；否則原樣回傳。"""
    if max_points is None or len(x) <= max_points:
        return x, y, z
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return x[:1], y[:1], z[:1]
    target = np.linspace(0.0, float(s[-1]), max(2, int(max_points)))
    return np.interp(target, s, x), np.interp(target, s, y), np.interp(target, s, z)


def simplify_sz(s: np.ndarray, z: np.ndarray, tol: float) -> np.ndarray:
    """(弧長, 高度) 折線的 Douglas-Peucker：回傳保留點的 index (首尾必留)。

    直線段上 XY 本來就共線，只有高度會偏離弦；用高度偏差 <= tol 決定能不能拿掉中間點：
      * 平飛 (振幅 0 / stair 平台) -> 只剩兩端
      * 線性升降 (triangle、stair 過渡) -> 只剩折點
      * sine 等曲線 -> 段長由曲率與 tol 決定 (tol 越大段越長、航點越少)
    tol <= 0 -> 全部保留。iterative 實作避免遞迴深度問題。
    """
    n = len(s)
    if n <= 2 or tol <= 0:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        s0, s1 = s[i0], s[i1]
        chord = z[i0] + (z[i1] - z[i0]) * (s[i0 + 1:i1] - s0) / max(s1 - s0, 1e-12)
        dev = np.abs(z[i0 + 1:i1] - chord)
        k = int(np.argmax(dev))
        if dev[k] > tol:
            m = i0 + 1 + k
            keep[m] = True
            stack.append((i0, m))
            stack.append((m, i1))
    return np.where(keep)[0]


def straight_mission_polyline(
    z_at, lap_xy: np.ndarray, laps: int, spacing: float, z_tol: float, anchors=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """直線型 pattern (往返/矩形/鋸齒) 的航點折線 —— 「直線平飛只留端點」。

    以 tile_closed(lap_xy, laps) 的精確頂點 (轉角) 當航點錨點 (一律保留)；每段各自以
    ~spacing 等分取樣、高度由 z_at(累積弧長) 求值，再用 (弧長, 高度) Douglas-Peucker
    (高度容差 z_tol) 拿掉多餘的中間點：
      * 高度不變 (振幅 0 / cycles 0) -> 只剩轉角：往返 = A,B,A,B,...、矩形 = 每圈 4 角
      * 線性升降 (triangle / stair 過渡) -> 只剩折點；stair 平台 -> 只剩兩端
      * sine 等連續變化 -> 段長由曲率與 z_tol 決定
    段越長, ArduPilot S-curve 每段能達到的速度越高 (太短的段跑不到巡航速度)。
    anchors: 必留的水平弧長位置 (單圈內; 每圈都會加上), 例如越過箱子的平台起訖 / 斜坡轉折 ——
    它們是高度折線的真正轉折點, 只靠等距取樣會落在兩點之間、讓航點間的直線切到平台下方。
    回傳 (xs, ys, zs)，首尾 = 起點 (閉合)。
    """
    verts = _dedupe(tile_closed(np.asarray(lap_xy, dtype=float), max(1, int(laps))))
    lap_len = arclength(np.asarray(lap_xy, dtype=float))
    anc = np.asarray(anchors, dtype=float) if anchors is not None else np.zeros(0)
    if len(anc) and lap_len > 0:
        anc = np.concatenate([anc + k * lap_len for k in range(max(1, int(laps)))])
    if len(verts) < 2:
        z0 = float(np.atleast_1d(z_at(np.zeros(1)))[0])
        return verts[:1, 0].copy(), verts[:1, 1].copy(), np.array([z0])
    seg = np.hypot(np.diff(verts[:, 0]), np.diff(verts[:, 1]))
    s_v = np.concatenate([[0.0], np.cumsum(seg)])

    xs, ys, zs = [], [], []
    for i in range(len(verts) - 1):
        L = float(seg[i])
        n_seg = max(1, int(round(L / max(spacing, 1e-3))))
        frac = np.arange(n_seg + 1) / n_seg                  # 0 .. 1 (含段終點, 供 DP 判斷)
        if len(anc):
            inside = anc[(anc > s_v[i] + 1e-9) & (anc < s_v[i + 1] - 1e-9)]
            if len(inside):
                frac = np.unique(np.concatenate([frac, (inside - s_v[i]) / L]))
        ss = s_v[i] + frac * L
        pz = np.asarray(z_at(ss), dtype=float)
        keep = simplify_sz(ss, pz, z_tol)[:-1]               # 段終點由下一段起點提供
        px = verts[i, 0] + frac[keep] * (verts[i + 1, 0] - verts[i, 0])
        py = verts[i, 1] + frac[keep] * (verts[i + 1, 1] - verts[i, 1])
        xs.append(px); ys.append(py); zs.append(pz[keep])
    xs.append(verts[-1:, 0]); ys.append(verts[-1:, 1])
    zs.append(np.atleast_1d(np.asarray(z_at(np.array([s_v[-1]])), dtype=float)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def _lap_polyline_smooth(z_at, lap_xy: np.ndarray, spacing: float, anchors=None):
    """曲線 pattern 的一圈航點：沿 lap_xy (閉合密折線) 依 spacing 取樣, 高度由 z_at(弧長)。
    anchors (水平弧長, 單圈內) 會換算成 3D 弧長併入取樣點 (越過箱子的斜坡轉折必留)。"""
    lap = _dedupe(np.asarray(lap_xy, dtype=float))
    seg = np.hypot(np.diff(lap[:, 0]), np.diff(lap[:, 1]))
    s_l = np.concatenate([[0.0], np.cumsum(seg)])
    z_l = np.asarray(z_at(s_l), dtype=float)
    anc3 = None
    if anchors is not None and len(anchors):
        # 錨點處先插入精確的 (x, y, z) 頂點, 再把它們的 3D 弧長當錨點
        a = np.asarray(anchors, dtype=float)
        a = a[(a > 1e-9) & (a < s_l[-1] - 1e-9)]
        s_all = np.unique(np.concatenate([s_l, a]))
        x_all = np.interp(s_all, s_l, lap[:, 0]); y_all = np.interp(s_all, s_l, lap[:, 1])
        z_all = np.asarray(z_at(s_all), dtype=float)
        d3 = np.sqrt(np.diff(x_all) ** 2 + np.diff(y_all) ** 2 + np.diff(z_all) ** 2)
        s3 = np.concatenate([[0.0], np.cumsum(d3)])
        anc3 = np.interp(a, s_all, s3)
        return resample_by_spacing(x_all, y_all, z_all, spacing, anchors=anc3)
    return resample_by_spacing(lap[:, 0], lap[:, 1], z_l, spacing)


def _tile_xyz(x, y, z, laps: int):
    """把閉合的一圈 3D 折線 (首尾同點) 接成 laps 圈的實飛折線。"""
    blk = np.column_stack([x, y, z])
    full = tile_closed(blk, laps)
    return full[:, 0], full[:, 1], full[:, 2]


def approach_points(field, x0: float, y0: float, z0: float, takeoff_alt: float,
                    home: Tuple[float, float] = (0.0, 0.0)):
    """從起飛點 home 起飛後飛到第一個航點的進場段：有障礙物擋住時用可視圖最短路插入中間航點。

    回傳 (xs, ys, zs) 中間點 (不含起飛點與第一個航點；直線可達或無障礙物時為空)。高度沿進場路徑
    由起飛高度線性變到第一個航點高度。field = core.obstacles.ObstacleField (None = 無障礙物)。
    起飛點是實際停機位置 (waypoints.takeoff_point)，不是房間原點 —— 這一段是實飛的第一段。
    """
    empty = (np.zeros(0), np.zeros(0), np.zeros(0))
    if field is None or not getattr(field, "enabled", False):
        return empty
    field = getattr(field, "approach_field", field)     # 進場段一律水平繞開所有箱子 (含可越過的)
    if not getattr(field, "active", False):
        return empty
    path = field.free_path((float(home[0]), float(home[1])), (float(x0), float(y0)))
    if path is None or len(path) <= 2:
        return empty
    pts = np.asarray(path, dtype=float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1]) if s[-1] > 0 else 1.0
    zs = float(takeoff_alt) + (float(z0) - float(takeoff_alt)) * (s / total)
    return pts[1:-1, 0].copy(), pts[1:-1, 1].copy(), zs[1:-1].copy()


def signed_turns_xy(xs: np.ndarray, ys: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """折線各頂點的『水平』帶號轉向角 (deg, +=左轉/逆時針, 範圍 [-180, 180)) 與出發方向 (ENU yaw, rad)。

    進入方向取該點之前最近一段水平長度 > eps 的線段、出發方向取之後最近一段 (純垂直段跳過,
    與 speed_profile 的轉角判定同一精神)。定義不出來的點 (首點沒有進入段、末點沒有出發段) 為 nan。
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    turn = np.full(n, np.nan)
    head = np.full(n, np.nan)
    if n < 2:
        return turn, head
    dx, dy = np.diff(xs), np.diff(ys)
    ok = np.hypot(dx, dy) > eps
    ang = np.arctan2(dy, dx)
    prv = np.full(n, -1)
    nxt = np.full(n, -1)
    last = -1
    for i in range(n):
        prv[i] = last                      # 第 i 點之前最近的有效段
        if i < n - 1 and ok[i]:
            last = i
    last = -1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and ok[i]:
            last = i
        nxt[i] = last                      # 第 i 點之後 (含從 i 出發那段) 最近的有效段
    has_out = nxt >= 0
    head[has_out] = ang[nxt[has_out]]
    both = has_out & (prv >= 0)
    d = np.degrees(ang[nxt[both]] - ang[prv[both]])
    turn[both] = (d + 180.0) % 360.0 - 180.0
    return turn, head


def _yaw_rotation_deg(turn_deg: float, direction: int) -> float:
    """以 CONDITION_YAW 方向 direction (0 最短 / 1 CW / -1 CCW) 完成帶號轉向 turn_deg 實際要轉幾度。
    turn_deg = nan (進入方向不明, 例如起飛後直接爬到正上方的航點) 視為最壞: 最短 180°、指定方向 360°。"""
    if not np.isfinite(turn_deg):
        return 180.0 if direction == 0 else 360.0
    a = abs(float(turn_deg))
    if direction == 0:
        return a
    natural = -1 if turn_deg > 0 else 1        # 左轉 (ENU 逆時針) = 羅盤航向變小 = CCW (-1)
    return a if natural == direction else 360.0 - a


def assign_turn_stops(mp: MissionPath, cfg: Dict, is_smooth: bool) -> MissionPath:
    """依 waypoints.turn_in_place 決定哪些導航任務項後面要「停下原地轉頭」, 填入 mp.turn_*。

    * 轉向角以實飛折線 (起飛點 -> 進場 -> 完整圈數) 的水平幾何計算；DO_JUMP block 裡的航點每圈
      經過一次, 進入方向可能不同 (第一圈從起飛點/進場段來), 取各次的最大轉角判定與估 NAV_DELAY。
    * 目標航向 = 從該點出發那一段的水平方向 (絕對航向, 每次經過都相同 -> 誤差不累積)。
    * 方向一般用最短 (0)；有原路折返 (≈180°, 最短方向不明確) 時用 u_turn_dir, 但若會讓其他次經過
      繞遠路 (> 180°) 就退回最短。
    * 最後一個導航點 (接 LAND/RTL) 不轉；曲線 pattern (is_smooth) 的航點除非 curves=true 否則不轉
      (進場繞障航點一律照轉角判定)。
    * after_takeoff: 起飛到高度後先原地轉向第一段 (起飛時機頭朝向未知 -> 最短方向、以 180° 估等待)。
    NAV_DELAY 秒數見 TurnInPlace.delay_s (一定 > 0)。
    """
    mp.turn_approach, mp.turn_nav, mp.turn_takeoff = {}, {}, None
    tip = TurnInPlace.from_config(cfg)
    if not tip.enabled:
        return mp
    xs, ys, _ = mp.flown_with_approach(0.0)
    turn, head = signed_turns_xy(xs, ys)
    last = len(xs) - 1
    if tip.after_takeoff and last >= 1 and np.isfinite(head[0]):
        mp.turn_takeoff = TurnStop(heading_deg=round(enu_to_compass_deg(float(head[0])), 2),
                                   rate_deg_s=tip.rate_deg_s, direction=0, turn_deg=180.0,
                                   delay_s=tip.delay_s(180.0))
    skip_nav = bool(is_smooth) and not tip.curves
    for kind, i, occ in mp.item_occurrences():
        if kind == "nav" and skip_nav:
            continue
        occ = [f for f in occ if f < last]
        if not occ or not np.isfinite(head[occ[0]]):
            continue
        ts = [float(turn[f]) for f in occ]
        finite = [abs(t) for t in ts if np.isfinite(t)]
        if not finite or max(finite) < tip.min_turn_deg:
            continue
        direction = 0
        if any(np.isfinite(t) and abs(t) >= 180.0 - _U_TURN_BAND_DEG for t in ts):
            for d in (tip.u_turn_dir, -tip.u_turn_dir):
                if max(_yaw_rotation_deg(t, d) for t in ts) <= 180.0 + _U_TURN_BAND_DEG:
                    direction = d
                    break
        rot = max(_yaw_rotation_deg(t, direction) for t in ts)
        stop = TurnStop(heading_deg=round(enu_to_compass_deg(float(head[occ[0]])), 2),
                        rate_deg_s=tip.rate_deg_s, direction=direction,
                        turn_deg=round(rot, 2), delay_s=tip.delay_s(rot))
        (mp.turn_approach if kind == "approach" else mp.turn_nav)[i] = stop
    return mp


def mission_polyline(
    traj: "Trajectory", cfg: Dict, is_smooth: bool,
    lap_xy: np.ndarray | None = None, laps: int = 1, mode: str = "precision",
    repeatable: bool = True, field=None,
) -> MissionPath:
    """AUTO 任務 (.waypoints) 的航點折線。匯出與工時預估共用。

    * waypoints.repeat = 'do_jump' (預設): 只產生「一圈」的航點 (block) + 收尾點；匯出時在 block
      後面加 DO_JUMP(回 block 第一點, 重複 圈數-1 次)。每圈高度必須相同 (planner 已把起伏
      次數調成每圈整數次)。flown_* 為 block 重複 laps 圈的完整折線 (估時用)。
      'unroll' = 全程展開 (舊行為)。
    * 曲線 (圓/8字): 沿弧長依 waypoints.point_spacing 取樣 (use_spline 時間距再乘 spline_spacing_mult)。
    * 直線型 (往返/矩形/鋸齒) 且 waypoints.sparse_straight (預設開): 以精確轉角為航點，段內
      用高度容差 z_tol 的 (弧長,高度) 精簡 —— 平飛只留端點、線性升降只留折點。
    * mode='compact': 超過 fc_budget - 固定項 時等弧長重取樣壓到預算內。
    * repeatable=False (pattern 宣告不重複, 例如隨機手飛): 一律 unroll、不寫 DO_JUMP。
    * field (core.obstacles.ObstacleField): 有障礙物擋在起飛點 (waypoints.takeoff_point) 與第一個
      航點之間時, 插入進場航點 (approach_*) 繞過去；precision/compact 版都有, compact 的預算會扣掉它們。
    高度一律由軌跡的 z(s) 依弧長內插 (高度是位置的純函數)。
    """
    w = cfg.get("waypoints", {})
    spacing = float(w.get("point_spacing", 0.25))
    use_spline = bool(w.get("use_spline", True)) and bool(is_smooth)
    if use_spline:
        spacing *= float(w.get("spline_spacing_mult", 3.0))
    z_tol = float(w.get("z_tol", DEFAULT_Z_TOL))
    home = takeoff_point(cfg)
    ax, ay, az = approach_points(field, float(traj.x[0]), float(traj.y[0]), float(traj.z[0]),
                                 float(w.get("takeoff_alt", 1.0)), home=home)
    laps = max(1, int(laps))
    repeat = str(w.get("repeat", "do_jump")).lower()
    if repeat not in ("do_jump", "unroll") or lap_xy is None or not repeatable:
        repeat = "unroll"
    max_points = None
    if mode == "compact":
        # 預算是「總任務項」: 固定項 (HOME/TAKEOFF/SPEED/結束) + 進場航點 + DO_JUMP (有的話) + 導航航點。
        # DO_JUMP 也佔一項, 沒扣掉的話匯出會剛好超過預算 1 項。
        budget = int(w.get("fc_budget", DEFAULT_FC_BUDGET))
        reserved = MISSION_OVERHEAD_ITEMS + len(ax) + (1 if (repeat == "do_jump" and laps > 1) else 0)
        max_points = max(2, budget - reserved)

    # 高度 = f(弧長)：從軌跡的 (s, z) 內插 (軌跡的 s 就是頂點折線的精確弧長)
    if traj.s is not None:
        s_tr = np.asarray(traj.s, dtype=float)
    else:
        s_tr = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(traj.x), np.diff(traj.y)))])

    def z_at(s):
        return np.interp(np.asarray(s, dtype=float), s_tr, traj.z)

    sparse = bool(w.get("sparse_straight", True)) and not is_smooth and lap_xy is not None
    anchors = traj.meta.get("z_anchors_s")           # 越過箱子的平台 / 斜坡轉折 (單圈弧長), 沒有則 None
    lap_len = float(traj.meta.get("single_lap_length_m", 0.0) or 0.0)

    if repeat == "do_jump":
        if sparse:
            blk = straight_mission_polyline(z_at, lap_xy, 1, spacing, z_tol, anchors=anchors)
        else:
            blk = _lap_polyline_smooth(z_at, lap_xy, spacing, anchors=anchors)

        def build(cap):
            bx, by, bz = thin_polyline(*blk, cap)
            fx, fy, fz = _tile_xyz(bx, by, bz, laps)
            return MissionPath(bx, by, bz, fx, fy, fz, use_spline, "do_jump",
                               block_len=max(0, len(bx) - 1), jump_repeat=laps - 1,
                               approach_x=ax, approach_y=ay, approach_z=az, home=home)
    elif sparse:
        full = straight_mission_polyline(z_at, lap_xy, laps, spacing, z_tol, anchors=anchors)

        def build(cap):
            xs, ys, zs = thin_polyline(*full, cap)
            return MissionPath(xs, ys, zs, xs, ys, zs, use_spline, "unroll", block_len=0, jump_repeat=0,
                               approach_x=ax, approach_y=ay, approach_z=az, home=home)
    else:
        anc3 = None
        if anchors is not None and len(anchors) and lap_len > 0:
            # 每圈的錨點 (水平弧長) -> 軌跡取樣點的 3D 弧長
            s3 = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(traj.x) ** 2 + np.diff(traj.y) ** 2
                                                          + np.diff(traj.z) ** 2))])
            all_a = np.concatenate([np.asarray(anchors, dtype=float) + k * lap_len for k in range(laps)])
            anc3 = np.interp(all_a, s_tr, s3)

        def build(cap):
            xs, ys, zs = resample_by_spacing(traj.x, traj.y, traj.z, spacing, max_points=cap, anchors=anc3)
            return MissionPath(xs, ys, zs, xs, ys, zs, use_spline, "unroll", block_len=0, jump_repeat=0,
                               approach_x=ax, approach_y=ay, approach_z=az, home=home)

    mp = assign_turn_stops(build(max_points), cfg, is_smooth)
    if max_points is None or not mp.n_turns:
        return mp
    # compact: 每個轉頭點多佔 CONDITION_YAW + NAV_DELAY 兩項。超過預算就再壓少航點 (重取樣後轉頭點數
    # 會跟著變, 迭代到收斂)；航點已壓到極少仍超過時丟掉轉角最小的轉頭點 —— 「寫得進飛控」優先。
    budget = int(w.get("fc_budget", DEFAULT_FC_BUDGET))
    cap = max_points
    for _ in range(12):
        over = MISSION_OVERHEAD_ITEMS + mp.n_items - budget
        if over <= 0 or cap <= 2:
            break
        cap = max(2, min(cap, len(mp.x)) - over)
        mp = assign_turn_stops(build(cap), cfg, is_smooth)
    over = MISSION_OVERHEAD_ITEMS + mp.n_items - budget
    if over > 0:
        ranked = sorted(([(-1.0, 2, 0)] if mp.turn_takeoff is not None else [])    # 起飛後轉向最先丟
                        + [(s.turn_deg, 0, i) for i, s in mp.turn_approach.items()]
                        + [(s.turn_deg, 1, i) for i, s in mp.turn_nav.items()])
        for _, kind, i in ranked[:math.ceil(over / TURN_ITEMS_PER_STOP)]:
            if kind == 2:
                mp.turn_takeoff = None
            else:
                (mp.turn_approach if kind == 0 else mp.turn_nav).pop(i)
    return mp


def _resample_constant_speed(
    poly: np.ndarray, speed: float, dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿折線弧長以定速重新取樣，回傳 (t, x, y, s)。(舊版行為, speed_profile=constant)"""
    poly = _dedupe(poly)
    seg = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    duration = total / speed
    n = max(2, int(round(duration / dt)) + 1)
    t = np.linspace(0.0, duration, n)
    dist = np.clip(speed * t, 0.0, total)
    x = np.interp(dist, s, poly[:, 0])
    y = np.interp(dist, s, poly[:, 1])
    return t, x, y, dist


def _rate_limit_yaw(t: np.ndarray, yaw: np.ndarray, max_rate: float) -> np.ndarray:
    """限制 yaw 角速度 <= max_rate (rad/s)。

    travel 模式在硬轉角處航向瞬間跳變，實機 yaw 追不上；限速後串流的 yaw 設點
    才是可跟隨的 (多旋翼為全向, yaw 落後不影響位置追蹤)。
    """
    if max_rate <= 0 or len(yaw) < 2:
        return yaw
    out = yaw.copy()
    for i in range(1, len(out)):
        lim = max_rate * float(t[i] - t[i - 1])
        step = out[i] - out[i - 1]
        if step > lim:
            out[i] = out[i - 1] + lim
        elif step < -lim:
            out[i] = out[i - 1] - lim
    return out


def _compute_yaw(
    t: np.ndarray, vx: np.ndarray, vy: np.ndarray, cfg: Dict
) -> np.ndarray:
    """依設定計算機頭朝向 (弧度, ENU)。"""
    ycfg = cfg.get("yaw", {})
    mode = ycfg.get("mode", "travel")
    max_rate = np.radians(float(ycfg.get("max_rate", 90.0)))
    if mode == "fixed":
        return np.full_like(t, np.radians(ycfg.get("fixed_deg", 0.0)))
    if mode == "spin":
        rate = np.radians(ycfg.get("spin_rate", 6.0))
        return np.unwrap((rate * t) % (2 * np.pi))
    # travel: 朝行進方向; 速度過小時沿用前一個有效角
    yaw = np.arctan2(vy, vx)
    speed = np.hypot(vx, vy)
    thresh = 0.05 * max(np.max(speed), 1e-6)
    last = yaw[0]
    for i in range(len(yaw)):
        if speed[i] < thresh:
            yaw[i] = last
        else:
            last = yaw[i]
    return _rate_limit_yaw(t, np.unwrap(yaw), max_rate)


def build_trajectory(
    name: str,
    lap_xy: np.ndarray,
    laps: int,
    box: SafeBox,
    cfg: Dict,
    altitude_fn,
) -> Trajectory:
    """把單圈折線 + 圈數組裝成完整 3D 軌跡。

    altitude_fn(t, box, cfg, s) -> z 陣列 (見 core/altitude.py)；s = 各取樣點沿頂點折線的
    累積水平弧長 —— 高度是位置 (弧長) 的函數，與 AUTO 航點內插行為一致。
    """
    speed = float(cfg["flight"]["cruise_speed"])
    dt = float(cfg["flight"]["sample_dt"])
    mode = str(cfg["flight"].get("speed_profile", "dynamic")).lower()

    full = tile_closed(np.asarray(lap_xy, dtype=float), laps)
    horiz_len = arclength(full)
    prof_meta: Dict = {}
    if mode == "constant":
        t, x, y, s = _resample_constant_speed(full, speed, dt)
    else:
        prof = compute_profile(full, ProfileParams.from_config(cfg))
        t, x, y = resample_profile(prof, dt)
        s = np.interp(t, prof.t, prof.s)             # 頂點折線的精確弧長 (非取樣點連線長)
        prof_meta = {
            "hard_corners_total": prof.hard_corners,
            "min_corner_speed": prof.min_corner_speed,
        }

    z = altitude_fn(t, box, cfg, s)

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    vz = np.gradient(z, t)
    yaw = _compute_yaw(t, vx, vy, cfg)

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return Trajectory(
        name=name,
        t=t, x=x, y=y, z=z, yaw=yaw, vx=vx, vy=vy, vz=vz, s=s,
        meta={
            "laps": laps,
            "cruise_speed": speed,
            "speed_profile": mode,
            "horizontal_length_m": horiz_len,
            "single_lap_length_m": arclength(lap_xy),
            "naive_duration_s": (horiz_len / speed) if speed > 0 else 0.0,
            "avg_speed_h": (horiz_len / duration) if duration > 0 else 0.0,
            **prof_meta,
        },
    )
