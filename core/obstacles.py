"""
障礙物 (箱子) 與水平避障。

* 來源：config `obstacles.items` (已知座標：手動輸入 / 儲存設定) 或 VRPN (core/vrpn_client 把
  Motive 串流的 rigid body 位置轉成同樣的 items)。
* 幾何模型：每個障礙物 = 水平凸多邊形底面 (footprint, 房間 ENU XY) × 高度區間 [z_bottom, z_top]。
  規劃時一律視為「無限高的垂直柱」—— 路徑只「繞開」(水平)、不「越過」(垂直)，高度只用於顯示與
  說明。
* 避障流程 (ObstacleField.build / avoid_lap)：
    1. 底面外擴 `clearance` (機身半徑 + 定位誤差 + 餘裕) 成「禁區」(圓角以每 90° corner_segments 段
       折線近似, 外接 -> 任一點離障礙物外緣 >= clearance)。
    2. 禁區互相重疊 / 相碰者合併成凸包 (兩箱間隙 < 2×clearance 本來就不能穿)。
    3. 禁區裁到安全盒 (牆外的部分到不了)。
    4. 對 pattern 單圈折線：頂點落在禁區內、或落在「禁區貼著牆封出來的口袋」裡 (從起飛點 HOME
       到不了) -> 推到禁區邊界上「前後兩段路最省」且可達的點 (曲線 pattern 則直接拿掉這些密集點)；
       逐段檢查, 穿過禁區的線段以「可視圖最短路」改道 —— 沿禁區的邊繞過去, 只多出必要的轉角
       (航點越少越好)。
  改道後的折線交回既有流程 (圈數 / 速度剖面 / 高度 = f(弧長) / 航點匯出) 一律不變。
* 低矮箱子可「越過」(obstacles.low_mode=over)：箱頂 <= over_max_top (預設 0.6 m) 且箱頂 + vertical_clearance
  不超過天花板邊界的箱子不做水平改道，改由 AltitudeFloor 在路徑進入其禁區時把高度拉到
  箱頂 + vertical_clearance (平台)，前後以固定垂直速度的斜坡銜接；高度仍是弧長的函數 (DO_JUMP 每圈相同)。
  起飛進場段一律水平繞開所有箱子 (approach_field)。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import SafeBox, takeoff_point

_EPS = 1e-9        # 「嚴格在內部」的容差 (m)
_PUSH = 1e-6       # 把點推出禁區邊界的距離 (m)
DEFAULT_CLEARANCE = 0.5
DEFAULT_CORNER_SEGMENTS = 2
DEFAULT_DRONE_SIZE = (0.50, 0.50, 0.15)  # 機身尺寸 (長 x, 寬 y, 高 z, m)
TRACKING_MARGIN = 0.10             # 建議安全距離裡留給定位/追蹤誤差的餘裕 (m)
VERTICAL_MARGIN = 0.15             # 建議垂直安全距離裡留給下洗氣流/追蹤的餘裕 (m)
DEFAULT_MARKER_ABOVE = 0.03        # 光球本身高出頂板的部分 (m)
CLEARANCE_SLACK = 0.01             # 安全距離比建議值少於這個量就不嘮叨 (m)
CLIMB_OVERSHOOT = 0.05             # 爬到目標高度時的超調餘裕 (m)
DEFAULT_OVER_MAX_TOP = 0.6         # 箱頂不超過此高度 (m) 的箱子可越過
DEFAULT_VERTICAL_CLEARANCE = 0.5   # 越過時離箱頂的最小垂直距離 (m)
OVER_RAMP_FRAC = 0.8               # over_ramp_speed=auto 時 = 此比例 × min(垂直上限, 3D 合成速度餘裕)


class ObstacleError(ValueError):
    """障礙物設定錯誤 (座標格式等)。"""


def drone_size(cfg: Dict) -> Tuple[float, float, float]:
    """機身尺寸 (長 x, 寬 y, 高 z, m)，取自 obstacles.drone_size。只給兩個值時高度用預設。"""
    v = (cfg.get("obstacles", {}) or {}).get("drone_size") or DEFAULT_DRONE_SIZE
    try:
        vals = [float(t) for t in list(v)[:3]]
    except (TypeError, ValueError) as e:
        raise ObstacleError(f"obstacles.drone_size 需為 [長, 寬] 或 [長, 寬, 高] (m)，得到 {v!r}") from e
    if len(vals) == 2:
        vals.append(float(DEFAULT_DRONE_SIZE[2]))
    if len(vals) != 3:
        raise ObstacleError(f"obstacles.drone_size 需為 [長, 寬] 或 [長, 寬, 高] (m)，得到 {v!r}")
    if not all(math.isfinite(t) and t > 0 for t in vals):
        raise ObstacleError(f"obstacles.drone_size 需為正的有限數值，得到 {v!r}")
    return vals[0], vals[1], vals[2]


def marker_height(cfg: Dict) -> Optional[float]:
    """停在地上時, 光球 (= 動捕與飛控回報的參考點) 離地高度 (m)。未設定時回 None。"""
    v = (cfg.get("obstacles", {}) or {}).get("marker_height", None)
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto", "center", "centre")):
        return None
    try:
        h = float(v)
    except (TypeError, ValueError) as e:
        raise ObstacleError(f"obstacles.marker_height 需為數值 (m)，得到 {v!r}") from e
    if not math.isfinite(h) or h <= 0:
        raise ObstacleError(f"obstacles.marker_height 需為正的有限數值，得到 {v!r}")
    return h


def pivot_below(cfg: Dict) -> float:
    """規劃高度『下方』還有多少機體 (m)。

    規劃與飛控回報的都是「參考點」的高度, 而參考點就是光球群的中心。光球全貼在頂板時參考點在機身
    上緣, 整個機體掛在它下面 —— 離地、越過箱子的餘裕都要從這個值扣, 不是從半高扣。
    obstacles.marker_height 未設定時退回「參考點在機身中心」的假設 (上下各半高)。
    """
    h = marker_height(cfg)
    return h if h is not None else drone_size(cfg)[2] / 2.0


def pivot_above(cfg: Dict) -> float:
    """規劃高度『上方』還有多少機體 (m)。光球貼頂板時只剩光球本身 (obstacles.marker_above)。"""
    if marker_height(cfg) is None:
        return drone_size(cfg)[2] / 2.0
    v = (cfg.get("obstacles", {}) or {}).get("marker_above", DEFAULT_MARKER_ABOVE)
    return float(DEFAULT_MARKER_ABOVE if v is None else v)


def recommended_vertical_clearance(cfg: Dict) -> float:
    """建議的最小垂直安全距離 (m) = 參考點下方的機體 + 下洗氣流/追蹤餘裕。"""
    return pivot_below(cfg) + VERTICAL_MARGIN


def drone_radius(cfg: Dict) -> float:
    """機身外接半徑 (m) = 半對角。

    規劃出來的是「機身中心」的路徑，而機頭方向 (yaw) 隨軌跡改變，所以會碰到東西的是離中心最遠的
    那個角，不是半寬。50×50 cm 的機身半寬 0.25 m、半對角 0.354 m —— 差的這 10 cm 就是撞到與否。
    """
    sx, sy, _ = drone_size(cfg)
    return math.hypot(sx / 2.0, sy / 2.0)


def recommended_clearance(cfg: Dict) -> float:
    """建議的最小水平安全距離 (m) = 機身半徑 + 航點切角 + 追蹤誤差餘裕。"""
    accept = float((cfg.get("waypoints", {}) or {}).get("accept_radius", 0.05))
    return drone_radius(cfg) + accept + TRACKING_MARGIN


# ----------------------------------------------------------------------
# 基本 2D 幾何 (凸多邊形)
# ----------------------------------------------------------------------
def _dedupe(poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if len(poly) <= 1:
        return poly
    keep = [0]
    for i in range(1, len(poly)):
        if np.hypot(*(poly[i] - poly[keep[-1]])) > tol:
            keep.append(i)
    return poly[keep]


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain -> 逆時針凸包 (去掉共線點)。點數 < 3 時原樣回傳 (去重)。"""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return hull if len(hull) >= 3 else pts[:2] if len(pts) >= 2 else pts


def _halfplanes(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """逆時針凸多邊形 -> 各邊的外法向 n (E,2) 與偏移 c (E,)：內部 <=> n·p - c <= 0。"""
    v0 = poly
    v1 = np.roll(poly, -1, axis=0)
    e = v1 - v0
    n = np.column_stack([e[:, 1], -e[:, 0]])
    ln = np.maximum(np.linalg.norm(n, axis=1), 1e-15)
    n = n / ln[:, None]
    c = np.einsum("ij,ij->i", n, v0)
    return n, c


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def points_inside(P: np.ndarray, poly: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """(M,2) 各點是否『嚴格』在凸多邊形內部 (邊界上不算)。"""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return np.zeros(len(P), dtype=bool)
    n, c = _halfplanes(poly)
    d = P @ n.T - c
    return np.all(d < -eps, axis=1)


def _segment_distances(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(M,2) 點到 (E,2)-(E,2) 各線段的距離 -> (M,E)。"""
    AB = B - A                                   # (E,2)
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-18)
    AP = P[:, None, :] - A[None, :, :]           # (M,E,2)
    t = np.clip(np.einsum("mej,ej->me", AP, AB) / L2[None, :], 0.0, 1.0)
    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
    return np.linalg.norm(P[:, None, :] - proj, axis=2)


def signed_distance(P: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(M,2) 各點到多邊形的帶號距離 (外正、內負；退化為點/線段時為一般距離)。"""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) == 0:
        return np.full(len(P), np.inf)
    if len(poly) == 1:
        return np.linalg.norm(P - poly[0], axis=1)
    if len(poly) == 2:
        return _segment_distances(P, poly[:1], poly[1:])[:, 0]
    d_edge = _segment_distances(P, poly, np.roll(poly, -1, axis=0)).min(axis=1)
    inside = points_inside(P, poly, eps=0.0)
    return np.where(inside, -d_edge, d_edge)


def segment_intervals(P: np.ndarray, Q: np.ndarray, poly: np.ndarray, eps: float = _EPS):
    """線段 P[i]->Q[i] 在凸多邊形『嚴格內部』的參數區間 -> (t0, t1, hit)，hit = 區間非空 (S,)。

    Cyrus–Beck：對每個半平面求 d(t) = dp + t·(dq−dp) < −eps 的 t 區間，全部交集即在內部的區間。
    沿著邊走 (d ≡ 0) 或只碰到頂點區間為空 —— 可視圖沿禁區邊繞行的路徑才會被允許。
    """
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    Q = np.asarray(Q, dtype=float).reshape(-1, 2)
    if len(poly) < 3 or len(P) == 0:
        z = np.zeros(len(P))
        return z, z.copy(), np.zeros(len(P), dtype=bool)
    n, c = _halfplanes(poly)
    dp = P @ n.T - c
    dq = Q @ n.T - c
    dd = dq - dp
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = (-eps - dp) / dd
    flat = dd == 0.0
    flat_ok = flat & (dp < -eps)
    lo = np.where(dd < 0, tc, -np.inf)
    hi = np.where(dd > 0, tc, np.inf)
    lo = np.where(flat, np.where(flat_ok, -np.inf, np.inf), lo)
    hi = np.where(flat, np.where(flat_ok, np.inf, -np.inf), hi)
    t0 = np.maximum(0.0, lo.max(axis=1))
    t1 = np.minimum(1.0, hi.min(axis=1))
    return t0, t1, (t1 - t0) > 1e-12


def segments_enter(P: np.ndarray, Q: np.ndarray, poly: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """線段 P[i]->Q[i] 是否有任何一點『嚴格』在凸多邊形內部 -> bool (S,)。"""
    return segment_intervals(P, Q, poly, eps)[2]


def polygons_intersect(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """兩凸多邊形是否相交 / 相碰 (分離軸定理)。"""
    for poly in (a, b):
        if len(poly) < 2:
            continue
        n, _ = _halfplanes(poly) if len(poly) >= 3 else (
            np.array([[poly[1, 1] - poly[0, 1], poly[0, 0] - poly[1, 0]]]), None)
        for axis in n:
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() - tol or pb.max() < pa.min() - tol:
                return False
    return True


def inflate(poly: np.ndarray, r: float, k: int = 8, phase: float = 0.0) -> np.ndarray:
    """把凸多邊形 (或點 / 線段) 外擴 r：與正 k 邊形做 Minkowski 和 (= 各頂點加一圈 k 邊形點取凸包)。

    k 邊形取『外接』(頂點半徑 r/cos(π/k)) 並把邊對齊 phase (箱子的偏航)：對齊的直邊外擴恰為 r、
    圓角外擴 r ~ 1.08r (k=8)，保證外擴多邊形完全涵蓋真正的 r 偏移區。
    """
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if r <= 0:
        return convex_hull(poly)
    K = max(4, int(k))
    rr = r / math.cos(math.pi / K)
    ang = phase + (np.arange(K) + 0.5) * (2.0 * np.pi / K)
    circ = rr * np.column_stack([np.cos(ang), np.sin(ang)])
    pts = (poly[:, None, :] + circ[None, :, :]).reshape(-1, 2)
    return convex_hull(pts)


def clip_to_rect(poly: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    """Sutherland–Hodgman：凸多邊形裁到矩形，回傳逆時針凸多邊形 (可能少於 3 點 = 空)。"""
    out = [tuple(p) for p in np.asarray(poly, dtype=float).reshape(-1, 2)]
    if len(out) < 3:
        return np.zeros((0, 2))
    planes = (
        (lambda p: p[0] >= x_min, lambda a, b: _isect_x(a, b, x_min)),
        (lambda p: p[0] <= x_max, lambda a, b: _isect_x(a, b, x_max)),
        (lambda p: p[1] >= y_min, lambda a, b: _isect_y(a, b, y_min)),
        (lambda p: p[1] <= y_max, lambda a, b: _isect_y(a, b, y_max)),
    )
    for inside, isect in planes:
        if not out:
            break
        inp, out = out, []
        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(isect(s, e))
                out.append(e)
            elif inside(s):
                out.append(isect(s, e))
            s = e
    if len(out) < 3:
        return np.zeros((0, 2))
    return convex_hull(np.array(out))


def _isect_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
    return (x, a[1] + t * (b[1] - a[1]))


def _isect_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
    return (a[0] + t * (b[0] - a[0]), y)


def rect_footprint(cx: float, cy: float, sx: float, sy: float, yaw: float = 0.0) -> np.ndarray:
    """中心 + 尺寸 + 偏航 (rad) -> 逆時針矩形 (尺寸 <= 0 退化為線段 / 點)。"""
    hx, hy = max(sx, 0.0) / 2.0, max(sy, 0.0) / 2.0
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return convex_hull(local @ R.T + np.array([cx, cy]))


def _polyline_length(p: np.ndarray) -> float:
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


# ----------------------------------------------------------------------
# 障礙物
# ----------------------------------------------------------------------
@dataclass
class Obstacle:
    """一個障礙物：水平凸底面 (房間 ENU, m) × 高度區間；kind = box (中心+尺寸+偏航) | points (光球座標凸包)。"""

    name: str
    footprint: np.ndarray            # (N,2) 逆時針凸多邊形 (未外擴; 退化時 1~2 點)
    z_bottom: float
    z_top: float
    kind: str = "box"
    source: str = "manual"           # manual | vrpn
    yaw: float = 0.0                 # rad (box 用; 外擴多邊形的圓角對齊)
    markers: Optional[np.ndarray] = None   # (M,3) 光球座標 (points 種類, 顯示用)
    raw: Dict = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        c = self.footprint.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def height(self) -> float:
        return self.z_top - self.z_bottom

    def summary(self) -> str:
        cx, cy = self.center
        if self.kind == "points":
            n = 0 if self.markers is None else len(self.markers)
            return f"{self.name} [光球×{n}] 中心 ({cx:.2f}, {cy:.2f}) 高 {self.z_top:.2f} m"
        sz = self.raw.get("size", [0, 0, self.height])
        return (f"{self.name} [箱] ({cx:.2f}, {cy:.2f}) "
                f"{float(sz[0]):.2f}×{float(sz[1]):.2f}×{self.height:.2f} m"
                + (f" 偏航 {math.degrees(self.yaw):.0f}°" if abs(self.yaw) > 1e-9 else ""))


def obstacle_from_item(item: Dict, index: int = 0) -> Obstacle:
    """config obstacles.items 的一項 -> Obstacle。

    box:    {name, kind: box, x, y, z_bottom (預設 0), size: [sx, sy, sz], yaw_deg}
    points: {name, kind: points, points: [[x, y, z], ...], z_bottom (預設 0)}  光球座標 -> XY 凸包底面,
            頂 = 最高的光球
    """
    if not isinstance(item, dict):
        raise ObstacleError(f"obstacles.items[{index}] 必須是物件 (dict)")
    kind = str(item.get("kind", "box")).strip().lower()
    name = str(item.get("name") or f"obs{index + 1}")
    source = str(item.get("source", "manual"))
    z_bottom = float(item.get("z_bottom", 0.0) or 0.0)
    if kind == "points":
        pts = np.asarray(item.get("points") or [], dtype=float)
        if pts.size == 0:
            raise ObstacleError(f"障礙物 {name}: points 不可為空 (每個光球 [x, y, z])")
        pts = pts.reshape(-1, 3) if pts.size % 3 == 0 else None
        if pts is None:
            raise ObstacleError(f"障礙物 {name}: points 每個光球需為 [x, y, z]")
        fp = convex_hull(pts[:, :2])
        z_top = max(float(pts[:, 2].max()), z_bottom + 0.01)
        return Obstacle(name, fp, z_bottom, z_top, kind="points", source=source,
                        markers=pts, raw=dict(item))
    size = item.get("size", [0.5, 0.5, 0.5])
    try:
        sx, sy, sz = (float(v) for v in size)
    except Exception as e:  # noqa: BLE001
        raise ObstacleError(f"障礙物 {name}: size 需為 [長, 寬, 高]") from e
    cx = float(item.get("x", 0.0))
    cy = float(item.get("y", 0.0))
    yaw = math.radians(float(item.get("yaw_deg", 0.0) or 0.0))
    fp = rect_footprint(cx, cy, sx, sy, yaw)
    return Obstacle(name, fp, z_bottom, z_bottom + max(sz, 0.01), kind="box", source=source,
                    yaw=yaw, raw=dict(item))


def obstacles_from_config(cfg: Dict) -> List[Obstacle]:
    o = cfg.get("obstacles", {}) or {}
    items = o.get("items") or []
    return [obstacle_from_item(it, i) for i, it in enumerate(items)]


# ----------------------------------------------------------------------
# 禁區場 + 可視圖避障
# ----------------------------------------------------------------------
@dataclass
class AvoidInfo:
    """avoid_lap 的統計。"""

    moved: int = 0            # 被推到禁區邊界的頂點數 (直線型 pattern)
    removed: int = 0          # 被拿掉的禁區內密集點數 (曲線 pattern)
    unreachable: int = 0      # 其中因「從 HOME 到不了」(禁區貼牆封出的口袋) 而被處理的點數
    detours: int = 0          # 改道的線段數
    failed: int = 0           # 無法在安全盒內繞開的線段數 (保留原線段, 由安全檢查擋下)
    length_before: float = 0.0
    length_after: float = 0.0

    @property
    def added_length(self) -> float:
        return self.length_after - self.length_before

    @property
    def touched(self) -> bool:
        return bool(self.moved or self.removed or self.detours or self.failed)

    def as_dict(self) -> Dict:
        return {"moved": self.moved, "removed": self.removed, "detours": self.detours,
                "unreachable": self.unreachable,
                "failed": self.failed, "length_before_m": self.length_before,
                "length_after_m": self.length_after, "added_length_m": self.added_length}


@dataclass
class OverZone:
    """可越過的箱子：進入其外擴禁區時高度必須 >= z_req。"""

    index: int                # obstacles 的索引
    poly: np.ndarray          # 外擴 clearance 的底面 (未裁盒)
    z_req: float              # 箱頂 + vertical_clearance


class ObstacleField:
    """障礙物 + 外擴禁區 + 可視圖 (規劃用)。build() 一次，之後 free_path / avoid_lap 反覆查詢。

    over_indices 指定「越過而不繞」的箱子：它們不進水平禁區 (zones)，改成 over 清單 (高度下限用)；
    approach_field 是把所有箱子都當水平禁區的版本 (起飛進場 / GUIDED 接近起點用, 進場高度低不越箱)。
    """

    def __init__(self, obstacles: Sequence[Obstacle], box: SafeBox, clearance: float,
                 corner_segments: int = DEFAULT_CORNER_SEGMENTS, enabled: bool = True,
                 over_indices: Sequence[int] = (), vertical_clearance: float = DEFAULT_VERTICAL_CLEARANCE,
                 home: Optional[Sequence[float]] = None):
        self.obstacles: List[Obstacle] = list(obstacles)
        self.box = box
        # 可達性錨點 = 起飛點 (飛機實際停放處)。「到不了」是相對於飛機出發的地方而言,
        # 所以障礙物把房間切成兩半時, 留下的是飛機那一半。
        self.home: np.ndarray = (np.zeros(2) if home is None
                                 else np.asarray(home, dtype=float).reshape(2).copy())
        self.clearance = float(max(clearance, 0.0))
        self.corner_segments = max(1, int(corner_segments))
        self.enabled = bool(enabled)
        self.vertical_clearance = float(max(vertical_clearance, 0.0))
        self.over_indices: List[int] = sorted({int(i) for i in over_indices if 0 <= int(i) < len(self.obstacles)})
        self.over: List[OverZone] = []
        self._approach_field: Optional["ObstacleField"] = None
        self.zones: List[np.ndarray] = []          # 禁區 (外擴 + 合併, 未裁盒; 逆時針凸多邊形) —— 穿越 / 在內判定用
        self.zones_clipped: List[np.ndarray] = []  # 同上裁到安全盒 —— 可視圖節點 / 投影目標 / 顯示用
        self.zone_members: List[List[int]] = []    # 每個禁區由哪些障礙物合成
        self._wall_edge: List[np.ndarray] = []     # 裁盒禁區各邊是否貼在安全盒邊界 (貼牆的邊不可當投影目標)
        self._nodes = np.zeros((0, 2))
        self._node_zone = np.zeros(0, dtype=int)
        self._adj = np.zeros((0, 0), dtype=bool)
        self._anchor = np.zeros(2)                 # 可達性錨點 (= 起飛點, 在禁區內時先推出到可達邊界)
        self._node_reach = np.zeros(0, dtype=bool)  # 各靜態節點是否與錨點連通
        self._build()

    # ---- 建構 ----
    def _build(self):
        K = 4 * self.corner_segments
        over = set(self.over_indices)
        self.over = [OverZone(i, inflate(o.footprint, self.clearance, K, o.yaw),
                              o.z_top + self.vertical_clearance)
                     for i, o in enumerate(self.obstacles) if i in over]
        around = [i for i in range(len(self.obstacles)) if i not in over]
        zones = [inflate(self.obstacles[i].footprint, self.clearance, K, self.obstacles[i].yaw) for i in around]
        members = [[i] for i in around]
        # 重疊 / 相碰的禁區合併成凸包 (中間的縫本來就 < 2×clearance, 不能穿)
        merged = True
        while merged and len(zones) > 1:
            merged = False
            for i in range(len(zones)):
                for j in range(i + 1, len(zones)):
                    if len(zones[i]) >= 3 and len(zones[j]) >= 3 and polygons_intersect(zones[i], zones[j]):
                        zones[i] = convex_hull(np.vstack([zones[i], zones[j]]))
                        members[i] += members[j]
                        del zones[j], members[j]
                        merged = True
                        break
                if merged:
                    break
        b = self.box
        self.zones, self.zones_clipped, self.zone_members, self._wall_edge = [], [], [], []
        for z, m in zip(zones, members):
            if len(z) < 3:
                continue
            c = clip_to_rect(z, b.x_min, b.x_max, b.y_min, b.y_max)
            if len(c) < 3 or polygon_area(c) < 1e-9:
                continue                      # 整個在安全盒外 -> 盒內的線段碰不到它
            self.zones.append(z)
            self.zones_clipped.append(c)
            self.zone_members.append(m)
            v0, v1 = c, np.roll(c, -1, axis=0)
            tol = 1e-7
            on_wall = np.zeros(len(c), dtype=bool)
            for side_val, axis in ((b.x_min, 0), (b.x_max, 0), (b.y_min, 1), (b.y_max, 1)):
                on_wall |= (np.abs(v0[:, axis] - side_val) < tol) & (np.abs(v1[:, axis] - side_val) < tol)
            self._wall_edge.append(on_wall)
        # 可視圖節點 = 裁盒禁區的頂點 (在盒內; 落在別的禁區內部者除外, 例如被禁區蓋住的盒角);
        # 靜態相鄰矩陣: 兩頂點連線不穿任何禁區內部 (沿邊走可以)
        if self.zones:
            nodes = np.vstack(self.zones_clipped)
            node_zone = np.concatenate([np.full(len(z), k) for k, z in enumerate(self.zones_clipped)])
            ok = self.zones_of(nodes) < 0
            self._nodes, self._node_zone = nodes[ok], node_zone[ok]
            n = len(self._nodes)
            ii, jj = np.triu_indices(n, k=1)
            blocked = self.blocked_many(self._nodes[ii], self._nodes[jj])
            adj = np.zeros((n, n), dtype=bool)
            adj[ii, jj] = ~blocked
            adj |= adj.T
            self._adj = adj
        self._anchor, self._node_reach = self._pick_anchor()

    def _pick_anchor(self) -> Tuple[np.ndarray, np.ndarray]:
        """可達性錨點 = 起飛點 self.home (夾到盒內)。在禁區內時, 從該禁區的邊界候選點中挑
        「連得到最多節點」者 (避免推到禁區與牆之間的口袋裡)。回傳 (錨點, 各靜態節點是否與錨點連通)。

        注意: 錨點會影響 avoid_lap 的結果 —— 障礙物把房間切成兩半時, 路徑會被收攏到「飛機出發的
        那一半」。這是刻意的: 換了起飛點就該換那一半。
        """
        b = self.box
        home = np.array([min(max(float(self.home[0]), b.x_min), b.x_max),
                         min(max(float(self.home[1]), b.y_min), b.y_max)])
        k = self.zone_of(home)
        if k < 0:
            return home, self._reach_from(home)
        best, best_reach, best_n = None, None, -1
        for c in self._edge_candidates(k) + self._vertex_candidates(k):
            if self.zone_of(c) >= 0:
                continue
            r = self._reach_from(c)
            if int(r.sum()) > best_n:
                best, best_reach, best_n = c, r, int(r.sum())
        if best is None:
            c = self.nearest_outside(home, k)
            return c, self._reach_from(c)
        return best, best_reach

    def _reach_from(self, a: np.ndarray) -> np.ndarray:
        """從點 a 出發, 沿靜態可視圖 (BFS) 可到達的節點。"""
        n = len(self._nodes)
        if n == 0:
            return np.zeros(0, dtype=bool)
        reach = ~self.blocked_many(np.repeat(a[None], n, axis=0), self._nodes)
        frontier = list(np.where(reach)[0])
        while frontier:
            u = frontier.pop()
            nb = np.where(self._adj[u] & ~reach)[0]
            reach[nb] = True
            frontier.extend(nb.tolist())
        return reach

    def reachable_many(self, P: np.ndarray) -> np.ndarray:
        """(M,2) 各點是否從 HOME 可達 (不在禁區內、也不在禁區與牆封住的口袋裡)。"""
        P = np.asarray(P, dtype=float).reshape(-1, 2)
        m = len(P)
        if m == 0:
            return np.zeros(0, dtype=bool)
        ok = ~self.blocked_many(P, np.repeat(self._anchor[None], m, axis=0))
        R = self._nodes[self._node_reach] if len(self._nodes) else np.zeros((0, 2))
        todo = np.where(~ok)[0]
        if len(R) and len(todo):
            PP = np.repeat(P[todo], len(R), axis=0)
            RR = np.tile(R, (len(todo), 1))
            free = ~self.blocked_many(PP, RR)
            ok[todo] = free.reshape(len(todo), len(R)).any(axis=1)
        return ok

    def reachable(self, p) -> bool:
        return bool(self.reachable_many(np.asarray(p, dtype=float)[None])[0])

    # ---- 查詢 ----
    @property
    def active(self) -> bool:
        return self.enabled and bool(self.zones)

    @property
    def approach_field(self) -> "ObstacleField":
        """所有箱子都當水平禁區的版本 (進場段用)；沒有可越過的箱子時就是自己。"""
        if not self.over:
            return self
        if self._approach_field is None:
            self._approach_field = ObstacleField(self.obstacles, self.box, self.clearance,
                                                 self.corner_segments, self.enabled,
                                                 home=self.home)
        return self._approach_field

    def altitude_floor(self, lap_xy: np.ndarray, slope_up: float, slope_dn: float) -> Optional["AltitudeFloor"]:
        """單圈折線上因越過低矮箱子而需要的高度下限 (None = 沒有可越過的箱子 / 路徑沒經過)。
        兩個平台之間若短到「降下去又得馬上爬回來」(間隙 < 垂直安全距離的下坡 + 上坡長度) 就合併成一個平台。"""
        if not self.enabled or not self.over:
            return None
        gap = self.vertical_clearance * (1.0 / max(slope_up, 1e-6) + 1.0 / max(slope_dn, 1e-6))
        fl = AltitudeFloor(lap_xy, self.over, slope_up, slope_dn, merge_gap=gap)
        return fl if fl.plateaus else None

    def zone_of(self, p) -> int:
        """點嚴格在哪個禁區內 (-1 = 都不在)。"""
        p = np.asarray(p, dtype=float).reshape(1, 2)
        for k, z in enumerate(self.zones):
            if points_inside(p, z)[0]:
                return k
        return -1

    def zones_of(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float).reshape(-1, 2)
        out = np.full(len(P), -1, dtype=int)
        for k, z in enumerate(self.zones):
            out[(out < 0) & points_inside(P, z)] = k
        return out

    def blocked_many(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float).reshape(-1, 2)
        Q = np.asarray(Q, dtype=float).reshape(-1, 2)
        out = np.zeros(len(P), dtype=bool)
        for z in self.zones:
            out |= segments_enter(P, Q, z)
        return out

    def blocked(self, p, q) -> bool:
        return bool(self.blocked_many(np.asarray(p)[None], np.asarray(q)[None])[0])

    def _boundary_candidates(self, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """禁區 k (裁盒後) 各『非牆』邊 (A, B) 列表 —— 投影 / 推出點只能落在這些邊上 (仍在盒內)。"""
        z = self.zones_clipped[k]
        out = []
        for i in range(len(z)):
            if not self._wall_edge[k][i]:
                out.append((z[i], z[(i + 1) % len(z)]))
        return out

    def nearest_outside(self, p, k: int) -> np.ndarray:
        """禁區 k 內的點 -> 最近的邊界點 (往外推 _PUSH; 跳過貼牆的邊)。"""
        p = np.asarray(p, dtype=float)
        best, best_d = p.copy(), np.inf
        for a, b in self._boundary_candidates(k):
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0.0, 1.0))
            q = a + t * ab
            d = float(np.linalg.norm(q - p))
            if d < best_d:
                best_d, best = d, self._push_out(q, a, b)
        return best

    def exit_point(self, p, k: int) -> np.ndarray:
        """禁區 k 內的點 -> 最近的『可達』邊界點 (不會推進口袋裡); 沒有可達候選時退回最近邊界點。"""
        p = np.asarray(p, dtype=float)
        cands = [c for c in self._edge_candidates(k, p) + self._vertex_candidates(k)
                 if self.zone_of(c) < 0]
        if cands:
            ok = self.reachable_many(np.array(cands))
            cands = [c for c, r in zip(cands, ok) if r]
        if not cands:
            return self.nearest_outside(p, k)
        return min(cands, key=lambda c: float(np.linalg.norm(c - p)))

    def _edge_candidates(self, k: int, p=None) -> List[np.ndarray]:
        """禁區 k 各非牆邊上的候選點: 邊中點 (+ 給了 p 時, 邊上離 p 最近的點)。"""
        out: List[np.ndarray] = []
        for a, b in self._boundary_candidates(k):
            ab = b - a
            out.append(self._push_out(a + 0.5 * ab, a, b))
            if p is not None:
                t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0.0, 1.0))
                out.append(self._push_out(a + t * ab, a, b))
        return out

    def _vertex_candidates(self, k: int) -> List[np.ndarray]:
        """禁區 k (裁盒後) 的頂點往外推一點 (被禁區蓋住的盒角除外; 只沿非牆邊法向推, 不會出盒)。"""
        z = self.zones_clipped[k]
        wall = self._wall_edge[k]
        out: List[np.ndarray] = []
        for i in range(len(z)):
            e0 = z[i] - z[i - 1]
            e1 = z[(i + 1) % len(z)] - z[i]
            n0 = np.array([e0[1], -e0[0]]) / max(np.linalg.norm(e0), 1e-15)
            n1 = np.array([e1[1], -e1[0]]) / max(np.linalg.norm(e1), 1e-15)
            w0, w1 = bool(wall[i - 1]), bool(wall[i])
            if w0 and w1:
                continue
            nn = (n1 if w0 else n0 if w1 else n0 + n1)
            q = z[i] + _PUSH * nn / max(np.linalg.norm(nn), 1e-15)
            if self.box.contains(q[0], q[1], self.box.z_min):
                out.append(q)
        return out

    @staticmethod
    def _push_out(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        e = b - a
        n = np.array([e[1], -e[0]])
        n /= max(np.linalg.norm(n), 1e-15)
        return q + _PUSH * n

    def free_path(self, a, b) -> Optional[List[np.ndarray]]:
        """a -> b 在安全盒內不穿禁區的最短折線 (含 a、b)。起/終點若在禁區內, 先走到最近邊界。
        無路可走回 None。"""
        a = np.asarray(a, dtype=float).reshape(2)
        b = np.asarray(b, dtype=float).reshape(2)
        prefix: List[np.ndarray] = []
        suffix: List[np.ndarray] = []
        za, zb = self.zone_of(a), self.zone_of(b)
        if za >= 0:
            prefix, a = [a], self.exit_point(a, za)
        if zb >= 0:
            suffix, b = [b], self.exit_point(b, zb)
        if not self.blocked(a, b):
            return prefix + [a, b] + suffix
        n = len(self._nodes)
        if n == 0:
            return None
        nodes = np.vstack([self._nodes, a[None], b[None]])
        adj = np.zeros((n + 2, n + 2), dtype=bool)
        adj[:n, :n] = self._adj
        for idx, p in ((n, a), (n + 1, b)):
            free = ~self.blocked_many(np.repeat(p[None], n, axis=0), self._nodes)
            adj[idx, :n] = free
            adj[:n, idx] = free
        path = _dijkstra(nodes, adj, n, n + 1)
        if path is None:
            return None
        return prefix + [nodes[i] for i in path] + suffix

    def path_length(self, a, b) -> float:
        p = self.free_path(a, b)
        return _polyline_length(np.array(p)) if p else math.inf

    def best_boundary_point(self, p, k: int = -1, prev=None, nxt=None) -> np.ndarray:
        """禁區內 (k >= 0) 或口袋裡 (k < 0, 從 HOME 到不了) 的頂點 -> 禁區邊界上『可達』且讓
        「prev -> 點 -> nxt」總路長最短的點 (候選: 各非牆邊上離 p 最近的點、邊中點、各頂點；
        先試該禁區, 沒有可達候選再試全部禁區)。鄰點到不了的就不計入。"""
        p = np.asarray(p, dtype=float)
        order = ([k] if k >= 0 else []) + [j for j in range(len(self.zones)) if j != k]
        cands: List[np.ndarray] = []
        for j in order:
            cs = [c for c in self._edge_candidates(j, p) + self._vertex_candidates(j) if self.zone_of(c) < 0]
            if cs:
                ok = self.reachable_many(np.array(cs))
                cands = [c for c, r in zip(cs, ok) if r]
            if cands:
                break
        if not cands:
            return self.nearest_outside(p, k) if k >= 0 else p.copy()
        nbrs = [np.asarray(q, dtype=float) for q in (prev, nxt) if q is not None]
        nbrs = [q for q in nbrs if self.reachable(q)]
        best, best_cost = None, math.inf
        for c in cands:
            cost = sum(self.path_length(q, c) for q in nbrs)
            cost += 1e-3 * float(np.linalg.norm(c - p))    # 同長 (或沒有可達鄰點) 時取離原頂點近者
            if cost < best_cost:
                best, best_cost = c, cost
        return best

    # ---- 對 pattern 單圈折線避障 ----
    def avoid_lap(self, lap_xy: np.ndarray, is_smooth: bool = False) -> Tuple[np.ndarray, AvoidInfo]:
        lap = _dedupe(np.asarray(lap_xy, dtype=float).reshape(-1, 2).copy())
        info = AvoidInfo(length_before=_polyline_length(lap))
        info.length_after = info.length_before
        if not self.active or len(lap) < 2:
            return lap, info
        closed = bool(np.allclose(lap[0], lap[-1]))
        inside = self.zones_of(lap)
        pocket = (inside < 0) & ~self.reachable_many(lap)   # 不在禁區內, 但在禁區貼牆封出的口袋裡 (HOME 到不了)
        bad = (inside >= 0) | pocket
        info.unreachable = int(pocket.sum()) - (1 if (closed and pocket[0] and len(lap) > 1) else 0)

        # 1) 起點 (閉合時 = 終點): 推到禁區邊界上可達、且前後最省路的點
        if bad[0]:
            prev_pt = lap[-2] if (closed and len(lap) > 2) else None
            next_pt = lap[1] if len(lap) > 1 else None
            c = self.best_boundary_point(lap[0], int(inside[0]), prev_pt, next_pt)
            lap[0] = c
            bad[0] = False
            if closed:
                lap[-1] = c
                bad[-1] = False
            info.moved += 1
        if bad[-1]:              # 未閉合的終點
            lap[-1] = self.best_boundary_point(lap[-1], int(inside[-1]), lap[-2], None)
            bad[-1] = False
            info.moved += 1

        # 2) 內部頂點
        if is_smooth:
            keep = ~bad
            keep[0] = keep[-1] = True
            info.removed = int((~keep).sum())
            lap = lap[keep]
        else:
            for i in range(1, len(lap) - 1):
                if bad[i]:
                    lap[i] = self.best_boundary_point(lap[i], int(inside[i]), lap[i - 1], lap[i + 1])
                    info.moved += 1

        # 3) 逐段改道 (改道段就是禁區邊上的幾個轉角, 不加密: 曲線 pattern 匯出時本來就會沿弧長重取樣,
        #    速度剖面對稀疏的真轉角也會用切角模型而不是把它當成極大曲率)
        blocked = self.blocked_many(lap[:-1], lap[1:])
        out: List[np.ndarray] = [lap[0]]
        for i in range(len(lap) - 1):
            q = lap[i + 1]
            if blocked[i]:
                path = self.free_path(lap[i], q)
                if path is None:
                    info.failed += 1
                    out.append(q)
                    continue
                info.detours += 1
                out.extend(np.array(path)[1:])
            else:
                out.append(q)
        lap2 = _dedupe(np.array(out))
        if closed and len(lap2) > 1:
            lap2[-1] = lap2[0]
        info.length_after = _polyline_length(lap2)
        return lap2, info

    # ---- 距離 (安全檢查用; 以「原始」底面為準) ----
    def distances(self, xy: np.ndarray) -> List[float]:
        """各障礙物到一組點的最小帶號距離 (m; 負 = 有點在障礙物內)。"""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if len(xy) == 0:
            return [math.inf] * len(self.obstacles)
        return [float(signed_distance(xy, o.footprint).min()) for o in self.obstacles]

    def describe(self) -> str:
        n_v = sum(1 for o in self.obstacles if o.source == "vrpn")
        s = f"障礙物 {len(self.obstacles)} 個"
        if n_v:
            s += f" (VRPN {n_v})"
        s += f", 安全距離 {self.clearance:.2f} m"
        if not self.enabled:
            s += " [避障已停用]"
            return s
        if self.over:
            names = ", ".join(self.obstacles[z.index].name for z in self.over)
            s += (f"; 越過 {len(self.over)} 個 [{names}] (箱頂+{self.vertical_clearance:.2f} m), "
                  f"繞開 {len(self.obstacles) - len(self.over)} 個")
        n_around = len(self.obstacles) - len(self.over)
        if len(self.zones) != n_around:
            s += f", 禁區 {len(self.zones)} 塊 (相鄰者已合併 / 盒外者略過)"
        return s


class AltitudeFloor:
    """沿單圈折線的高度下限 z_floor(s)：路徑在可越過箱子的禁區內 -> 平台 z_req；平台前以 slope_up、
    平台後以 slope_dn (每公尺水平距離的高度變化) 線性斜坡；週期 = 單圈長 L (s 以 mod L 求值, 斜坡可跨圈)。
    最終高度 = max(高度剖面, z_floor)。"""

    def __init__(self, lap_xy: np.ndarray, zones: Sequence[OverZone], slope_up: float, slope_dn: float,
                 merge_gap: float = 0.0):
        lap = _dedupe(np.asarray(lap_xy, dtype=float).reshape(-1, 2))
        seg = np.hypot(*np.diff(lap, axis=0).T) if len(lap) > 1 else np.zeros(0)
        s_v = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(s_v[-1]) if len(s_v) else 0.0
        self.slope_up = max(float(slope_up), 1e-6)
        self.slope_dn = max(float(slope_dn), 1e-6)
        self.plateaus: List[Tuple[float, float, float, int]] = []    # (a, b, z_req, obstacle index)
        if len(lap) < 2 or self.L <= 0:
            return
        P, Q = lap[:-1], lap[1:]
        for z in zones:
            t0, t1, hit = segment_intervals(P, Q, z.poly, eps=-1e-9)     # 含邊界
            iv = [(s_v[i] + t0[i] * seg[i], s_v[i] + t1[i] * seg[i]) for i in np.where(hit)[0]]
            for a, b in _merge_intervals(iv):
                self.plateaus.append((float(a), float(b), float(z.z_req), int(z.index)))
        self.plateaus.sort()
        # 相鄰平台間隙太短 (含跨圈) -> 合併成一個平台 (高度取較高者), 免得航點弦線在間隙處切到箱子上方
        if merge_gap > 0 and len(self.plateaus) > 1:
            merged: List[Tuple[float, float, float, int]] = []
            for a, b, zr, idx in self.plateaus:
                if merged and a - merged[-1][1] < merge_gap:
                    pa, pb, pz, pi = merged[-1]
                    merged[-1] = (pa, max(pb, b), max(pz, zr), pi if pz >= zr else idx)
                else:
                    merged.append((a, b, zr, idx))
            if len(merged) > 1 and (merged[0][0] + self.L) - merged[-1][1] < merge_gap:
                a0, b0, z0, i0 = merged[0]
                a1, b1, z1, i1 = merged[-1]
                merged[0] = (a1 - self.L, b0, max(z0, z1), i0 if z0 >= z1 else i1)   # 跨圈: 起點往前延伸
                merged.pop()
            self.plateaus = merged

    def __call__(self, s) -> np.ndarray:
        s = np.mod(np.asarray(s, dtype=float), self.L) if self.L > 0 else np.asarray(s, dtype=float)
        out = np.full(s.shape, -np.inf)
        for a, b, zr, _ in self.plateaus:
            for shift in (-self.L, 0.0, self.L):
                sa, sb = a + shift, b + shift
                val = np.where(s < sa, zr - self.slope_up * (sa - s),
                               np.where(s > sb, zr - self.slope_dn * (s - sb), zr))
                out = np.maximum(out, val)
        return out

    def anchors(self, s_grid: np.ndarray, z_profile: np.ndarray) -> np.ndarray:
        """匯出航點時必須保留的弧長位置 (單圈內, [0, L))：平台起訖點 + 斜坡與高度剖面的交會點。
        沒有這些點, 稀疏航點之間的直線內插會切到平台 / 斜坡下方。"""
        pts = []
        for a, b, _, _ in self.plateaus:
            pts += [a % self.L, b % self.L]
        s_grid = np.asarray(s_grid, dtype=float)
        if len(s_grid) > 1:
            active = self(s_grid) > np.asarray(z_profile, dtype=float) + 1e-9
            edges = np.where(active[1:] != active[:-1])[0]
            for i in edges:
                pts.append(float(s_grid[i + 1] if active[i + 1] else s_grid[i]) % self.L)
        if not pts:
            return np.zeros(0)
        arr = np.unique(np.round(np.array(pts), 6))
        return arr[(arr >= 0) & (arr < self.L)]


def _merge_intervals(iv: List[Tuple[float, float]], tol: float = 1e-6) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + tol:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def over_ramp_slopes(cfg: Dict) -> Tuple[float, float, float, float]:
    """越過箱子的斜坡：回傳 (上升垂直速度, 下降垂直速度, 上升坡度, 下降坡度)；坡度 = 垂直速度 / 巡航速度
    (每公尺水平距離的高度變化)。obstacles.over_ramp_speed = auto -> OVER_RAMP_FRAC × min(speed_up/down,
    √(max_speed² − cruise²)) (同 stair 模式的 auto 邏輯: 不觸發垂直速度警告、3D 合成速度不超過上限)。"""
    o = cfg.get("obstacles", {}) or {}
    f = cfg.get("flight", {}) or {}
    cruise = max(float(f.get("cruise_speed", 0.5)), 1e-6)
    v = o.get("over_ramp_speed", "auto")
    if v is None or (isinstance(v, str) and v.strip().lower() == "auto"):
        vmax = float(f.get("max_speed", 1.0))
        room = float(np.sqrt(max(vmax * vmax - cruise * cruise, 0.0)))
        v_up = OVER_RAMP_FRAC * min(float(f.get("speed_up", 1.0)), room)
        v_dn = OVER_RAMP_FRAC * min(float(f.get("speed_down", 0.6)), room)
    else:
        v_up = v_dn = float(v)
    v_up, v_dn = max(v_up, 1e-3), max(v_dn, 1e-3)
    return v_up, v_dn, v_up / cruise, v_dn / cruise


def _dijkstra(nodes: np.ndarray, adj: np.ndarray, src: int, dst: int) -> Optional[List[int]]:
    n = len(nodes)
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[src] = 0.0
    done = np.zeros(n, dtype=bool)
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        if u == dst:
            break
        nb = np.where(adj[u] & ~done)[0]
        if len(nb) == 0:
            continue
        w = np.linalg.norm(nodes[nb] - nodes[u], axis=1)
        for v, wv in zip(nb, w):
            nd = d + float(wv)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, int(v)))
    if not np.isfinite(dist[dst]):
        return None
    path = [dst]
    while path[-1] != src:
        path.append(int(prev[path[-1]]))
    return path[::-1]


# ----------------------------------------------------------------------
# config 入口
# ----------------------------------------------------------------------
def resolve_clearance(cfg: Dict) -> float:
    """水平安全距離 (m)。obstacles.clearance = 'auto' -> 由機身尺寸推導 (見 recommended_clearance)。"""
    v = (cfg.get("obstacles", {}) or {}).get("clearance", DEFAULT_CLEARANCE)
    if v is None or (isinstance(v, str) and v.strip().lower() == "auto"):
        return recommended_clearance(cfg)
    return float(v)


def field_from_config(cfg: Dict, box: SafeBox) -> ObstacleField:
    o = cfg.get("obstacles", {}) or {}
    obstacles = obstacles_from_config(cfg)
    vclear = float(o.get("vertical_clearance", DEFAULT_VERTICAL_CLEARANCE))
    max_top = float(o.get("over_max_top", DEFAULT_OVER_MAX_TOP))
    low_mode = str(o.get("low_mode", "over")).strip().lower()
    over = []
    if low_mode == "over":
        # 箱頂夠低、且拉高後仍在天花板邊界內的才越過; 其餘繞開
        over = [i for i, ob in enumerate(obstacles)
                if ob.z_top <= max_top + 1e-9 and ob.z_top + vclear <= box.z_max + 1e-9]
    return ObstacleField(
        obstacles, box,
        clearance=resolve_clearance(cfg),
        corner_segments=int(o.get("corner_segments", DEFAULT_CORNER_SEGMENTS)),
        enabled=bool(o.get("enabled", True)),
        over_indices=over, vertical_clearance=vclear,
        home=takeoff_point(cfg),
    )
