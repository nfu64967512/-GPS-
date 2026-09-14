"""Random walk — 隨機手飛風格路徑。

模擬人手飛行在房間裡「隨意繞」：一連串隨機的轉折點 (步幅、轉向角都隨機, 但方向有慣性、
偶爾大轉彎)，再用閉合的 centripetal Catmull-Rom 樣條穿過所有轉折點, 成為平滑、起點==終點
的曲線 (單圈)。曲線最後整體縮放到剛好填滿可用盒 (width_fill / height_fill)。

* 決定性：同一個 seed 一定產生同一條路徑 (GUI 每次「產生」與圈數細修都會重算, 必須一致);
  換 seed 就換一條。
* is_smooth=True -> 匯出用 SPLINE 航點；配合 altitude 的 smooth_random 高度就是全隨機 3D 手飛。
* 隨機飛行不重複：repeatable=False (不用 DO_JUMP)，且 n_points=auto (預設) 時整條路徑的長度
  自動配合 flight.target_duration × cruise_speed —— 一圈就是整段飛行 (圈數自動為 1)。
  想要固定轉折點數就給整數 (路徑變短時圈數會自動 >1, 但仍是全展開、各圈高度相位錯開)。

參數 (config patterns.random_walk):
  seed            亂數種子 (換一條路徑)
  n_points        auto = 依目標工時自動決定轉折點數 (整段不重複); 或給整數 (>= 4)
  auto_length_frac n_points=auto 時, 路徑長 = 此比例 × target_duration × cruise_speed
                  (1.0 時 AUTO 航線時間平均 ≈ 目標; 轉角減速拉長的時間與 spline 航點切角縮短的
                  時間大致抵銷)
  turn_sigma_deg  每步轉向角的標準差 (deg); 越大越「亂」, 越小越像平滑巡航
  turn_max_deg    每步轉向角上限 (deg, 預設 110): 避免髮夾彎 (實機要近懸停才轉得過, spline
                  航點也畫不出來); 飛回起點閉合時同樣受限
  step_min_fill / step_max_fill  每步步幅佔盒寬 (min(寬,高)) 的比例範圍
  width_fill / height_fill       佔可用盒的比例 (1.0 = 用滿)
  samples_per_seg 每段樣條取樣點數 (密度; 匯出仍依 point_spacing 重取)
"""

from __future__ import annotations

import numpy as np

from .base import PatternResult, pattern_cfg

_ALPHA = 0.5   # centripetal Catmull-Rom (不會打結/過衝)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _random_control_points(rng: np.random.Generator, n: int, hx: float, hy: float,
                           step_min: float, step_max: float, turn_sigma: float,
                           turn_max: float) -> np.ndarray:
    """有方向慣性的隨機轉折點 + 平順飛回起點閉合。回傳 (m, 2), m >= n。

    * 每步：步幅 ~U(step_min, step_max)×盒寬, 轉向角 ~N(0, σ) 夾在 ±turn_max (避免髮夾彎)；
      出界就換個方向重試, 重試不成就在轉角上限內朝盒中心轉、縮步幅。
    * 走完 n 點後「飛回起點」：在轉角上限內逐步轉向「起點後方的進場點」再到起點, 讓閉合處
      (最後一段 -> 第一段) 的轉角也在上限內、進入起點的方向 ≈ 第一段方向。
    """
    size = 2.0 * min(hx, hy)
    lim = np.array([hx, hy])
    p0 = rng.uniform(-0.6, 0.6, size=2) * lim
    pts = [p0]
    heading = float(rng.uniform(-np.pi, np.pi))
    h0 = heading                                       # 第一段方向 (由 p0 出發)

    def inside(c):
        return bool(np.all(np.abs(c) <= lim))

    def step_from(prev, h, step):
        return prev + step * np.array([np.cos(h), np.sin(h)])

    def toward(prev, target, h_now, s_min, s_max):
        """在轉角上限內朝 target 走一步 (步幅夾在 [s_min, s_max] 且不超過到 target 的距離)。"""
        d = target - prev
        dist = float(np.hypot(*d))
        h_t = float(np.arctan2(d[1], d[0]))
        h = h_now + float(np.clip(_wrap(h_t - h_now), -turn_max, turn_max))
        step = float(np.clip(dist, s_min, s_max))
        c = step_from(prev, h, step)
        while not inside(c) and step > 0.05 * size:
            step *= 0.5
            c = step_from(prev, h, step)
        return np.clip(c, -lim, lim), h

    for _ in range(1, n):
        cand = None
        for _attempt in range(30):
            step = float(rng.uniform(step_min, step_max)) * size
            dh = float(np.clip(rng.normal(0.0, turn_sigma), -turn_max, turn_max))
            c = step_from(pts[-1], heading + dh, step)
            if inside(c):
                cand = c
                break
        if cand is None:                              # 卡在角落: 朝盒中心轉 (轉角上限內)
            cand, _ = toward(pts[-1], np.zeros(2), heading, step_min * size, step_max * size)
        heading = float(np.arctan2(cand[1] - pts[-1][1], cand[0] - pts[-1][0]))
        pts.append(cand)

    # 飛回起點: 先到「起點後方」的進場點 (讓進入起點的方向 ≈ 第一段方向 h0), 再到起點
    approach = p0 - (step_min * size) * np.array([np.cos(h0), np.sin(h0)])
    approach = np.clip(approach, -lim, lim)
    for target in (approach, p0):
        for _guard in range(40):
            d = target - pts[-1]
            dist = float(np.hypot(*d))
            need = abs(_wrap(float(np.arctan2(d[1], d[0])) - heading))
            if dist <= step_max * size and need <= turn_max:
                break                                  # 一步可到且轉角在上限內 -> 下一個目標
            cand, heading = toward(pts[-1], target, heading, step_min * size, step_max * size)
            if float(np.hypot(*(cand - pts[-1]))) < 0.02 * size:
                break                                  # 幾乎沒前進 (極端角落), 放棄再修
            pts.append(cand)
        if target is approach:
            # 進場點本身不當轉折點 (太靠近起點會做出小圈); 只用它來把方向修正到 ≈ h0
            pass
    # 最後一點若離起點太近 (< 0.25 步), 拿掉以免閉合段太短
    if len(pts) > n and float(np.hypot(*(pts[-1] - p0))) < 0.25 * step_min * size:
        pts.pop()
    return np.asarray(pts, dtype=float)


def _catmull_rom_closed(P: np.ndarray, samples_per_seg: int, alpha: float = _ALPHA) -> np.ndarray:
    """閉合 centripetal Catmull-Rom：穿過每個控制點, 回傳密取樣折線 (首尾同點)。"""
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** alpha
        t = np.linspace(t1, t2, samples_per_seg, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
        out.append(C)
    curve = np.vstack(out)
    return np.vstack([curve, curve[:1]])          # 閉合：終點 == 起點 (精確)


def _build_lap(seed: int, n_points: int, hx: float, hy: float, step_min: float, step_max: float,
               turn_sigma: float, turn_max: float, samples_per_seg: int) -> np.ndarray:
    """seed + 轉折點數 -> 閉合曲線 (保證在 [-hx,hx]×[-hy,hy] 內)。同 seed 下, 前 n 個轉折點是同一條
    隨機漫步的前綴 (rng 循序消耗), 所以 n 越大路徑大致越長 (自動長度搜尋才會收斂)。
    只「縮」不「放」：樣條在轉折點外側稍微鼓出盒子時, 整條置中後等比例縮小到剛好在盒內；
    不放大 (放大會讓路徑長度隨走的範圍劇烈跳動, 且不像手飛)。"""
    rng = np.random.default_rng(seed)
    ctrl = _random_control_points(rng, n_points, hx, hy, step_min, step_max, turn_sigma, turn_max)
    curve = _catmull_rom_closed(ctrl, samples_per_seg)
    lo, hi = curve.min(axis=0), curve.max(axis=0)
    curve = curve - (lo + hi) / 2.0                                   # 置中 (只平移)
    ext = np.maximum(np.abs(curve).max(axis=0), 1e-9)
    scale = min(1.0, hx / ext[0], hy / ext[1])                        # 只縮不放
    curve = curve * scale
    curve[-1] = curve[0]                                               # 仍精確閉合
    return curve


def _length(curve: np.ndarray) -> float:
    return float(np.hypot(*np.diff(curve, axis=0).T).sum())


def generate(box, cfg, key, display_name) -> PatternResult:
    p = pattern_cfg(cfg, key)
    seed = int(p.get("seed", 0))
    turn_sigma = np.radians(float(p.get("turn_sigma_deg", 60.0)))
    turn_max = np.radians(float(np.clip(float(p.get("turn_max_deg", 110.0)), 20.0, 175.0)))
    step_min = float(p.get("step_min_fill", 0.35))
    step_max = max(step_min, float(p.get("step_max_fill", 0.9)))
    samples_per_seg = max(8, int(p.get("samples_per_seg", 60)))
    hx = box.half_x * float(p.get("width_fill", 1.0))
    hy = box.half_y * float(p.get("height_fill", 1.0))
    args = (hx, hy, step_min, step_max, turn_sigma, turn_max, samples_per_seg)

    n_cfg = p.get("n_points", "auto")
    auto = isinstance(n_cfg, str) and n_cfg.strip().lower() == "auto"
    if auto:
        # 整段飛行一條路徑：路徑長 ≈ frac × 目標工時 × 巡航速度; 用「同 seed 前綴」性質迭代調整轉折點數,
        # 長度對 n 不嚴格單調 (回起點段長度會變), 故記住最接近目標的候選、避免在兩個 n 之間打轉
        f = cfg.get("flight", {})
        target_len = (float(p.get("auto_length_frac", 1.0))
                      * float(f.get("target_duration", 270)) * float(f.get("cruise_speed", 0.5)))
        avg_step = 0.5 * (step_min + step_max) * 2.0 * min(hx, hy)
        n = max(4, int(round(target_len / max(avg_step, 1e-6))))
        best = None
        tried = set()
        for _ in range(10):
            tried.add(n)
            curve_n = _build_lap(seed, n, *args)
            L = _length(curve_n)
            if best is None or abs(L - target_len) < abs(best[2] - target_len):
                best = (n, curve_n, L)
            if abs(L - target_len) <= 0.03 * target_len:
                break
            n_new = max(4, int(round(n * target_len / max(L, 1e-6))))
            if n_new == n or n_new in tried:
                n_new = n + (1 if L < target_len else -1)
            if n_new < 4 or n_new in tried:
                break
            n = n_new
        n, curve, _ = best
    else:
        n = max(4, int(n_cfg))
        curve = _build_lap(seed, n, *args)

    L = _length(curve)
    return PatternResult(
        key=key,
        display_name=display_name,
        lap_xy=curve,
        is_smooth=True,
        description=(f"隨機手飛 {n} 個轉折點{' (自動: 整段一條不重複)' if auto else ''}, "
                     f"路徑 {L:.1f} m, 範圍 {2*hx:.2f} m × {2*hy:.2f} m (seed {seed})"),
        repeatable=False,          # 隨機飛行不用 DO_JUMP 重複
    )
