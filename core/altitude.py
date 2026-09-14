"""
垂直高度調變 z = f(沿路徑進度)。

讓水平圖形在飛行中自由變化高度 -> 成為 3D 軌跡，對 VIO / 預測模型是更好的測試資料。
高度是「沿水平路徑弧長分率 u = s/S」的純函數 (不是時間的函數)：AUTO 任務的航點帶高度、飛控
在航點間沿路徑線性內插，本來就是位置的函數；DO_JUMP 重複的每一圈才會高度完全相同。
模式：sine (正弦) / triangle (三角波, 線性升降) / smooth_random (平滑亂數) /
stair (階梯: 平飛保持高度 -> 以固定垂直速度升/降一階 -> 再平飛, 逐階升到頂再逐階降到底)。
所有輸出都夾限在 SafeBox 的 [z_min, z_max] 安全帶內。
"""

from __future__ import annotations

from math import gcd
from typing import Dict

import numpy as np

from .geometry import SafeBox

# stair 模式 ramp_speed=auto 時的升降速度 = 此比例 × min(垂直上限, 3D 合成速度餘裕)
# (留餘裕給實機位置控制器追蹤; 也避免恰好貼在 safety 的垂直速度警告 / 超速門檻上)
STAIR_AUTO_RAMP_FRAC = 0.8
DEFAULT_STAIR_STEPS = 4


def coprime_cycles(cycles, laps: int) -> int:
    """把高度起伏次數調成與圈數「互質」的最接近整數。

    若 cycles 與 laps 不互質 (例如 cycles==laps、或同為偶數)，每一圈在相同 XY 位置的
    高度會相同 -> 各圈在 3D 中完全重疊，看起來「收斂成一圈」。互質時各圈相位均勻錯開，
    形成 3D 編織/螺旋，每圈皆不同 (對 VIO/預測模型也是更好的資料)。
    """
    c = int(round(float(cycles)))
    if laps <= 1 or c <= 0:
        return c
    if gcd(c, laps) == 1:
        return c
    for d in range(1, laps + 1):          # 找最接近且互質的整數
        for cand in (c + d, c - d):
            if cand >= 1 and gcd(cand, laps) == 1:
                return cand
    return c


def _resolve_base_amp(box: SafeBox, cfg: Dict) -> tuple[float, float]:
    """解析 base 與 amplitude (auto 時用滿安全 z 帶)。"""
    a = cfg.get("altitude", {})
    base = a.get("base", "auto")
    amp = a.get("amplitude", "auto")

    base = box.z_mid if base == "auto" else float(base)
    if amp == "auto":
        # 用安全帶半幅的 ~90%，留一點餘裕避免夾限變平頂
        amp = 0.45 * box.z_span
    else:
        amp = float(amp)

    # 確保 base ± amp 不超出安全帶
    amp = min(amp, box.z_max - base, base - box.z_min)
    amp = max(amp, 0.0)
    return base, amp


def resolve_stair_params(cfg: Dict) -> tuple[int, float, float]:
    """解析 stair 模式參數 -> (steps, 上升速度 m/s, 下降速度 m/s)。

    altitude.steps      : 由最低到最高分幾階 (>=1; 1 = 方波)。
    altitude.ramp_speed : 階與階之間升降的垂直速度 (m/s)。
        auto = STAIR_AUTO_RAMP_FRAC × min(flight.speed_up 或 speed_down,
                                          sqrt(max_speed² − cruise_speed²))
               上升/下降各自取。第二項是「3D 合成速度不超過 max_speed」的餘裕：升降時
               水平仍在巡航, 合成速度 = sqrt(v_h² + v_z²), 否則巡航稍快時 auto 會自己
               觸發超速 ERROR。
        數值 = 升降皆用該速度 (超過上限由 safety 警告 / 擋下)。
    """
    a = cfg.get("altitude", {})
    f = cfg.get("flight", {})
    steps = max(1, int(round(float(a.get("steps", DEFAULT_STAIR_STEPS)))))
    v = a.get("ramp_speed", "auto")
    if v is None or (isinstance(v, str) and v.strip().lower() == "auto"):
        cruise = float(f.get("cruise_speed", 0.5))
        vmax = float(f.get("max_speed", 1.0))
        room_3d = float(np.sqrt(max(vmax * vmax - cruise * cruise, 0.0)))
        v_up = STAIR_AUTO_RAMP_FRAC * min(float(f.get("speed_up", 1.0)), room_3d)
        v_dn = STAIR_AUTO_RAMP_FRAC * min(float(f.get("speed_down", 0.6)), room_3d)
    else:
        v_up = v_dn = float(v)
    return steps, max(v_up, 1e-3), max(v_dn, 1e-3)


def altitude_at_fraction(u: np.ndarray, box: SafeBox, cfg: Dict, total_len: float,
                         period: float | None = None) -> np.ndarray:
    """高度 z 是「沿水平路徑的進度分率 u = s / S」的純函數 (公尺, 相對地板)。

    這正是 AUTO 任務的真實行為 (航點帶高度、飛控在航點間沿路徑線性內插)，也讓 DO_JUMP
    重複的每一圈高度完全相同；GUIDED 軌跡則以 z(s(t)) 取樣，兩者一致。
    total_len = 整段水平路徑長 S (m)，stair 的升降過渡寬度需要它 (過渡水平距離 = 巡航速度 × 一階高 / 升降速度)。
    period    = 週期 (u 的分率, 例如 1/圈數)：給了就「每個週期內」求值 (u mod period)，起伏次數
                cycles 也視為每週期的次數 (= 總次數 × period)。sine/triangle/stair 在 cycles 為週期
                整數倍時本來就週期，結果相同；smooth_random 則變成「每圈同一段隨機形狀」——
                DO_JUMP 重複一圈時需要如此，否則各圈高度會不同、與匯出的 block 不符。
    """
    a = cfg.get("altitude", {})
    mode = a.get("mode", "sine")
    cycles = float(a.get("cycles", 6))
    base, amp = _resolve_base_amp(box, cfg)
    u = np.asarray(u, dtype=float)
    if amp <= 0 or cycles <= 0 or total_len <= 0:
        return np.full_like(u, base)
    periodic = False
    if period is not None and 0 < period < 1:
        u = np.mod(u / period, 1.0)                    # 週期內分率
        u[np.isclose(np.asarray(u), 1.0)] = 0.0
        cycles = cycles * period                       # 每週期起伏次數
        total_len = total_len * period                 # 每週期路徑長
        periodic = True

    if mode == "triangle":
        # 三角波: 用反正弦把正弦轉成線性升降
        z = base + amp * (2.0 / np.pi) * np.arcsin(np.sin(2 * np.pi * cycles * u))
    elif mode == "smooth_random":
        z = _smooth_random(u, base, amp, cycles, int(a.get("seed", 0)), periodic)
    elif mode == "stair":
        steps, v_up, v_dn = resolve_stair_params(cfg)
        cruise = max(float(cfg.get("flight", {}).get("cruise_speed", 0.5)), 1e-6)
        # 以巡航速度走完 S 的等效時間；過渡在路徑上佔的距離 = cruise * h / v_ramp
        z = _stair(u, base, amp, cycles, steps, v_up, v_dn, total_len / cruise)
    else:  # sine (預設)
        z = base + amp * np.sin(2 * np.pi * cycles * u)

    return np.clip(z, box.z_min, box.z_max)


def altitude_profile(t: np.ndarray, box: SafeBox, cfg: Dict, s: np.ndarray,
                     period: float | None = None) -> np.ndarray:
    """回傳與 t 等長的 z 陣列。s = 各取樣點的累積水平弧長 (m, 由 0 起)；z = f(s / S)。
    period 見 altitude_at_fraction (DO_JUMP 一圈重複時 = 1/圈數)。"""
    s = np.asarray(s, dtype=float)
    total = float(s[-1]) if len(s) else 0.0
    if total <= 0:
        base, _ = _resolve_base_amp(box, cfg)
        return np.full(len(t), base)
    return altitude_at_fraction(s / total, box, cfg, total, period=period)


def _stair(
    u: np.ndarray, base: float, amp: float, cycles: float,
    steps: int, v_up: float, v_dn: float, t_equiv: float,
) -> np.ndarray:
    """階梯波：把三角波量化成 steps 階 (steps+1 個高度層)，階與階之間以固定垂直速度
    (上升 v_up / 下降 v_dn) 線性過渡，其餘時間平飛保持高度。

    相位與 sine / triangle 一致：起點在 base、先逐階升到 base+amp，再逐階降到 base-amp，
    回到 base 為一個 cycle。每個 cycle 有 2*steps 個平台；過渡置中於理想的換階位置，故
    z(0)=z(end)=base。u 為路徑進度分率 [0,1]，t_equiv = 以巡航速度走完整段路徑的時間 (s)，
    過渡的「路徑寬度」= 巡航速度 × 一階高 / 升降速度 (在巡航時垂直速度恰為 v_up / v_dn)。
    若過渡 >= 每階寬度 (cycles/steps 太多、路徑太短)，平台消失、退化成三角波，此時垂直
    速度會超過設定值，由 safety 檢查警告。
    """
    phase = cycles * u                                          # 進行到第幾個 cycle
    tri = (2.0 / np.pi) * np.arcsin(np.sin(2 * np.pi * phase))  # 三角波 [-1,1], 起點 0 且上升
    rising = np.cos(2 * np.pi * phase) >= 0                    # 三角波正在上升?
    x = (tri + 1.0) * steps / 2.0                              # 連續「階數座標」[0, steps]
    k = np.floor(x + 0.5)                                       # 目前所在階 (round half up)
    d = x - k                                                   # 距階中心 [-0.5, 0.5)

    # 過渡半寬 (階數座標)。一階高 h=2amp/steps、過渡時間 h/v，而 x 變化率 = 2*steps*cycles/t_equiv
    #   -> 半寬 w = (h/v/2)*(2*steps*cycles/t_equiv) = 2*amp*cycles/(v*t_equiv)
    w = np.where(rising,
                 2.0 * amp * cycles / (v_up * t_equiv),
                 2.0 * amp * cycles / (v_dn * t_equiv))
    w = np.clip(w, 1e-9, 0.5)   # w=0.5 -> 過渡佔滿整階 -> 純三角波

    g = k.copy()
    hi = d > (0.5 - w)          # 靠近上方換階點 (k+0.5): 往 k+1 過渡中
    lo = d < -(0.5 - w)         # 靠近下方換階點 (k-0.5): 往 k-1 過渡中
    g[hi] = k[hi] + (d[hi] - 0.5 + w[hi]) / (2.0 * w[hi])
    g[lo] = k[lo] + (d[lo] + 0.5 - w[lo]) / (2.0 * w[lo])
    g = np.clip(g, 0.0, float(steps))
    return base + amp * (2.0 * g / steps - 1.0)


def _smooth_random(
    u: np.ndarray, base: float, amp: float, cycles: float, seed: int, periodic: bool = False,
) -> np.ndarray:
    """平滑亂數高度: 在數個等距控制點間做三次 Hermite (Catmull-Rom) 內插。

    純函數 (只看 u 與 seed)：軌跡取樣與航點匯出在任意位置求值都一致。
    periodic=False: 控制點鋪滿整段 [0,1]，起終點在 base、端點切線 0 (整段不重複的隨機起伏)。
    periodic=True : 一個週期 (例如 DO_JUMP 的一圈) 內的閉合隨機形狀，u=1 接回 u=0 且切線連續，
                    起點在 base；每個週期形狀相同。
    """
    rng = np.random.default_rng(seed)
    n_ctrl = max(3, int(round(cycles)) + 2)
    z_ctrl = base + amp * rng.uniform(-1.0, 1.0, size=n_ctrl)
    z_ctrl[0] = base  # 起降平順
    uu = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if periodic:
        # n_ctrl 個控制點等分一個週期 (最後一段接回第 0 點), Catmull-Rom 切線含環繞
        h = 1.0 / n_ctrl
        zc = z_ctrl
        m = (np.roll(zc, -1) - np.roll(zc, 1)) / (2.0 * h)
        i = np.clip((uu / h).astype(int), 0, n_ctrl - 1)
        tt = (uu - i * h) / h
        z0, z1 = zc[i], zc[(i + 1) % n_ctrl]
        m0, m1 = m[i], m[(i + 1) % n_ctrl]
    else:
        z_ctrl[-1] = base
        u_ctrl = np.linspace(0.0, 1.0, n_ctrl)
        h = u_ctrl[1] - u_ctrl[0]
        m = np.zeros(n_ctrl)
        m[1:-1] = (z_ctrl[2:] - z_ctrl[:-2]) / (2.0 * h)     # 端點切線 0
        i = np.clip((uu / h).astype(int), 0, n_ctrl - 2)
        tt = (uu - u_ctrl[i]) / h
        z0, z1 = z_ctrl[i], z_ctrl[i + 1]
        m0, m1 = m[i], m[i + 1]
    h00 = 2 * tt**3 - 3 * tt**2 + 1
    h10 = tt**3 - 2 * tt**2 + tt
    h01 = -2 * tt**3 + 3 * tt**2
    h11 = tt**3 - tt**2
    return h00 * z0 + h10 * h * m0 + h01 * z1 + h11 * h * m1
