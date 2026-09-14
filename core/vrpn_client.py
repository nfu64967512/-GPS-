"""
最小 VRPN Tracker 客戶端 (純 Python socket, 不需 vrpn 原生綁定) + Motive 座標 -> 房間 ENU 轉換。

Motive (OptiTrack) 的 VRPN 串流：每個 rigid body = 一個 vrpn_Tracker (名稱 = rigid body 名稱,
sensor 0)，位置 (m) + 四元數；預設 port 3883。本模組只做「連上 -> 收幾秒 -> 每個 tracker 最後
一筆位置 / 姿態 -> 斷線」，用來把貼了光球的箱子 (在 Motive 建成 rigid body) 讀進規劃器當障礙物
(core/obstacles)。read_trackers() 會列出伺服器上所有 tracker，不必事先知道名稱。

協定 (vrpn_Connection 7.x)：
  * TCP 連線後雙方各送 24 bytes cookie "vrpn: ver. 07.35  0" (補 NUL)。
  * 之後為訊息串：header 24 bytes = [總長 (含 header, 不含補齊)][sec][usec][sender][type][seq]，
    皆 big-endian int32；payload 補到 8 bytes 對齊。
  * type < 0 為系統訊息：-1 sender 名稱描述、-2 type 名稱描述 (payload = int32 長度 + 字串 (含 NUL)，
    被描述的 id 放在 header 的 sender 欄)、-3 UDP 描述、-5 斷線。
  * "vrpn_Tracker Pos_Quat" payload (64 bytes) = int32 sensor, int32 padding, float64 pos[3],
    float64 quat[x, y, z, w]。
  * 客戶端沒宣告 UDP port 時伺服器所有訊息都走 TCP (低延遲類別退回 TCP)，只讀 TCP 即可。
"""

from __future__ import annotations

import math
import socket
import struct
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

VRPN_DEFAULT_PORT = 3883
VRPN_MAGIC = b"vrpn: ver. 07.35"
VRPN_COOKIE_SIZE = 24
VRPN_HEADER_SIZE = 24
VRPN_ALIGN = 8

MSG_SENDER_DESCRIPTION = -1
MSG_TYPE_DESCRIPTION = -2
MSG_UDP_DESCRIPTION = -3
MSG_LOG_DESCRIPTION = -4
MSG_DISCONNECT = -5

TRACKER_POS_QUAT = "vrpn_Tracker Pos_Quat"


class VRPNError(RuntimeError):
    pass


class _Closed(Exception):
    """對方關閉了 TCP 連線 (內部用)。"""


@dataclass
class TrackerSample:
    """一個 tracker (rigid body) 最後一筆資料 (伺服器座標系, m)。"""

    name: str
    sensor: int
    pos: np.ndarray      # (3,)
    quat: np.ndarray     # (4,) x, y, z, w
    stamp: float         # 伺服器時間戳 (s)
    count: int = 1       # 期間收到幾筆

    @property
    def key(self) -> str:
        return self.name if self.sensor == 0 else f"{self.name}#{self.sensor}"


# ----------------------------------------------------------------------
# 協定
# ----------------------------------------------------------------------
def make_cookie(log_mode: int = 0) -> bytes:
    return (VRPN_MAGIC + b"  " + bytes([ord("0") + int(log_mode)])).ljust(VRPN_COOKIE_SIZE, b"\0")


def check_cookie(buf: bytes) -> Tuple[int, int]:
    """驗證對方 cookie，回傳 (major, minor)。主版本需為 7。"""
    if not buf.startswith(b"vrpn: ver. "):
        raise VRPNError(f"不是 VRPN 伺服器 (收到 {buf[:16]!r})")
    try:
        ver = buf[11:16].decode("ascii")
        major, minor = (int(v) for v in ver.split("."))
    except Exception as e:  # noqa: BLE001
        raise VRPNError(f"VRPN cookie 版本格式不明: {buf[:24]!r}") from e
    if major != 7:
        raise VRPNError(f"VRPN 主版本 {major} 不相容 (需要 7.x)")
    return major, minor


def pack_message(msg_type: int, sender: int, payload: bytes, stamp: float = 0.0, seq: int = 0) -> bytes:
    """組一則 VRPN 訊息 (header + payload 補齊)。給測試用的假伺服器 / 之後要送訊息時用。"""
    sec = int(stamp)
    usec = int((stamp - sec) * 1e6)
    header = struct.pack(">iiiiii", VRPN_HEADER_SIZE + len(payload), sec, usec, sender, msg_type, seq)
    pad = (-len(payload)) % VRPN_ALIGN
    return header + payload + b"\0" * pad


def pack_description(msg_type: int, which: int, name: str) -> bytes:
    raw = name.encode("utf-8") + b"\0"
    return pack_message(msg_type, which, struct.pack(">i", len(raw)) + raw)


def pack_pos_quat(sensor: int, pos, quat, stamp: float = 0.0) -> bytes:
    p = [float(v) for v in pos]
    q = [float(v) for v in quat]
    return struct.pack(">ii3d4d", int(sensor), 0, *p, *q)


def _recv_exact(sock: socket.socket, n: int, deadline: float) -> Optional[bytes]:
    """讀滿 n bytes；逾時 (deadline) 回 None；對方關閉丟 _Closed。"""
    buf = b""
    while len(buf) < n:
        remain = deadline - time.monotonic()
        if remain <= 0:
            return None
        sock.settimeout(min(remain, 0.5))
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            raise _Closed()
        buf += chunk
    return buf


def parse_server(server: str, default_port: int = VRPN_DEFAULT_PORT) -> Tuple[str, int]:
    """'host[:port]' 或 'Tracker@host[:port]' -> (host, port)。"""
    s = str(server or "").strip()
    if "@" in s:
        s = s.split("@", 1)[1]
    if not s:
        s = "localhost"
    if ":" in s and not s.startswith("["):
        host, port = s.rsplit(":", 1)
        try:
            return host or "localhost", int(port)
        except ValueError:
            return s, default_port
    return s, default_port


def read_trackers(
    server: str,
    seconds: float = 2.0,
    names: Optional[List[str]] = None,
    log: Optional[Callable[[str], None]] = None,
    connect_timeout: float = 3.0,
    exclude: Optional[List[str]] = None,
) -> Dict[str, TrackerSample]:
    """連上 VRPN 伺服器收 `seconds` 秒，回傳 {tracker 名稱: 最後一筆 Pos_Quat}。

    names:   只留這些 tracker (空 / None = 全部)。
    exclude: 略過這些 tracker (例如無人機自己的 rigid body)。
    找不到任何 tracker 時回空 dict (不丟例外)。
    """
    host, port = parse_server(server)
    want = {str(n).strip() for n in (names or []) if str(n).strip()}
    skip = {str(n).strip() for n in (exclude or []) if str(n).strip()}
    say = log or (lambda m: None)
    try:
        sock = socket.create_connection((host, port), timeout=connect_timeout)
    except OSError as e:
        raise VRPNError(f"連不上 VRPN 伺服器 {host}:{port} ({e})") from e
    out: Dict[str, TrackerSample] = {}
    senders: Dict[int, str] = {}
    types: Dict[int, str] = {}
    try:
        sock.sendall(make_cookie())
        deadline = time.monotonic() + max(float(seconds), 0.2) + connect_timeout
        try:
            cookie = _recv_exact(sock, VRPN_COOKIE_SIZE, deadline)
        except _Closed:
            raise VRPNError(f"VRPN 伺服器 {host}:{port} 在交換 cookie 前就關閉連線") from None
        if cookie is None:
            raise VRPNError(f"等待 VRPN cookie 逾時 ({host}:{port})")
        major, minor = check_cookie(cookie)
        say(f"VRPN {host}:{port} 已連線 (協定 {major:02d}.{minor:02d})，收 {seconds:.1f} s ...")
        deadline = time.monotonic() + max(float(seconds), 0.2)
        while True:
            try:
                hdr = _recv_exact(sock, VRPN_HEADER_SIZE, deadline)
                if hdr is None:
                    break
                total, sec, usec, sender, msg_type, _seq = struct.unpack(">iiiiii", hdr)
                payload_len = total - VRPN_HEADER_SIZE
                if payload_len < 0 or payload_len > 1_000_000:
                    raise VRPNError(f"VRPN 訊息長度異常 ({total})")
                padded = payload_len + ((-payload_len) % VRPN_ALIGN)
                body = _recv_exact(sock, padded, deadline) if padded else b""
            except _Closed:
                say("VRPN 伺服器關閉連線 (以目前收到的資料為準)")
                break
            if body is None:
                break
            payload = body[:payload_len]
            if msg_type == MSG_SENDER_DESCRIPTION or msg_type == MSG_TYPE_DESCRIPTION:
                if len(payload) >= 4:
                    (n,) = struct.unpack(">i", payload[:4])
                    name = payload[4:4 + max(n, 0)].split(b"\0", 1)[0].decode("utf-8", "replace")
                    (senders if msg_type == MSG_SENDER_DESCRIPTION else types)[sender] = name
            elif msg_type == MSG_DISCONNECT:
                say("VRPN 伺服器要求斷線")
                break
            elif msg_type >= 0 and types.get(msg_type) == TRACKER_POS_QUAT:
                if len(payload) >= 64:
                    sensor, _pad, px, py, pz, qx, qy, qz, qw = struct.unpack(">ii3d4d", payload[:64])
                elif len(payload) >= 60:      # 舊版無 padding
                    sensor, px, py, pz, qx, qy, qz, qw = struct.unpack(">i3d4d", payload[:60])
                else:
                    continue
                name = senders.get(sender, f"sender{sender}")
                if (want and name not in want) or name in skip:
                    continue
                key = name if sensor == 0 else f"{name}#{sensor}"
                prev = out.get(key)
                out[key] = TrackerSample(
                    name=name, sensor=int(sensor),
                    pos=np.array([px, py, pz], dtype=float),
                    quat=np.array([qx, qy, qz, qw], dtype=float),
                    stamp=sec + usec * 1e-6,
                    count=(prev.count + 1) if prev else 1,
                )
    finally:
        try:
            sock.close()
        except OSError:
            pass
    say(f"VRPN 收到 {len(out)} 個 tracker: {', '.join(sorted(out)) or '(無)'}"
        + (f"; 伺服器有 {len(senders)} 個 sender" if senders and not out else ""))
    return out


# ----------------------------------------------------------------------
# 座標轉換 (Motive -> 房間 ENU)
# ----------------------------------------------------------------------
AXES_PRESETS = {
    "y_up": "x,-z,y",     # Motive 預設 Y-up (右手): ENU x = X, y = -Z, z = Y
    "z_up": "x,y,z",      # Motive 串流設定改 Z-up 時
}


def axes_matrix(spec: str) -> np.ndarray:
    """'x,-z,y' (ENU 的 x/y/z 各取伺服器的哪一軸) 或預設名 (y_up / z_up) -> 3×3 帶號置換矩陣。"""
    s = str(spec or "y_up").strip().lower()
    s = AXES_PRESETS.get(s, s)
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    if len(parts) != 3:
        raise VRPNError(f"axes 需為 y_up / z_up 或 'x,-z,y' 形式 (得到 {spec!r})")
    M = np.zeros((3, 3))
    for row, p in enumerate(parts):
        sign = -1.0 if p.startswith("-") else 1.0
        ax = p.lstrip("+-")
        if ax not in ("x", "y", "z"):
            raise VRPNError(f"axes 軸名需為 x/y/z (得到 {p!r})")
        M[row, "xyz".index(ax)] = sign
    if abs(abs(np.linalg.det(M)) - 1.0) > 1e-9:
        raise VRPNError(f"axes 三個軸需互不相同 (得到 {spec!r})")
    return M


def quat_to_matrix(q) -> np.ndarray:
    """四元數 (x, y, z, w) -> 旋轉矩陣。"""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def to_enu(pos, M: np.ndarray) -> np.ndarray:
    return M @ np.asarray(pos, dtype=float).reshape(3)


def yaw_enu(quat, M: np.ndarray) -> float:
    """rigid body 姿態 -> 在房間 ENU 中繞 z (上) 的偏航 (rad, 0 = +x 東, 逆時針正)。"""
    R = M @ quat_to_matrix(quat) @ M.T
    return float(math.atan2(R[1, 0], R[0, 0]))


def box_dims_from_height(edges, h_meas: float, orientation: str = "auto_square",
                         tol: float = 0.08) -> Tuple[float, float, float, str]:
    """同一種箱子 (三邊 edges, 任意順序) 有躺有立：用量到的高度 h_meas 判斷哪一邊垂直。

    回傳 (sx, sy, sz, 說明)。垂直邊 = 三邊中最接近 h_meas 者 (差 <= tol; 都不像就視為立放, 最長邊垂直)；
    底面 = 另外兩邊, 依 orientation 擺：
      auto_square = 保守取較長邊的正方形 (VRPN 看不出哪一邊沿 x, 預設)
      auto_long_x = 長邊沿 x、短邊沿 y;  auto_long_y = 反之
    """
    e = sorted(float(v) for v in edges)
    if len(e) != 3:
        raise VRPNError("box_size 需為三個邊長")
    k = min(range(3), key=lambda i: abs(e[i] - h_meas))
    if abs(e[k] - h_meas) <= tol:
        sz = e[k]
        others = [e[i] for i in range(3) if i != k]
        pose = "躺放" if sz < e[2] - 1e-9 else "立放"
        why = f"頂面高 {h_meas:.2f} m ≈ {sz:.2f} 邊"
    else:
        sz = e[2]
        others = e[:2]
        pose = "立放?"
        why = f"頂面高 {h_meas:.2f} m 不像任一邊 (箱子不在地上?), 視為最長邊垂直"
    lo, hi = min(others), max(others)
    o = str(orientation).strip().lower()
    if o == "auto_long_x":
        sx, sy = hi, lo
    elif o == "auto_long_y":
        sx, sy = lo, hi
    else:
        sx = sy = hi
    return sx, sy, sz, f"{pose}: {why}, 底面 {sx:.2f}×{sy:.2f}"


DEFAULT_DRONE_NAME = "drone_01"


def read_scene(server: str, vcfg: Dict, log: Optional[Callable[[str], None]] = None,
               connect_timeout: float = 3.0) -> Tuple[Optional[TrackerSample], Dict[str, TrackerSample]]:
    """讀一次 VRPN，分成「飛機」與「箱子」兩份。

    回傳 (飛機的 TrackerSample 或 None, {其他 tracker: TrackerSample})。
    飛機的 rigid body 名稱取自 vcfg['drone'] (預設 drone_01)：它一定會被讀進來 (要當起飛點)，
    但絕不會出現在箱子那一份裡。vcfg['exclude'] 的其他名稱照舊完全略過。

    比對用 sample.name 而不是 dict 的 key —— key 在 sensor != 0 時是 "name#sensor"，
    用 key 比對會同時漏掉飛機、又把飛機當成障礙物箱子。
    """
    drone_name = str(vcfg.get("drone", DEFAULT_DRONE_NAME) or "").strip()
    skip_all = [str(n).strip() for n in (vcfg.get("exclude") or []) if str(n).strip()]
    skip_read = [n for n in skip_all if n != drone_name]          # 飛機要讀進來
    names = [str(n).strip() for n in (vcfg.get("trackers") or []) if str(n).strip()] or None
    if names and drone_name and drone_name not in names:
        names = names + [drone_name]                              # 只讀指定 tracker 時飛機仍要讀
    samples = read_trackers(server, seconds=float(vcfg.get("seconds", 2.0)), names=names,
                            exclude=skip_read, log=log, connect_timeout=connect_timeout)
    drone: Optional[TrackerSample] = None
    boxes: Dict[str, TrackerSample] = {}
    for key, s in samples.items():
        if drone_name and s.name == drone_name:
            if drone is None or s.count > drone.count:
                drone = s
            continue
        if s.name in skip_all:
            continue
        boxes[key] = s
    return drone, boxes


def sample_to_enu(sample: TrackerSample, vcfg: Dict) -> Tuple[float, float, float]:
    """rigid body 樣本 -> 房間 ENU 座標 (含 axes 轉換與 offset 平移)。"""
    M = axes_matrix(vcfg.get("axes", "y_up"))
    off = np.asarray(vcfg.get("offset", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    p = to_enu(sample.pos, M) + off
    return float(p[0]), float(p[1]), float(p[2])


def trackers_to_items(samples: Dict[str, TrackerSample], vcfg: Dict) -> List[Dict]:
    """VRPN 樣本 -> config obstacles.items 格式 (source: vrpn)。

    vcfg (config obstacles.vrpn)：
      axes        y_up (Motive 預設) | z_up | 'x,-z,y'
      mode        each_box = 每個 rigid body 一個箱子 (尺寸見下, 偏航取自姿態)
                  hull     = 所有 rigid body 位置合成一個障礙物 (XY 凸包; 例如每個角放一組光球)
      box_size    箱子三邊 [a, b, c] (m)。orientation=fixed 時就是 [長 x, 寬 y, 高 z]
      orientation auto_square (預設) | auto_long_x | auto_long_y = 同一種箱子有躺有立, 用量到的頂面高度判斷
                  哪一邊垂直 (pivot 需為 top 或 center), 底面取另外兩邊 (見 box_dims_from_height);
                  fixed = 一律照 box_size 的 x y z 順序
      sizes       {rigid body 名稱: [sx, sy, sz]} 個別指定尺寸 (優先於自動判斷)
      height_tol  自動判斷時頂面高度與邊長的容許差 (m, 預設 0.08; 光球中心略高於箱面)
      pivot       rigid body 樞紐點在箱子的 top (光球貼頂面, 預設) | center | bottom
      offset      [dx, dy, dz] 讀進來的座標再加上此平移 (Motive 原點與飛控 EKF 原點不同時用)
    """
    M = axes_matrix(vcfg.get("axes", "y_up"))
    mode = str(vcfg.get("mode", "each_box")).strip().lower()
    off = np.asarray(vcfg.get("offset", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0], dtype=float).reshape(3)
    size = [float(v) for v in (vcfg.get("box_size") or [0.5, 0.5, 0.5])]
    pivot = str(vcfg.get("pivot", "top")).strip().lower()
    orientation = str(vcfg.get("orientation", "auto_square")).strip().lower()
    overrides = vcfg.get("sizes") or {}
    height_tol = float(vcfg.get("height_tol", 0.08))
    items: List[Dict] = []
    if not samples:
        return items
    if mode == "hull":
        pts = [(to_enu(s.pos, M) + off).tolist() for s in samples.values()]
        items.append({
            "name": str(vcfg.get("hull_name", "vrpn_hull")), "kind": "points",
            "points": [[round(v, 4) for v in p] for p in pts],
            "z_bottom": 0.0, "source": "vrpn",
            "tracker": ",".join(sorted(samples)),
        })
        return items
    for key in sorted(samples):
        s = samples[key]
        p = to_enu(s.pos, M) + off
        z = float(p[2])
        if key in overrides or s.name in overrides:
            sx, sy, sz = (float(v) for v in overrides.get(key, overrides.get(s.name)))
            note = f"尺寸: 個別指定 {sx:.2f}×{sy:.2f}×{sz:.2f}"
        elif orientation.startswith("auto") and pivot in ("top", "center"):
            h_meas = z if pivot == "top" else 2.0 * z
            sx, sy, sz, note = box_dims_from_height(size, h_meas, orientation, height_tol)
        else:
            sx, sy, sz = size
            note = f"尺寸: 固定 x y z = {sx:.2f}×{sy:.2f}×{sz:.2f}"
        if pivot == "bottom":
            zb = z
        elif pivot == "center":
            zb = z - sz / 2.0
        else:                                   # top
            zb = z - sz
        zt = zb + sz
        zb = max(zb, 0.0)                       # 箱底不會低於地板: 保留量到的頂面高度, 高度縮短
        items.append({
            "name": key, "kind": "box",
            "x": round(float(p[0]), 4), "y": round(float(p[1]), 4),
            "z_bottom": round(zb, 4),
            "size": [round(sx, 4), round(sy, 4), round(max(zt - zb, 0.01), 4)],
            "yaw_deg": round(math.degrees(yaw_enu(s.quat, M)), 2),
            "source": "vrpn", "tracker": key,
            "vrpn_z": round(z, 4),
            "note": note,
        })
    return items
