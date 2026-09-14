"""
GUIDED 即時位置串流 (pymavlink)。

流程：連線 -> 等心跳 -> GUIDED -> arm -> takeoff -> 平滑接近軌跡起點 ->
      以軌跡時間戳串流 SET_POSITION_TARGET_LOCAL_NED -> LAND。

座標假設：飛控的 EKF 原點 == 房間 (mocap) 原點。則本地 ENU(x東,y北,z上) 直接對應
          MAV_FRAME_LOCAL_NED 的 (north=y, east=x, down=-z)。室內定位由你既有的
          mocap->飛控 橋接提供 (本程式不負責)。

安全：這會驅動真實/SITL 飛機。run() 內含安全檢查，confirm 必須為 True 才會 arm；
      Ctrl+C 會切 LAND。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

import numpy as np

from core.geometry import enu_to_ned
from core.planner import PlanResult

# SET_POSITION_TARGET_LOCAL_NED type_mask 位元 (設 1 = 忽略)
_IGN_VX, _IGN_VY, _IGN_VZ = 8, 16, 32
_IGN_AX, _IGN_AY, _IGN_AZ = 64, 128, 256
_IGN_YAW, _IGN_YAWRATE = 1024, 2048
_FRAME_LOCAL_NED = 1


def _enu_yaw_to_ned_heading(yaw_enu: float) -> float:
    """ENU yaw (0=東,CCW) -> NED heading (0=北,CW)。"""
    return math.pi / 2.0 - yaw_enu


class GuidedRunner:
    def __init__(self, connection: str, log: Optional[Callable[[str], None]] = None):
        self.connection = connection
        self.log = log or print
        self.master = None
        self._abort = False

    # ---- 連線 / 模式 ----
    def connect(self, timeout: float = 30.0):
        from pymavlink import mavutil

        self.log(f"連線中: {self.connection}")
        self.master = mavutil.mavlink_connection(self.connection)
        self.log("等待心跳...")
        self.master.wait_heartbeat(timeout=timeout)
        self.log(f"已連線 (sys={self.master.target_system}, comp={self.master.target_component})")

    def _set_mode(self, mode: str):
        mapping = self.master.mode_mapping() or {}
        if mode not in mapping:
            raise RuntimeError(f"韌體不支援模式 {mode} (可用: {sorted(mapping)})")
        self.master.set_mode(mapping[mode])
        self.log(f"模式 -> {mode}")

    def _arm(self):
        self.master.arducopter_arm()
        self.log("解鎖中 (arming)...")
        self.master.motors_armed_wait()
        self.log("已解鎖 (armed)")

    def _takeoff(self, alt: float):
        from pymavlink import mavutil

        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt,
        )
        self.log(f"起飛 -> {alt:.2f} m")
        # 等到約達目標高度
        t0 = time.time()
        while time.time() - t0 < 30:
            msg = self.master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=2)
            if msg and -msg.z >= alt * 0.95:
                self.log("到達起飛高度")
                return
        self.log("起飛逾時，仍繼續 (請留意)")

    def _current_ned(self):
        """目前位置 (NED)。收不到位置就中止 —— 絕不可猜成原點。

        猜 (0,0,0) 會讓「接近起點」從房間原點的地板高度規劃並串流, 而飛機其實在別處且已在空中:
        繞障路線是為錯誤的起點算的, 可能直接穿過箱子。run() 的 finally 仍會切 LAND, GUI 也會把
        例外顯示出來, 所以中止是安全的收場。這裡不能拿 waypoints.takeoff_point 頂替 ——
        那是規劃時量的水平位置, 起飛後飛機可能已飄移, 而且沒有高度資訊。
        """
        msg = self.master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=3)
        if not msg:
            raise RuntimeError(
                "收不到 LOCAL_POSITION_NED，無法確定飛機目前位置；中止接近 "
                "(請檢查 mocap→飛控 橋接與 SRx_POSITION 串流率)"
            )
        return msg.x, msg.y, msg.z

    def _send_setpoint(self, north, east, down, vx, vy, vz, yaw, mask, t_ms):
        self.master.mav.set_position_target_local_ned_send(
            t_ms, self.master.target_system, self.master.target_component,
            _FRAME_LOCAL_NED, mask,
            north, east, down, vx, vy, vz, 0, 0, 0, yaw, 0,
        )

    # ---- 主流程的小工具 (各自單一職責) ----
    def _log_report(self, report) -> bool:
        """輸出安全報告；回傳是否通過 (errors 為空才可飛)。"""
        if not report.ok:
            self.log("安全檢查未通過，已中止：")
            for m in report.errors:
                self.log("  [ERROR] " + m)
            return False
        for m in report.warnings:
            self.log("  [WARN] " + m)
        return True

    @staticmethod
    def _build_type_mask(send_vel: bool, send_yaw: bool) -> int:
        """組 SET_POSITION_TARGET_LOCAL_NED 的 type_mask (設 1 = 忽略該欄)。"""
        mask = _IGN_AX | _IGN_AY | _IGN_AZ | _IGN_YAWRATE
        if not send_vel:
            mask |= _IGN_VX | _IGN_VY | _IGN_VZ
        if not send_yaw:
            mask |= _IGN_YAW
        return mask

    @staticmethod
    def _traj_ned_arrays(t):
        """把軌跡的 ENU 位置/速度/yaw 轉成 NED 串流陣列。

        回傳 (north, east, down, vN, vE, vD, head)。位置與速度共用 geometry.enu_to_ned
        的同一套軸交換 (north=y, east=x, down=-z)，避免在此重寫；head 由 ENU yaw 轉 NED heading。
        """
        north, east, down = enu_to_ned(t.x, t.y, t.z)
        vN, vE, vD = enu_to_ned(t.vx, t.vy, t.vz)
        head = np.array([_enu_yaw_to_ned_heading(y) for y in t.yaw])
        return north, east, down, vN, vE, vD, head

    def _stream(self, t, north, east, down, vN, vE, vD, head, mask, send_vel, send_yaw):
        """依軌跡時間戳即時送出位置設點 (以單調時鐘對齊各樣本)。"""
        start = time.monotonic()
        for i in range(t.n):
            if self._abort:
                break
            # 對齊到該樣本的時間戳
            target_t = float(t.t[i])
            while True:
                now = time.monotonic() - start
                if now >= target_t or self._abort:
                    break
                time.sleep(min(0.005, target_t - now))
            self._send_setpoint(
                north[i], east[i], down[i],
                (vN[i] if send_vel else 0.0),
                (vE[i] if send_vel else 0.0),
                (vD[i] if send_vel else 0.0),
                (head[i] if send_yaw else 0.0),
                mask, int((time.monotonic() - start) * 1000) & 0xFFFFFFFF,
            )

    # ---- 主流程 (薄編排層) ----
    def run(self, plan: PlanResult, cfg: Dict, confirm: bool = False) -> bool:
        """執行整段飛行。confirm 必須為 True 才會真的解鎖起飛。"""
        if not self._log_report(plan.report):
            return False

        g = cfg["guided"]
        send_yaw = bool(g.get("send_yaw", True))
        send_vel = bool(g.get("send_velocity", True))
        takeoff_alt = float(g.get("takeoff_alt", 1.0))

        mask = self._build_type_mask(send_vel, send_yaw)
        t = plan.trajectory
        north, east, down, vN, vE, vD, head = self._traj_ned_arrays(t)

        if not confirm:
            self.log("乾跑模式 (confirm=False)：不解鎖、不起飛。檢查無誤後再以 confirm=True 執行。")
            self.log(f"將飛 {plan.pattern.display_name}, 工時 {t.duration:.0f}s, "
                     f"{t.n} 個設點, 起飛高度 {takeoff_alt} m")
            try:  # 順便測一下連線 (失敗不致命)
                if self.master is None:
                    self.connect(timeout=8.0)
                self.log("連線測試成功。")
            except Exception as e:  # noqa: BLE001
                self.log(f"連線測試失敗 (乾跑仍視為通過): {e}")
            return True

        try:
            if self.master is None:
                self.connect()
            self._set_mode("GUIDED")
            self._arm()
            self._takeoff(takeoff_alt)

            # 平滑接近軌跡起點 (有障礙物時沿可視圖最短路水平繞過去, 不越過)
            self._approach_start(north[0], east[0], down[0], head[0], mask, send_vel, send_yaw,
                                 field=getattr(plan, "obstacles", None))

            # 依時間戳串流
            self.log(f"開始串流軌跡：{plan.pattern.display_name} ({t.duration:.0f}s)")
            self._stream(t, north, east, down, vN, vE, vD, head, mask, send_vel, send_yaw)
            self.log("軌跡完成，降落中...")
        except KeyboardInterrupt:
            self.log("使用者中斷 -> LAND")
        finally:
            try:
                self._set_mode("LAND")
            except Exception as e:  # noqa: BLE001
                self.log(f"切 LAND 失敗: {e}")
        return True

    def _approach_start(self, n0, e0, d0, yaw0, mask, send_vel, send_yaw,
                        approach_speed: float = 0.4, field=None):
        """從目前位置平滑移動到軌跡起點。

        field (core.obstacles.ObstacleField) 給了且啟用時, 先用可視圖算「目前位置 -> 起點」不穿禁區的
        水平折線, 逐段直線飛 (高度沿途線性變到起點高度); 沒有障礙物或直線可達就是一段直線。
        """
        cn, ce, cd = self._current_ned()
        legs = [(n0, e0, d0)]
        if field is not None:
            field = getattr(field, "approach_field", field)   # 接近起點高度低, 連可越過的低矮箱子也繞
        if field is not None and getattr(field, "active", False):
            # NED (north=y, east=x) -> 房間 ENU xy
            path = field.free_path((ce, cn), (e0, n0))
            if path is None:
                self.log("[WARN] 目前位置到起點被障礙物擋住, 找不到繞行路徑; 直線接近 (請留意!)")
            elif len(path) > 2:
                pts = np.asarray(path, dtype=float)
                seg = np.hypot(*np.diff(pts, axis=0).T)
                s = np.concatenate([[0.0], np.cumsum(seg)])
                frac = s / max(float(s[-1]), 1e-9)
                legs = [(float(p[1]), float(p[0]), cd + (d0 - cd) * float(f))
                        for p, f in zip(pts[1:], frac[1:])]
                self.log(f"接近起點: 繞開障礙物, 共 {len(legs)} 段")
        start = time.monotonic()
        for (tn, te, td) in legs:
            dist = math.sqrt((tn - cn) ** 2 + (te - ce) ** 2 + (td - cd) ** 2)
            dur = max(2.0, dist / max(approach_speed, 0.1))
            self.log(f"接近 ({te:.2f}, {tn:.2f}) (距離 {dist:.2f} m, 約 {dur:.1f}s)")
            steps = max(2, int(dur * 10))
            for k in range(steps + 1):
                if self._abort:
                    return
                a = k / steps
                self._send_setpoint(
                    cn + a * (tn - cn), ce + a * (te - ce), cd + a * (td - cd),
                    0.0, 0.0, 0.0, (yaw0 if send_yaw else 0.0),
                    mask, int((time.monotonic() - start) * 1000) & 0xFFFFFFFF,
                )
                time.sleep(0.1)
            cn, ce, cd = tn, te, td

    def abort(self):
        self._abort = True
