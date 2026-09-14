"""
右側參數面板。把控制項的值收集成 overrides，套用到 cfg。
"""

from __future__ import annotations

from typing import Dict

from PyQt6 import QtWidgets

from core.config import deep_update


def _dspin(lo, hi, val, step=0.1, decimals=2):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(val)
    return w


class ParamPanel(QtWidgets.QScrollArea):
    def __init__(self, cfg: Dict, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        self.setWidget(host)
        form = QtWidgets.QVBoxLayout(host)
        form.setSpacing(8)

        # ---- 飛行空間 ----
        g = QtWidgets.QGroupBox("飛行空間 (m)")
        gl = QtWidgets.QFormLayout(g)
        self.size_x = _dspin(0.5, 50, cfg["volume"]["size_x"], 0.5)
        self.size_y = _dspin(0.5, 50, cfg["volume"]["size_y"], 0.5)
        self.size_z = _dspin(0.5, 20, cfg["volume"]["size_z"], 0.5)
        self.m_wall = _dspin(0.0, 5, cfg["margin"]["wall"], 0.1)
        self.m_floor = _dspin(0.0, 5, cfg["margin"]["floor"], 0.1)
        self.m_ceil = _dspin(0.0, 5, cfg["margin"]["ceiling"], 0.1)
        gl.addRow("長 X", self.size_x)
        gl.addRow("寬 Y", self.size_y)
        gl.addRow("高 Z", self.size_z)
        gl.addRow("離牆", self.m_wall)
        gl.addRow("離地", self.m_floor)
        gl.addRow("離頂", self.m_ceil)
        form.addWidget(g)

        # ---- 飛行 ----
        g = QtWidgets.QGroupBox("飛行")
        gl = QtWidgets.QFormLayout(g)
        self.speed = _dspin(0.05, 5, cfg["flight"]["cruise_speed"], 0.05)
        self.max_speed = _dspin(0.1, 10, cfg["flight"]["max_speed"], 0.1)
        self.target = _dspin(10, 1200, cfg["flight"]["target_duration"], 10, 0)
        self.laps_auto = QtWidgets.QCheckBox("自動圈數")
        self.laps_auto.setChecked(str(cfg["flight"].get("laps", "auto")) == "auto")
        self.laps = QtWidgets.QSpinBox()
        self.laps.setRange(1, 999)
        self.laps.setValue(int(cfg["flight"]["laps"]) if str(cfg["flight"].get("laps")).isdigit() else 8)
        self.laps.setEnabled(not self.laps_auto.isChecked())
        self.laps_auto.toggled.connect(lambda c: self.laps.setEnabled(not c))
        self.duration_basis = QtWidgets.QComboBox()
        self.duration_basis.addItems(["auto", "guided"])
        self.duration_basis.setCurrentText(
            "guided" if str(cfg["flight"].get("duration_basis", "auto")).lower() == "guided" else "auto")
        self.duration_basis.setToolTip(
            "圈數與 200~300 s 工時判定以什麼為準：\n"
            "auto = AUTO 任務 (.waypoints 上傳) 的航線時間 (含進場, 不含起飛/降落) —— 只飛 AUTO 用這個\n"
            "guided = GUIDED 串流軌跡時間 (舊行為); AUTO 通常比它長 5~25 s")
        gl.addRow("巡航速度 m/s", self.speed)
        gl.addRow("速度上限 m/s", self.max_speed)
        gl.addRow("目標工時 s", self.target)
        gl.addRow("工時基準", self.duration_basis)
        gl.addRow("", self.laps_auto)
        gl.addRow("手動圈數", self.laps)
        form.addWidget(g)

        # ---- 高度調變 ----
        g = QtWidgets.QGroupBox("垂直高度調變 (3D)")
        gl = QtWidgets.QFormLayout(g)
        self.alt_mode = QtWidgets.QComboBox()
        self.alt_mode.addItems(["sine", "triangle", "smooth_random", "stair"])
        self.alt_mode.setCurrentText(cfg["altitude"].get("mode", "sine"))
        self.alt_cycles = _dspin(0, 30, float(cfg["altitude"].get("cycles", 6)), 1, 0)
        self.alt_amp_auto = QtWidgets.QCheckBox("自動振幅")
        self.alt_amp_auto.setChecked(cfg["altitude"].get("amplitude", "auto") == "auto")
        self.alt_amp = _dspin(0, 10, 0.8, 0.1)
        if isinstance(cfg["altitude"].get("amplitude"), (int, float)):
            self.alt_amp.setValue(float(cfg["altitude"]["amplitude"]))
        self.alt_amp.setEnabled(not self.alt_amp_auto.isChecked())
        self.alt_amp_auto.toggled.connect(lambda c: self.alt_amp.setEnabled(not c))
        # stair (階梯狀升降) 專用: 階數 + 階間升降速度 (auto = 依 flight.speed_up/down)
        self.alt_steps = QtWidgets.QSpinBox()
        self.alt_steps.setRange(1, 20)
        self.alt_steps.setValue(int(cfg["altitude"].get("steps", 4)))
        self.alt_steps.setToolTip("由最低到最高分幾階 (每階高 = 2×振幅/階數; 1 = 方波)")
        self.alt_ramp_auto = QtWidgets.QCheckBox("自動升降速度")
        self.alt_ramp_auto.setToolTip(
            "auto = 80% × min(speed_up/speed_down, √(max_speed²−巡航速度²))，\n"
            "保證不觸發垂直速度警告、升降時 3D 合成速度也不超過 max_speed")
        ramp_cfg = cfg["altitude"].get("ramp_speed", "auto")
        self.alt_ramp_auto.setChecked(not isinstance(ramp_cfg, (int, float)))
        self.alt_ramp = _dspin(0.05, 5, 0.5, 0.05)
        self.alt_ramp.setToolTip("階與階之間升降的垂直速度 (m/s); 超過 speed_up/down 會被警告")
        if isinstance(ramp_cfg, (int, float)):
            self.alt_ramp.setValue(float(ramp_cfg))
        self.alt_mode.currentTextChanged.connect(lambda _: self._sync_stair_widgets())
        self.alt_ramp_auto.toggled.connect(lambda _: self._sync_stair_widgets())
        self._sync_stair_widgets()
        gl.addRow("模式", self.alt_mode)
        gl.addRow("起伏次數", self.alt_cycles)
        gl.addRow("", self.alt_amp_auto)
        gl.addRow("振幅 m", self.alt_amp)
        gl.addRow("階數 (stair)", self.alt_steps)
        gl.addRow("", self.alt_ramp_auto)
        gl.addRow("升降速度 m/s", self.alt_ramp)
        form.addWidget(g)

        # ---- 隨機手飛 (random_walk pattern) ----
        g = QtWidgets.QGroupBox("隨機手飛路徑 (Random walk)")
        gl = QtWidgets.QFormLayout(g)
        rw = (cfg.get("patterns", {}) or {}).get("random_walk", {}) or {}
        self.rw_seed = QtWidgets.QSpinBox()
        self.rw_seed.setRange(0, 99999)
        self.rw_seed.setValue(int(rw.get("seed", 0)))
        self.rw_seed.setToolTip("亂數種子：換一個數字就換一條路徑 (同 seed 永遠同一條)")
        n_cfg = rw.get("n_points", "auto")
        n_auto = not isinstance(n_cfg, (int, float))
        self.rw_auto = QtWidgets.QCheckBox("自動長度 (整段一條不重複, 圈數 1)")
        self.rw_auto.setChecked(n_auto)
        self.rw_auto.setToolTip("依目標工時 × 巡航速度自動決定轉折點數，整段飛行就是一條路徑、不用 DO_JUMP；\n"
                                "取消 = 手動指定轉折點數 (路徑短時圈數會 >1，仍全展開、各圈高度錯開)")
        self.rw_points = QtWidgets.QSpinBox()
        self.rw_points.setRange(4, 200)
        self.rw_points.setValue(int(n_cfg) if not n_auto else 12)
        self.rw_points.setToolTip("轉折點數：越多路徑越長、越曲折")
        self.rw_points.setEnabled(not n_auto)
        self.rw_auto.toggled.connect(lambda c: self.rw_points.setEnabled(not c))
        self.rw_turn = _dspin(5, 150, float(rw.get("turn_sigma_deg", 60.0)), 5, 0)
        self.rw_turn.setToolTip("每步轉向角標準差 (deg)：越大越亂、越小越像平滑巡航")
        gl.addRow("seed", self.rw_seed)
        gl.addRow("", self.rw_auto)
        gl.addRow("轉折點數", self.rw_points)
        gl.addRow("轉向幅度 deg", self.rw_turn)
        form.addWidget(g)

        # ---- 朝向 ----
        g = QtWidgets.QGroupBox("機頭朝向")
        gl = QtWidgets.QFormLayout(g)
        self.yaw_mode = QtWidgets.QComboBox()
        self.yaw_mode.addItems(["travel", "fixed", "spin"])
        self.yaw_mode.setCurrentText(cfg["yaw"].get("mode", "travel"))
        gl.addRow("模式", self.yaw_mode)
        form.addWidget(g)

        # ---- 匯出 (.waypoints) ----
        g = QtWidgets.QGroupBox("匯出 .waypoints")
        gl = QtWidgets.QFormLayout(g)
        self.origin_lat = _dspin(-90, 90, cfg["waypoints"]["origin_lat"], 0.0001, 8)
        self.origin_lon = _dspin(-180, 180, cfg["waypoints"]["origin_lon"], 0.0001, 8)
        self.spacing = _dspin(0.05, 5, cfg["waypoints"]["point_spacing"], 0.05)
        self.use_spline = QtWidgets.QCheckBox("曲線用 SPLINE")
        self.use_spline.setChecked(bool(cfg["waypoints"].get("use_spline", True)))
        self.sparse_straight = QtWidgets.QCheckBox("直線段精簡航點 (轉角 + 高度容差)")
        self.sparse_straight.setChecked(bool(cfg["waypoints"].get("sparse_straight", True)))
        self.sparse_straight.setToolTip(
            "往返/矩形/鋸齒：以精確轉角為航點，段內只要折線與高度剖面的差 ≤ 高度容差就拿掉中間點\n"
            "(振幅 0 → 往返 = A,B,A,B… 兩點一直線、矩形 = 每圈 4 角；線性升降只留折點)。\n"
            "段越長，飛控 S-curve 每段能跑到的速度越高 (0.25 m 的段跑不到 0.5 m/s)。\n"
            "取消 = 一律沿弧長依「航點間距」密集取樣 (舊行為)")
        self.z_tol = _dspin(0.0, 0.5, float(cfg["waypoints"].get("z_tol", 0.02)), 0.01)
        self.z_tol.setToolTip("直線段航點精簡的高度容差 (m)；越大航點越少、段越長 (速度越跑得滿)，0 = 不精簡")
        self.do_jump = QtWidgets.QCheckBox("用 DO_JUMP 重複圈數 (只寫一圈航點)")
        self.do_jump.setChecked(str(cfg["waypoints"].get("repeat", "do_jump")).lower() != "unroll")
        self.do_jump.setToolTip(
            "只寫一圈的航點 + DO_JUMP(回第一點, 重複 圈數−1 次) + 收尾點：任務項數與圈數無關。\n"
            "每圈高度必須相同 → 起伏次數自動調成每圈整數次 (統計欄會顯示)。\n"
            "取消 = 全部圈數展開寫出 (各圈高度相位可錯開，但航點數 × 圈數)")
        tip = cfg["waypoints"].get("turn_in_place") or {}
        tip = tip if isinstance(tip, dict) else {"enabled": bool(tip)}
        self.turn_enable = QtWidgets.QCheckBox("到航點停下 → 原地轉頭 → 再前進 (CONDITION_YAW)")
        self.turn_enable.setChecked(bool(tip.get("enabled", False)))
        self.turn_enable.setToolTip(
            "轉角航點後面多寫 CONDITION_YAW(朝下一段) + NAV_DELAY 兩項：\n"
            "飛機在該航點完全停下 → 原地轉頭朝下一段 → NAV_DELAY 秒後才前進 (DO_JUMP 每圈都轉)；\n"
            "起飛到高度後也會先原地轉向第一段。只影響 AUTO 任務 (GUIDED / CSV 不變)。\n"
            "停頓會拉長 AUTO 航線時間 (自動圈數會跟著減少)。飛控請維持 WP_YAW_BEHAVIOR = 2 (預設)")
        self.turn_min_deg = _dspin(5, 180, float(tip.get("min_turn_deg", 30)), 5, 0)
        self.turn_min_deg.setToolTip("水平轉向角 ≥ 此值的航點才停下轉頭；小角度照常邊飛邊轉")
        self.turn_rate = _dspin(1, 180, float(tip.get("rate_deg_s", 45)), 5, 0)
        self.turn_rate.setToolTip("CONDITION_YAW 轉頭角速度 (deg/s)；飛控另受 ATC_SLEW_YAW 上限 (預設 60 deg/s)")
        self.turn_settle = _dspin(0, 10, float(tip.get("settle_s", 1.0)), 0.5, 1)
        self.turn_settle.setToolTip("轉完再多停的秒數；NAV_DELAY = 要轉的角度 / 角速度 + 此值")
        self.turn_curves = QtWidgets.QCheckBox("曲線 pattern 的航點也轉 (圓 / 8字 / 隨機手飛)")
        self.turn_curves.setChecked(bool(tip.get("curves", False)))
        self.turn_enable.toggled.connect(lambda _: self._sync_turn_widgets())
        self._sync_turn_widgets()
        self.takeoff_alt = _dspin(0.2, 10, cfg["waypoints"]["takeoff_alt"], 0.1)
        self.end_action = QtWidgets.QComboBox()
        self.end_action.addItems(["land", "rtl"])
        self.end_action.setCurrentText(cfg["waypoints"].get("end_action", "land"))
        gl.addRow("原點緯度", self.origin_lat)
        gl.addRow("原點經度", self.origin_lon)
        gl.addRow("航點間距 m", self.spacing)
        gl.addRow("", self.use_spline)
        gl.addRow("", self.sparse_straight)
        gl.addRow("高度容差 m", self.z_tol)
        gl.addRow("", self.do_jump)
        gl.addRow("", self.turn_enable)
        gl.addRow("轉頭門檻 deg", self.turn_min_deg)
        gl.addRow("轉頭角速度 deg/s", self.turn_rate)
        gl.addRow("轉完多停 s", self.turn_settle)
        gl.addRow("", self.turn_curves)
        gl.addRow("起飛高度 m", self.takeoff_alt)
        gl.addRow("結束動作", self.end_action)
        form.addWidget(g)

        # ---- GUIDED ----
        g = QtWidgets.QGroupBox("GUIDED 即時串流")
        gl = QtWidgets.QFormLayout(g)
        self.connection = QtWidgets.QLineEdit(cfg["guided"]["connection"])
        self.g_takeoff = _dspin(0.2, 10, cfg["guided"]["takeoff_alt"], 0.1)
        gl.addRow("連線", self.connection)
        gl.addRow("起飛高度 m", self.g_takeoff)
        form.addWidget(g)

        form.addStretch(1)

    def _sync_stair_widgets(self):
        """stair 專用控制項只在模式為 stair 時可用。"""
        is_stair = self.alt_mode.currentText() == "stair"
        self.alt_steps.setEnabled(is_stair)
        self.alt_ramp_auto.setEnabled(is_stair)
        self.alt_ramp.setEnabled(is_stair and not self.alt_ramp_auto.isChecked())

    def _sync_turn_widgets(self):
        """原地轉頭的細項只在功能開啟時可用。"""
        on = self.turn_enable.isChecked()
        for w in (self.turn_min_deg, self.turn_rate, self.turn_settle, self.turn_curves):
            w.setEnabled(on)

    def apply_to(self, base_cfg: Dict) -> Dict:
        """把面板值套到 base_cfg，回傳新的 cfg。"""
        laps = "auto" if self.laps_auto.isChecked() else int(self.laps.value())
        amp = "auto" if self.alt_amp_auto.isChecked() else float(self.alt_amp.value())
        ramp = "auto" if self.alt_ramp_auto.isChecked() else float(self.alt_ramp.value())
        overrides = {
            "volume": {
                "size_x": self.size_x.value(),
                "size_y": self.size_y.value(),
                "size_z": self.size_z.value(),
            },
            "margin": {
                "wall": self.m_wall.value(),
                "floor": self.m_floor.value(),
                "ceiling": self.m_ceil.value(),
            },
            "flight": {
                "cruise_speed": self.speed.value(),
                "max_speed": self.max_speed.value(),
                "target_duration": self.target.value(),
                "duration_basis": self.duration_basis.currentText(),
                "laps": laps,
            },
            "altitude": {
                "mode": self.alt_mode.currentText(),
                "cycles": self.alt_cycles.value(),
                "amplitude": amp,
                "steps": int(self.alt_steps.value()),
                "ramp_speed": ramp,
            },
            "yaw": {"mode": self.yaw_mode.currentText()},
            "patterns": {
                "random_walk": {
                    "seed": int(self.rw_seed.value()),
                    "n_points": "auto" if self.rw_auto.isChecked() else int(self.rw_points.value()),
                    "turn_sigma_deg": float(self.rw_turn.value()),
                },
            },
            "waypoints": {
                "origin_lat": self.origin_lat.value(),
                "origin_lon": self.origin_lon.value(),
                "point_spacing": self.spacing.value(),
                "use_spline": self.use_spline.isChecked(),
                "sparse_straight": self.sparse_straight.isChecked(),
                "z_tol": float(self.z_tol.value()),
                "repeat": "do_jump" if self.do_jump.isChecked() else "unroll",
                "turn_in_place": {
                    "enabled": self.turn_enable.isChecked(),
                    "min_turn_deg": float(self.turn_min_deg.value()),
                    "rate_deg_s": float(self.turn_rate.value()),
                    "settle_s": float(self.turn_settle.value()),
                    "curves": self.turn_curves.isChecked(),
                },
                "takeoff_alt": self.takeoff_alt.value(),
                "end_action": self.end_action.currentText(),
            },
            "guided": {
                "connection": self.connection.text().strip(),
                "takeoff_alt": self.g_takeoff.value(),
            },
        }
        return deep_update(base_cfg, overrides)
