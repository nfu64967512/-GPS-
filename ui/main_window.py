"""
主視窗：左 pattern 清單 / 中 3D 預覽 / 右 參數面板 / 下 狀態與安全訊息。
"""

from __future__ import annotations

import os
import sys

from PyQt6 import QtCore, QtGui, QtWidgets

from core import patterns
from core.config import get, load_config, save_config
from core.geometry import describe_takeoff_point, takeoff_is_origin
from core.planner import duration_basis, plan as plan_pattern
from .obstacle_panel import ObstaclePanel
from .param_panel import ParamPanel
from .preview3d import Preview3D

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_OUTPUT = os.path.join(_ROOT, "output")


class FlyWorker(QtCore.QThread):
    """背景執行 GUIDED 串流，避免卡住 UI。"""

    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(bool)

    def __init__(self, plan, cfg, confirm):
        super().__init__()
        self._plan, self._cfg, self._confirm = plan, cfg, confirm

    def run(self):
        try:
            from runner import GuidedRunner
            runner = GuidedRunner(self._cfg["guided"]["connection"],
                                  log=lambda m: self.log.emit(m))
            ok = runner.run(self._plan, self._cfg, confirm=self._confirm)
            self.done.emit(bool(ok))
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"[ERROR] 飛行執行緒例外: {e}")
            self.done.emit(False)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indoor Trajectory Studio — 室內 3D 軌跡規劃")
        self.resize(1280, 820)
        self.base_cfg = load_config()
        self.current_plan = None
        self._worker = None

        self._build_ui()
        self._select_first()

    # ---- UI 組裝 ----
    def _build_ui(self):
        self._build_toolbar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 左：pattern 清單
        self.pattern_list = QtWidgets.QListWidget()
        self.pattern_list.setMaximumWidth(220)
        for key, name in patterns.list_patterns():
            it = QtWidgets.QListWidgetItem(name)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            self.pattern_list.addItem(it)
        self.pattern_list.currentRowChanged.connect(lambda _: self.generate())
        root.addWidget(self.pattern_list)

        # 中：預覽 + 狀態
        mid = QtWidgets.QWidget()
        midl = QtWidgets.QVBoxLayout(mid)
        midl.setContentsMargins(0, 0, 0, 0)
        self.preview = Preview3D(dpi=int(get(self.base_cfg, "ui.preview_dpi", 130)))
        midl.addWidget(self.preview, 1)
        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(150)
        self.stats.setObjectName("stats")
        midl.addWidget(self.stats)
        root.addWidget(mid, 1)

        # 右：參數 / 障礙物 (分頁)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(360)
        self.tabs.setMaximumWidth(440)
        root.addWidget(self.tabs)
        self._build_panels(self.base_cfg)

        self.statusBar().showMessage("就緒")
        self._apply_style()

    def _build_panels(self, cfg):
        """(重) 建右側的參數面板與障礙物面板。

        先建好新的再拆舊的 —— 設定檔壞掉 (例如 waypoints.takeoff_point 格式錯) 會讓面板建構丟例外,
        若先拆再建就會留下沒有面板的視窗, 而 Qt 在 slot 裡吃到例外會直接殺掉整個程式。
        """
        panel = ParamPanel(cfg)
        obs_panel = ObstaclePanel(cfg)
        for i in range(self.tabs.count()):
            self.tabs.widget(i).deleteLater()
        self.tabs.clear()
        self.panel, self.obs_panel = panel, obs_panel
        self.obs_panel.changed.connect(self.generate)
        self.tabs.addTab(self.panel, "參數")
        self.tabs.addTab(self.obs_panel, "障礙物")

    def _build_toolbar(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)

        def act(text, slot):
            a = QtGui.QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act("產生 / 預覽", self.generate)
        tb.addSeparator()
        act("匯出 .waypoints", lambda: self.export("waypoints"))
        act("匯出 CSV", lambda: self.export("csv"))
        act("匯出 PNG", lambda: self.export("png"))
        tb.addSeparator()
        act("連線並飛行 (GUIDED)", self.fly)
        tb.addSeparator()
        act("載入設定", self.load_cfg)
        act("儲存設定", self.save_cfg)

    def _apply_style(self):
        qss = os.path.join(_THIS, "style.qss")
        if os.path.exists(qss):
            with open(qss, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def _select_first(self):
        if self.pattern_list.count():
            self.pattern_list.setCurrentRow(0)

    # ---- 動作 ----
    def _selected_key(self):
        it = self.pattern_list.currentItem()
        return it.data(QtCore.Qt.ItemDataRole.UserRole) if it else None

    def _cfg(self):
        return self.obs_panel.apply_to(self.panel.apply_to(self.base_cfg))

    def generate(self):
        key = self._selected_key()
        if not key:
            return
        try:
            cfg = self._cfg()
            self.current_plan = plan_pattern(key, cfg)
            self.preview.show_plan(self.current_plan, cfg)
            self._update_stats(self.current_plan)
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"產生失敗: {e}")
            self.stats.setPlainText(f"[ERROR] {e}")

    def _update_stats(self, pr):
        t = pr.trajectory
        b = t.bounds()
        cfg = self._cfg()
        from io_export.waypoints import DEFAULT_FC_BUDGET, trajectory_to_waypoints
        pl, _ = trajectory_to_waypoints(pr, cfg, mode="precision")
        cl, _ = trajectory_to_waypoints(pr, cfg, mode="compact")
        n_prec = len(pl) - 1
        n_comp = len(cl) - 1
        budget = int(cfg["waypoints"].get("fc_budget", DEFAULT_FC_BUDGET))
        comp_tag = "<=預算" if n_comp <= budget else "仍超過!"
        m = t.meta
        f = cfg["flight"]
        req = (f"(需求 ≥{float(f.get('min_duration', 200)):.0f}s, "
               f"目標 {float(f.get('target_duration', 270)):.0f}s, "
               f"上限 {float(f.get('max_duration', 300)):.0f}s)")
        prof_tag = ""
        if m.get("speed_profile") == "dynamic":
            prof_tag = (f"   [定速舊估 {m.get('naive_duration_s', 0):.0f}s, "
                        f"硬轉角 {m.get('hard_corners_total', 0)} 處]")
        auto = (m.get("auto_estimate") or {}).get("compact")
        acc_cm = float(cfg["waypoints"].get("accept_radius", 0.05)) * 100.0
        auto_detail = (
            f"整段任務 {auto['total_s']:.0f} s = 起飛 {auto['takeoff_s']:.0f} + "
            f"航線 {auto['nav_s']:.0f} + 降落 {auto['land_s']:.0f}s (接受半徑 {acc_cm:.0f}cm)"
            if auto else ""
        )
        # 工時基準 (flight.duration_basis): auto -> AUTO 航線工時擺第一行、GUIDED 退為參考
        basis = duration_basis(cfg)
        if basis == "auto" and auto:
            dur_lines = [
                f"AUTO 航線工時: {auto['nav_s']:.0f} s   {req}   |   {auto_detail}",
                f"GUIDED 串流工時 (參考): {t.duration:.0f} s{prof_tag}",
            ]
        else:
            dur_lines = [f"純飛行工時 (GUIDED): {t.duration:.0f} s   {req}{prof_tag}"]
            if auto:
                dur_lines.append(f"AUTO 預估: {auto_detail}")
        lines = [
            f"軌跡: {pr.pattern.display_name}    圈數: {pr.laps}    "
            f"[工時基準: {'AUTO 任務' if basis == 'auto' else 'GUIDED 串流'}]",
            *dur_lines,
            f"3D 路徑長: {t.path_length_3d:.1f} m    最大速度: {t.max_speed:.2f} m/s"
            f"    平均水平速度: {m.get('avg_speed_h', 0):.2f} m/s",
            f"高度範圍 Z: {b['z'][0]:.2f} ~ {b['z'][1]:.2f} m",
        ]
        # 任務項數: do_jump = (進場) + 一圈 block + DO_JUMP + 收尾; unroll = 全展開
        appr = f" + 進場 {auto['n_approach']}" if auto and auto.get("n_approach") else ""
        n_turn = int((auto or {}).get("n_turns", 0) or 0)
        turn_txt = f" + 轉頭 {n_turn}×2" if n_turn else ""
        if auto and auto.get("repeat") == "do_jump" and auto.get("jump_repeat", 0) > 0:
            lines.append(
                f"AUTO 任務項數: {n_comp}/{budget} [{comp_tag}] = 固定 4{appr} + 一圈 {auto['block_len']} 航點"
                f" + DO_JUMP×{auto['jump_repeat']} + 收尾 1{turn_txt}   (高精度版 {n_prec})"
            )
        else:
            lines.append(f"AUTO 任務項數 — 高精度: {n_prec}   精簡(上飛控): {n_comp}/{budget} [{comp_tag}]"
                         + (f"   (含進場繞障 {auto['n_approach']} 點)" if appr else "")
                         + (f"   (含轉頭 {n_turn}×2 項)" if n_turn else ""))
        # 原地轉頭 (waypoints.turn_in_place)
        from core.trajectory import TurnInPlace
        if n_turn:
            lines.append(f"原地轉頭: {n_turn} 個轉頭點 (CONDITION_YAW + NAV_DELAY), 實飛停下 {auto['turn_stops']} 次, "
                         f"懸停轉頭共 {auto['turn_s']:.0f} s (已含在 AUTO 航線工時)")
        elif auto and TurnInPlace.from_config(cfg).enabled:
            lines.append("原地轉頭: 已開啟, 但沒有轉向角 ≥ 門檻的航點"
                         + (" (曲線 pattern 需勾「曲線 pattern 的航點也轉」)" if pr.pattern.is_smooth else ""))
        # 起飛點 (AUTO 進場段的起點)
        n_ap = int((auto or {}).get("n_approach", 0) or 0)
        if not takeoff_is_origin(cfg) or (m.get("obstacles") or {}).get("count"):
            tk = f"起飛點: {describe_takeoff_point(cfg)}"
            tk += f"   進場繞障 {n_ap} 點" if n_ap else "   進場: 直線可達"
            lines.append(tk)
        # 障礙物 / 避障
        ob = m.get("obstacles") or {}
        if ob.get("count"):
            av = ob.get("avoid")
            if av:
                if av.get("moved") or av.get("removed") or av.get("detours") or av.get("failed"):
                    txt = (f"避障: 改道 {av['detours']} 段, 頂點推到禁區邊界 {av['moved']} 個, "
                           f"單圈路徑 {av['length_before_m']:.1f} → {av['length_after_m']:.1f} m")
                    if av.get("failed"):
                        txt += f", 繞不開 {av['failed']} 段!"
                else:
                    txt = "避障: 路徑本來就沒碰到障礙物, 未改道"
            else:
                txt = "避障: 已停用 (只顯示障礙物)"
            lines.append(f"{ob.get('description', '')}   |   {txt}")
        req_c = m.get("altitude_cycles_requested")
        eff_c = m.get("altitude_cycles_effective")
        per_lap = m.get("altitude_cycles_per_lap", 0)
        if m.get("mission_repeat") == "do_jump" and per_lap:
            note = f"高度起伏: 每圈 {per_lap} 次 × {pr.laps} 圈 = 全程 {eff_c} 次 (DO_JUMP 每圈相同"
            note += f"; 原設定 {req_c} 次)" if eff_c != req_c else ")"
            lines.append(note)
        elif req_c is not None and eff_c != req_c:
            lines.append(f"高度起伏次數: {req_c} → {eff_c} (自動避開與圈數 {pr.laps} 共振, 防止各圈重疊)")
        lines.append("—")
        lines += pr.report.as_lines()
        self.stats.setPlainText("\n".join(lines))
        ok = pr.report.ok
        self.statusBar().showMessage("安全檢查通過" if ok else "安全檢查未通過，請調整參數")

    def export(self, fmt):
        if not self.current_plan:
            self.generate()
        if not self.current_plan:
            return
        cfg = self._cfg()
        key = self.current_plan.key
        os.makedirs(_OUTPUT, exist_ok=True)
        try:
            if fmt == "waypoints":
                from io_export import export_waypoints_dual
                base = os.path.join(_OUTPUT, key + ".waypoints")
                res = export_waypoints_dual(self.current_plan, cfg, base)
                p, c = res["precision"], res["compact"]
                tag = "OK" if c["within_budget"] else "仍超過!"
                msg = (f"已匯出雙通道: 高精度 {p['total_items']} 項 | "
                       f"精簡 {c['total_items']}/{res['budget']} 項 [{tag}]")
            elif fmt == "csv":
                from io_export import export_csv
                path = os.path.join(_OUTPUT, key + ".csv")
                n = export_csv(self.current_plan, cfg, path)
                msg = f"已匯出 {path} ({n} 列)"
            else:  # png
                import viz
                path = os.path.join(_OUTPUT, key + "_preview.png")
                viz.save_png(self.current_plan, cfg, path)
                msg = f"已匯出 {path}"
            self.statusBar().showMessage(msg)
            self.stats.appendPlainText("[OK] " + msg)
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"匯出失敗: {e}")
            self.stats.appendPlainText(f"[ERROR] 匯出失敗: {e}")

    def fly(self):
        if not self.current_plan:
            self.generate()
        if not self.current_plan or not self.current_plan.report.ok:
            QtWidgets.QMessageBox.warning(self, "無法飛行", "安全檢查未通過或尚未產生軌跡。")
            return
        cfg = self._cfg()
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("GUIDED 飛行")
        box.setText(
            f"連線: {cfg['guided']['connection']}\n"
            f"軌跡: {self.current_plan.pattern.display_name}, "
            f"工時 {self.current_plan.trajectory.duration:.0f}s\n\n"
            "「乾跑」只連線測試不解鎖；「真的起飛」會解鎖並飛行 (請確認場地淨空)。"
        )
        dry = box.addButton("乾跑", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        real = box.addButton("真的起飛", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (dry, real):
            return
        confirm = clicked is real
        self._start_fly(cfg, confirm)

    def _start_fly(self, cfg, confirm):
        self.stats.appendPlainText(f"--- 開始{'飛行' if confirm else '乾跑'} ---")
        self._worker = FlyWorker(self.current_plan, cfg, confirm)
        self._worker.log.connect(lambda m: self.stats.appendPlainText(m))
        self._worker.done.connect(lambda ok: self.statusBar().showMessage(
            "飛行流程結束" if ok else "飛行流程中止"))
        self._worker.start()

    def load_cfg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "載入設定", _ROOT, "YAML (*.yaml *.yml)")
        if path:
            try:
                cfg = load_config(path)
                self._build_panels(cfg)           # 重建面板 (含障礙物清單)
            except Exception as e:  # noqa: BLE001
                self.statusBar().showMessage(f"載入失敗: {e}")
                self.stats.setPlainText(f"[ERROR] 載入 {path} 失敗: {e}")
                return
            self.base_cfg = cfg
            self.generate()
            self.statusBar().showMessage(f"已載入 {path}")

    def save_cfg(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "儲存設定", _ROOT, "YAML (*.yaml)")
        if path:
            save_config(self._cfg(), path)
            self.statusBar().showMessage(f"已儲存 {path}")


def launch(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication([sys.argv[0]] + argv)
    win = MainWindow()
    if "--obstacles" in argv:          # python main.py --obstacles: 直接切到「障礙物」頁
        win.tabs.setCurrentWidget(win.obs_panel)
    win.show()
    sys.exit(app.exec())
