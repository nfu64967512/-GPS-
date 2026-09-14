"""
障礙物面板：已知障礙物 (箱子 / 光球座標) 清單 + 編輯表單 + 從 VRPN (Motive) 讀取。

面板自己持有 items (config obstacles.items 格式的 dict 清單)，apply_to() 把整個 obstacles 區塊
套回 cfg (儲存設定時一併存成 YAML)。VRPN 讀取在背景執行緒進行，成功後取代 source=vrpn 的項目、
保留手動輸入的項目，並發出 changed 讓主視窗重新產生路徑。
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

from PyQt6 import QtCore, QtWidgets

from core.config import deep_update
from core.geometry import takeoff_point as resolve_takeoff_point
from core.obstacles import (DEFAULT_CLEARANCE, DEFAULT_CORNER_SEGMENTS, DEFAULT_DRONE_SIZE,
                            DEFAULT_OVER_MAX_TOP, DEFAULT_VERTICAL_CLEARANCE, drone_radius,
                            pivot_above, pivot_below, recommended_clearance,
                            recommended_vertical_clearance)
from core.vrpn_client import DEFAULT_DRONE_NAME


def _dspin(lo, hi, val, step=0.1, decimals=2):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(float(val))
    return w


class VRPNWorker(QtCore.QThread):
    """背景連 VRPN 收幾秒, 轉成 obstacles.items。"""

    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object, str)      # (items 或 None, 錯誤訊息)

    def __init__(self, vcfg: Dict):
        super().__init__()
        self._vcfg = copy.deepcopy(vcfg)

    def run(self):
        try:
            from core.vrpn_client import read_scene, sample_to_enu, trackers_to_items
            v = self._vcfg
            drone, boxes = read_scene(v.get("server", "localhost:3883"), v,
                                      log=lambda m: self.log.emit(m))
            payload = {
                "items": trackers_to_items(boxes, v),
                "drone": sample_to_enu(drone, v) if drone is not None else None,
                "drone_name": str(v.get("drone", "") or ""),
            }
            err = ""
            if not payload["items"] and payload["drone"] is None:
                err = "VRPN 沒有收到任何 rigid body (確認 Motive 已開 VRPN 串流、rigid body 已啟用)"
            self.done.emit(payload, err)
        except Exception as e:  # noqa: BLE001
            self.done.emit(None, str(e))


class ObstaclePanel(QtWidgets.QScrollArea):
    changed = QtCore.pyqtSignal()      # 障礙物清單有變 (VRPN 讀取完成 / 新增刪除) -> 主視窗重新產生

    def __init__(self, cfg: Dict, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QtWidgets.QWidget()
        self.setWidget(host)
        form = QtWidgets.QVBoxLayout(host)
        form.setSpacing(8)

        o = cfg.get("obstacles", {}) or {}
        self.items: List[Dict] = [copy.deepcopy(it) for it in (o.get("items") or [])]
        self._loading = False
        self._worker: Optional[VRPNWorker] = None
        # 保留原始的 vrpn 設定, vrpn_config() 只覆蓋面板上有的鍵 ——
        # 否則 YAML 裡設的 sizes / height_tol / hull_name 會被面板悄悄丟掉
        self._vrpn_base: Dict = copy.deepcopy(o.get("vrpn", {}) or {})

        # ---- 避障 ----
        g = QtWidgets.QGroupBox("避障 (只繞不越)")
        gl = QtWidgets.QFormLayout(g)
        self.enabled = QtWidgets.QCheckBox("啟用避障 (路徑水平繞開障礙物)")
        self.enabled.setChecked(bool(o.get("enabled", True)))
        self.enabled.setToolTip("取消 = 不改道; 障礙物仍會顯示, 路徑穿過時安全檢查會擋下")
        ds = list(o.get("drone_size") or DEFAULT_DRONE_SIZE)
        while len(ds) < 3:
            ds.append(DEFAULT_DRONE_SIZE[2])
        self.drone_sx = _dspin(0.05, 5.0, float(ds[0]), 0.05)
        self.drone_sy = _dspin(0.05, 5.0, float(ds[1]), 0.05)
        self.drone_sz = _dspin(0.02, 3.0, float(ds[2]), 0.05)
        mh = o.get("marker_height", None)
        self.marker_h = _dspin(0.02, 3.0, float(mh) if mh is not None else float(ds[2]), 0.01)
        self.marker_h.setToolTip(
            "停在地上時光球離地高度 (m)。光球貼在頂板 -> 參考點在機身上緣, 整個機體掛在規劃高度下方,\n"
            "離地與越箱的餘裕要從這個值扣")
        self.marker_h.valueChanged.connect(lambda _: self._sync_clearance())
        for wdg in (self.drone_sx, self.drone_sy, self.drone_sz):
            wdg.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            wdg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            wdg.setToolTip("機身水平尺寸 (m)。規劃的是機身中心的路徑, 會碰到東西的是離中心最遠的角")
            wdg.valueChanged.connect(lambda _: self._sync_clearance())
        clr_cfg = o.get("clearance", DEFAULT_CLEARANCE)
        self.clr_auto = QtWidgets.QCheckBox("自動 (依機身尺寸推導)")
        self.clr_auto.setChecked(not isinstance(clr_cfg, (int, float)))
        self.clr_auto.setToolTip("= 機身半對角 + 航點切角 (accept_radius) + 0.10 m 追蹤餘裕")
        self.clearance = _dspin(0.0, 5.0, float(clr_cfg) if isinstance(clr_cfg, (int, float))
                                else DEFAULT_CLEARANCE, 0.05)
        self.clearance.setToolTip("路徑中心離障礙物外緣的最小水平距離 (m); 小於機身半對角會被安全檢查擋下")
        self.clr_auto.toggled.connect(lambda _: self._sync_clearance())
        self.clearance.valueChanged.connect(lambda _: self._sync_clearance())
        self.clr_hint = QtWidgets.QLabel("")
        self.clr_hint.setWordWrap(True)
        self.corner = QtWidgets.QSpinBox()
        self.corner.setRange(1, 6)
        self.corner.setValue(int(o.get("corner_segments", DEFAULT_CORNER_SEGMENTS)))
        self.corner.setToolTip("繞過箱角時每 90° 用幾段折線: 越少航點越少但繞得越開 (1 = 直接切 45° 角)")
        self.low_mode = QtWidgets.QComboBox()
        self.low_mode.addItems(["over", "around"])
        self.low_mode.setCurrentText("around" if str(o.get("low_mode", "over")).lower() == "around" else "over")
        self.low_mode.setToolTip(
            "over = 箱頂不超過下面高度的箱子不繞, 路徑進入其禁區時拉高到箱頂 + 垂直安全距離越過 (前後自動斜坡);\n"
            "其餘 (太高 / 拉高後超過天花板邊界) 仍水平繞開。around = 一律水平繞開")
        self.over_max_top = _dspin(0.0, 5.0, o.get("over_max_top", DEFAULT_OVER_MAX_TOP), 0.05)
        self.over_max_top.setToolTip("箱頂高度 (m) 不超過此值才越過; 疊起來 / 放桌上的箱子會超過而改為繞開")
        self.vclear = _dspin(0.1, 3.0, o.get("vertical_clearance", DEFAULT_VERTICAL_CLEARANCE), 0.05)
        self.vclear.setToolTip("越過時路徑離箱頂的最小垂直距離 (m); 下洗氣流會吹動輕的紙箱, 不要太小")
        gl.addRow("", self.enabled)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(self.drone_sx)
        size_row.addWidget(self.drone_sy)
        size_row.addWidget(self.drone_sz)
        gl.addRow("機身 長×寬×高 m", size_row)
        gl.addRow("光球離地高 m", self.marker_h)
        gl.addRow("", self.clr_auto)
        gl.addRow("安全距離 m", self.clearance)
        gl.addRow("", self.clr_hint)
        gl.addRow("繞角段數 /90°", self.corner)
        gl.addRow("低矮箱子", self.low_mode)
        gl.addRow("可越過箱頂 ≤ m", self.over_max_top)
        gl.addRow("垂直安全距離 m", self.vclear)
        form.addWidget(g)

        # ---- 起飛點 ----
        g = QtWidgets.QGroupBox("起飛點 (飛機實際停放位置)")
        gl = QtWidgets.QFormLayout(g)
        tx, ty = resolve_takeoff_point(cfg)
        is_origin = abs(tx) < 1e-9 and abs(ty) < 1e-9
        self.tk_mode = QtWidgets.QComboBox()
        self.tk_mode.addItems(["房間原點 (0,0)", "指定座標"])
        self.tk_mode.setCurrentIndex(0 if is_origin else 1)
        self.tk_mode.setToolTip(
            "AUTO 起飛後飛向第一個航點的那一段, 是從這裡出發算的。\n"
            "飛機沒放在這個點上, 那一段就可能穿過箱子 —— 用下面的 VRPN 讀取量實際位置最準")
        self.tk_x = _dspin(-50, 50, tx, 0.05, 3)
        self.tk_y = _dspin(-50, 50, ty, 0.05, 3)
        self.tk_status = QtWidgets.QLabel("")
        self.tk_status.setWordWrap(True)
        self.tk_mode.currentIndexChanged.connect(lambda _: self._sync_takeoff())
        for w in (self.tk_x, self.tk_y):
            w.valueChanged.connect(lambda _: self._sync_takeoff())
        gl.addRow("來源", self.tk_mode)
        gl.addRow("X (東) m", self.tk_x)
        gl.addRow("Y (北) m", self.tk_y)
        gl.addRow("", self.tk_status)
        form.addWidget(g)

        # ---- 清單 ----
        g = QtWidgets.QGroupBox("障礙物清單 (房間 ENU, m)")
        gl = QtWidgets.QVBoxLayout(g)
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(110)
        self.list.currentRowChanged.connect(self._on_select)
        gl.addWidget(self.list)
        row = QtWidgets.QGridLayout()
        b_add = QtWidgets.QPushButton("新增箱子")
        b_add.clicked.connect(self._add_box)
        b_pts = QtWidgets.QPushButton("新增光球組")
        b_pts.clicked.connect(self._add_points)
        b_del = QtWidgets.QPushButton("刪除選取")
        b_del.clicked.connect(self._delete)
        b_clr = QtWidgets.QPushButton("全部清空")
        b_clr.clicked.connect(self._clear)
        for i, b in enumerate((b_add, b_pts, b_del, b_clr)):
            row.addWidget(b, i // 2, i % 2)
        gl.addLayout(row)
        form.addWidget(g)

        # ---- 編輯 ----
        g = QtWidgets.QGroupBox("編輯選取的障礙物")
        self.edit_group = g
        gl = QtWidgets.QFormLayout(g)
        self.e_name = QtWidgets.QLineEdit()
        self.e_kind = QtWidgets.QComboBox()
        self.e_kind.addItems(["box", "points"])
        self.e_x = _dspin(-50, 50, 0.0, 0.05, 3)
        self.e_y = _dspin(-50, 50, 0.0, 0.05, 3)
        self.e_zb = _dspin(0.0, 20, 0.0, 0.05, 3)
        self.e_sx = _dspin(0.0, 20, 0.5, 0.05, 3)
        self.e_sy = _dspin(0.0, 20, 0.5, 0.05, 3)
        self.e_sz = _dspin(0.01, 20, 0.5, 0.05, 3)
        self.e_yaw = _dspin(-180, 180, 0.0, 5, 1)
        self.e_points = QtWidgets.QPlainTextEdit()
        self.e_points.setPlaceholderText("每行一顆光球: x y z (m)\n例:\n1.0 0.5 0.45\n1.4 0.5 0.45\n1.4 0.8 0.45")
        self.e_points.setMaximumHeight(90)
        self.e_source = QtWidgets.QLabel("")
        self.e_source.setWordWrap(True)
        gl.addRow("名稱", self.e_name)
        gl.addRow("種類", self.e_kind)
        gl.addRow("中心 X", self.e_x)
        gl.addRow("中心 Y", self.e_y)
        gl.addRow("底部 Z", self.e_zb)
        gl.addRow("長 (x)", self.e_sx)
        gl.addRow("寬 (y)", self.e_sy)
        gl.addRow("高 (z)", self.e_sz)
        gl.addRow("偏航 deg", self.e_yaw)
        gl.addRow("光球座標", self.e_points)
        gl.addRow("來源", self.e_source)
        form.addWidget(g)
        for w in (self.e_x, self.e_y, self.e_zb, self.e_sx, self.e_sy, self.e_sz, self.e_yaw):
            w.valueChanged.connect(self._on_edit)
        self.e_name.editingFinished.connect(self._on_edit)
        self.e_kind.currentTextChanged.connect(self._on_edit)
        self.e_points.textChanged.connect(self._on_edit)

        # ---- VRPN ----
        g = QtWidgets.QGroupBox("從 VRPN (Motive) 讀取光球 / rigid body")
        gl = QtWidgets.QFormLayout(g)
        v = o.get("vrpn", {}) or {}
        self.v_server = QtWidgets.QLineEdit(str(v.get("server", "localhost:3883")))
        self.v_server.setToolTip("Motive: Data Streaming -> VRPN Broadcast 開啟, 預設 port 3883")
        self.v_seconds = _dspin(0.5, 30, v.get("seconds", 2.0), 0.5, 1)
        self.v_axes = QtWidgets.QComboBox()
        self.v_axes.setEditable(True)
        self.v_axes.addItems(["y_up", "z_up", "x,-z,y", "x,y,z"])
        self.v_axes.setCurrentText(str(v.get("axes", "y_up")))
        self.v_axes.setToolTip("Motive 串流的 Up Axis: y_up (預設) -> ENU (x, -z, y); z_up -> 不轉;\n"
                               "或自訂 'x,-z,y' (ENU 的 x/y/z 各取 Motive 的哪一軸)。需與 mocap->飛控橋接同一座標系")
        self.v_trackers = QtWidgets.QLineEdit(",".join(str(t) for t in (v.get("trackers") or [])))
        self.v_trackers.setPlaceholderText("空 = 全部; 或 box1,box2")
        self.v_drone = QtWidgets.QLineEdit(str(v.get("drone", DEFAULT_DRONE_NAME) or ""))
        self.v_drone.setPlaceholderText(DEFAULT_DRONE_NAME)
        self.v_drone.setToolTip("飛機的 rigid body 名稱: 讀進來當起飛點, 且不會變成障礙物")
        self.v_take = QtWidgets.QCheckBox("讀取時一併更新起飛點")
        self.v_take.setChecked(bool(v.get("takeoff_from_drone", True)))
        self.v_take.setToolTip("勾選 = 每次「從 VRPN 讀取」都把飛機目前的位置設成起飛點")
        self.v_exclude = QtWidgets.QLineEdit(",".join(str(t) for t in (v.get("exclude") or [])))
        self.v_exclude.setPlaceholderText("例: drone_01 (無人機自己不當障礙物)")
        self.v_exclude.setToolTip("這些 rigid body 名稱不會變成障礙物 (逗號分隔), 例如無人機本身")
        self.v_mode = QtWidgets.QComboBox()
        self.v_mode.addItems(["each_box", "hull"])
        self.v_mode.setCurrentText(str(v.get("mode", "each_box")))
        self.v_mode.setToolTip("each_box = 每個 rigid body 一個箱子 (下面的尺寸, 偏航取自姿態)\n"
                               "hull = 全部 rigid body 位置合成一個障礙物 (XY 凸包; 例如每個角各放一組光球)")
        bs = v.get("box_size") or [0.5, 0.5, 0.5]
        self.v_sx = _dspin(0.0, 20, bs[0], 0.05, 2)
        self.v_sy = _dspin(0.0, 20, bs[1], 0.05, 2)
        self.v_sz = _dspin(0.01, 20, bs[2], 0.05, 2)
        self.v_pivot = QtWidgets.QComboBox()
        self.v_pivot.addItems(["top", "center", "bottom"])
        self.v_pivot.setCurrentText(str(v.get("pivot", "top")))
        self.v_pivot.setToolTip("rigid body 樞紐點 (Motive 預設 = 光球幾何中心) 在箱子的頂面 / 中心 / 底面")
        self.v_orient = QtWidgets.QComboBox()
        self.v_orient.addItems(["auto_square", "auto_long_x", "auto_long_y", "fixed"])
        self.v_orient.setCurrentText(str(v.get("orientation", "auto_square")))
        self.v_orient.setToolTip(
            "同一種箱子有躺有立時, 用量到的頂面高度判斷哪一邊垂直 (樞紐點需為 top / center):\n"
            "auto_square = 底面保守取較長邊的正方形 (VRPN 看不出哪一邊沿 x; 預設)\n"
            "auto_long_x / auto_long_y = 底面長邊沿 x / 沿 y\n"
            "fixed = 一律照「箱子三邊」的 x y z 順序")
        off = v.get("offset") or [0.0, 0.0, 0.0]
        self.v_ox = _dspin(-20, 20, off[0], 0.05, 2)
        self.v_oy = _dspin(-20, 20, off[1], 0.05, 2)
        self.v_oz = _dspin(-20, 20, off[2], 0.05, 2)
        for w in (self.v_sx, self.v_sy, self.v_sz, self.v_ox, self.v_oy, self.v_oz):
            # 三個並排的小數字框: 拿掉上下箭頭, 數字才看得清楚 (仍可用滾輪 / 鍵盤上下調)
            w.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            w.setMinimumWidth(52)
        for w, tip in ((self.v_sx, "長 x (m)"), (self.v_sy, "寬 y (m)"), (self.v_sz, "高 z (m)"),
                       (self.v_ox, "平移 x (m)"), (self.v_oy, "平移 y (m)"), (self.v_oz, "平移 z (m)")):
            w.setToolTip(tip)
        self.v_btn = QtWidgets.QPushButton("從 VRPN 讀取")
        self.v_btn.clicked.connect(self.read_vrpn)
        self.v_status = QtWidgets.QLabel("")
        self.v_status.setWordWrap(True)
        gl.addRow("伺服器", self.v_server)
        gl.addRow("收幾秒", self.v_seconds)
        gl.addRow("座標軸", self.v_axes)
        gl.addRow("飛機 rigid body", self.v_drone)
        gl.addRow("", self.v_take)
        gl.addRow("只讀 tracker", self.v_trackers)
        gl.addRow("排除 tracker", self.v_exclude)
        gl.addRow("模式", self.v_mode)
        size_row = QtWidgets.QHBoxLayout()
        for w in (self.v_sx, self.v_sy, self.v_sz):
            size_row.addWidget(w)
        gl.addRow("箱子三邊 m", size_row)
        gl.addRow("躺/立判斷", self.v_orient)
        gl.addRow("樞紐點", self.v_pivot)
        off_row = QtWidgets.QHBoxLayout()
        for w in (self.v_ox, self.v_oy, self.v_oz):
            off_row.addWidget(w)
        gl.addRow("座標平移 x y z", off_row)
        gl.addRow("", self.v_btn)
        gl.addRow("", self.v_status)
        form.addWidget(g)

        form.addStretch(1)
        self._loading = True
        self._sync_clearance()
        self._loading = False
        self._sync_takeoff()
        self._refresh_list()
        if self.items:
            self.list.setCurrentRow(0)
        else:
            self._load_item(None)

    # ---- 清單 <-> 表單 ----
    @staticmethod
    def _item_text(it: Dict) -> str:
        name = it.get("name", "?")
        src = " (VRPN)" if it.get("source") == "vrpn" else ""
        if str(it.get("kind", "box")) == "points":
            n = len(it.get("points") or [])
            return f"{name}{src}  光球×{n}"
        sz = it.get("size") or [0, 0, 0]
        yaw = float(it.get("yaw_deg", 0) or 0)
        return (f"{name}{src}  ({float(it.get('x', 0)):.2f}, {float(it.get('y', 0)):.2f})  "
                f"{float(sz[0]):.2f}×{float(sz[1]):.2f}×{float(sz[2]):.2f}"
                + (f"  {yaw:.0f}°" if abs(yaw) > 0.5 else ""))

    def _refresh_list(self, keep_row: Optional[int] = None):
        self._loading = True
        row = self.list.currentRow() if keep_row is None else keep_row
        self.list.clear()
        for it in self.items:
            self.list.addItem(self._item_text(it))
        if 0 <= row < len(self.items):
            self.list.setCurrentRow(row)
        self._loading = False

    def _on_select(self, row: int):
        if self._loading:
            return
        self._load_item(self.items[row] if 0 <= row < len(self.items) else None)

    def _load_item(self, it: Optional[Dict]):
        self._loading = True
        self.edit_group.setEnabled(it is not None)
        if it is None:
            self.e_name.setText("")
            self.e_source.setText("")
            self.e_points.setPlainText("")
        else:
            self.e_name.setText(str(it.get("name", "")))
            self.e_kind.setCurrentText("points" if str(it.get("kind", "box")) == "points" else "box")
            self.e_x.setValue(float(it.get("x", 0.0) or 0.0))
            self.e_y.setValue(float(it.get("y", 0.0) or 0.0))
            self.e_zb.setValue(float(it.get("z_bottom", 0.0) or 0.0))
            sz = it.get("size") or [0.5, 0.5, 0.5]
            self.e_sx.setValue(float(sz[0])); self.e_sy.setValue(float(sz[1])); self.e_sz.setValue(float(sz[2]))
            self.e_yaw.setValue(float(it.get("yaw_deg", 0.0) or 0.0))
            pts = it.get("points") or []
            self.e_points.setPlainText("\n".join(" ".join(f"{float(v):.3f}" for v in p) for p in pts))
            src = "VRPN rigid body: " + str(it.get("tracker", "")) if it.get("source") == "vrpn" else "手動輸入"
            if it.get("vrpn_z") is not None:
                src += f"  (樞紐 z {float(it['vrpn_z']):.2f} m)"
            if it.get("note"):
                src += "\n" + str(it["note"])
            self.e_source.setText(src)
        self._sync_kind_widgets()
        self._loading = False

    def _sync_kind_widgets(self):
        is_pts = self.e_kind.currentText() == "points"
        for w in (self.e_x, self.e_y, self.e_sx, self.e_sy, self.e_sz, self.e_yaw):
            w.setEnabled(not is_pts)
        self.e_points.setEnabled(is_pts)

    @staticmethod
    def parse_points(text: str) -> List[List[float]]:
        pts = []
        for ln in text.splitlines():
            ln = ln.replace(",", " ").strip()
            if not ln or ln.startswith("#"):
                continue
            vals = [float(v) for v in ln.split()]
            if len(vals) == 2:
                vals.append(0.0)
            if len(vals) != 3:
                raise ValueError(f"光球座標每行需為 x y z: {ln!r}")
            pts.append(vals)
        return pts

    def _on_edit(self, *_):
        if self._loading:
            return
        row = self.list.currentRow()
        if not (0 <= row < len(self.items)):
            return
        it = self.items[row]
        it["name"] = self.e_name.text().strip() or it.get("name", f"obs{row + 1}")
        it["kind"] = self.e_kind.currentText()
        if it["kind"] == "points":
            try:
                it["points"] = self.parse_points(self.e_points.toPlainText())
                self.e_points.setStyleSheet("")
            except ValueError:
                self.e_points.setStyleSheet("border: 1px solid #E74C3C;")
            it["z_bottom"] = float(self.e_zb.value())
        else:
            it["x"] = float(self.e_x.value())
            it["y"] = float(self.e_y.value())
            it["z_bottom"] = float(self.e_zb.value())
            it["size"] = [float(self.e_sx.value()), float(self.e_sy.value()), float(self.e_sz.value())]
            it["yaw_deg"] = float(self.e_yaw.value())
        self._sync_kind_widgets()
        self._loading = True
        self.list.item(row).setText(self._item_text(it))
        self._loading = False

    # ---- 按鈕 ----
    def _next_name(self, prefix: str) -> str:
        names = {it.get("name") for it in self.items}
        k = 1
        while f"{prefix}{k}" in names:
            k += 1
        return f"{prefix}{k}"

    def _add_box(self):
        self.items.append({"name": self._next_name("box"), "kind": "box", "x": 0.0, "y": 0.0,
                           "z_bottom": 0.0, "size": [0.5, 0.5, 0.5], "yaw_deg": 0.0, "source": "manual"})
        self._refresh_list(len(self.items) - 1)
        self._load_item(self.items[-1])
        self.changed.emit()

    def _add_points(self):
        self.items.append({"name": self._next_name("markers"), "kind": "points", "z_bottom": 0.0,
                           "points": [[1.0, 1.0, 0.5], [1.4, 1.0, 0.5], [1.4, 1.3, 0.5], [1.0, 1.3, 0.5]],
                           "source": "manual"})
        self._refresh_list(len(self.items) - 1)
        self._load_item(self.items[-1])
        self.changed.emit()

    def _delete(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.items):
            del self.items[row]
            self._refresh_list(min(row, len(self.items) - 1))
            self._load_item(self.items[self.list.currentRow()] if self.items else None)
            self.changed.emit()

    def _clear(self):
        if not self.items:
            return
        self.items.clear()
        self._refresh_list()
        self._load_item(None)
        self.changed.emit()

    def set_items(self, items: List[Dict]):
        self.items = [copy.deepcopy(it) for it in items]
        self._refresh_list(0 if self.items else -1)
        self._load_item(self.items[0] if self.items else None)

    # ---- VRPN ----
    def vrpn_config(self) -> Dict:
        """面板上的 VRPN 設定, 疊在載入時的原始設定之上 (不在面板上的鍵原樣保留)。"""
        return {
            **self._vrpn_base,
            "server": self.v_server.text().strip() or "localhost:3883",
            "seconds": float(self.v_seconds.value()),
            "axes": self.v_axes.currentText().strip() or "y_up",
            "offset": [float(self.v_ox.value()), float(self.v_oy.value()), float(self.v_oz.value())],
            "trackers": [t.strip() for t in self.v_trackers.text().split(",") if t.strip()],
            "exclude": [t.strip() for t in self.v_exclude.text().split(",") if t.strip()],
            "mode": self.v_mode.currentText(),
            "box_size": [float(self.v_sx.value()), float(self.v_sy.value()), float(self.v_sz.value())],
            "orientation": self.v_orient.currentText(),
            "pivot": self.v_pivot.currentText(),
            "drone": self.v_drone.text().strip() or DEFAULT_DRONE_NAME,
            "takeoff_from_drone": bool(self.v_take.isChecked()),
        }

    # ---- 安全距離 ----
    def _sync_clearance(self):
        """依機身尺寸更新安全距離的可用狀態與說明。"""
        cfg_like = {"obstacles": {"drone_size": [float(self.drone_sx.value()),
                                                 float(self.drone_sy.value()),
                                                 float(self.drone_sz.value())],
                                  "marker_height": float(self.marker_h.value())},
                    "waypoints": {"accept_radius": 0.05}}
        try:
            rad = drone_radius(cfg_like)
            rec = recommended_clearance(cfg_like)
            below, above = pivot_below(cfg_like), pivot_above(cfg_like)
            vrec = recommended_vertical_clearance(cfg_like)
        except Exception:  # noqa: BLE001
            self.clr_hint.setText("機身尺寸不合理")
            return
        auto = self.clr_auto.isChecked()
        self.clearance.setEnabled(not auto)
        if auto:
            self.clearance.blockSignals(True)
            self.clearance.setValue(rec)
            self.clearance.blockSignals(False)
        clr = float(self.clearance.value())
        txt = f"機身半對角 {rad:.2f} m，建議安全距離 ≥ {rec:.2f} m"
        if clr < rad:
            txt += f"\n[!] 目前 {clr:.2f} m 小於機身半徑，機身一定會碰到"
        else:
            txt += f"；目前留給定位誤差 {clr - rad:.2f} m"
        vclr = float(self.vclear.value())
        txt += (f"\n規劃高度下方機體 {below:.2f} m、上方 {above:.2f} m，"
                f"越箱垂直距離建議 ≥ {vrec:.2f} m")
        if vclr < below:
            txt += f"（目前 {vclr:.2f} m 不足，機身底部會碰到箱頂）"
        self.clr_hint.setText(txt)
        if not self._loading:
            self.changed.emit()

    # ---- 起飛點 ----
    def _sync_takeoff(self):
        """起飛點欄位啟用狀態 + 說明文字。"""
        manual = self.tk_mode.currentIndex() == 1
        self.tk_x.setEnabled(manual)
        self.tk_y.setEnabled(manual)
        if not manual:
            self.tk_status.setText("飛機放在房間原點起飛 (舊行為)")
        else:
            x, y = float(self.tk_x.value()), float(self.tk_y.value())
            self.tk_status.setText(
                f"距房間原點 {(x * x + y * y) ** 0.5:.2f} m (X {x * 100:+.0f} cm, Y {y * 100:+.0f} cm)")
        if not self._loading:
            self.changed.emit()

    def takeoff_point(self):
        """回傳 'origin' 或 [x, y] (寫進 waypoints.takeoff_point 的值)。"""
        if self.tk_mode.currentIndex() == 0:
            return "origin"
        return [round(float(self.tk_x.value()), 4), round(float(self.tk_y.value()), 4)]

    def set_takeoff_point(self, x: float, y: float) -> bool:
        """設定起飛點。回傳是否採用 —— 非有限值或超出欄位範圍時拒絕, 不讓 spinbox 把 NaN 夾成 ±50
        再當成「實測」寫進設定。不發出 changed (由呼叫端統一發一次)。"""
        x, y = float(x), float(y)
        lo, hi = self.tk_x.minimum(), self.tk_x.maximum()
        if not (math.isfinite(x) and math.isfinite(y)) or not (lo <= x <= hi and lo <= y <= hi):
            return False
        self._loading = True
        try:
            self.tk_mode.setCurrentIndex(1)
            self.tk_x.setValue(x)
            self.tk_y.setValue(y)
            self._sync_takeoff()
        finally:
            self._loading = False
        return True

    def read_vrpn(self):
        if self._worker is not None and self._worker.isRunning():
            return
        self.v_btn.setEnabled(False)
        self.v_status.setText("連線中 ...")
        self._worker = VRPNWorker(self.vrpn_config())
        self._worker.log.connect(lambda m: self.v_status.setText(m))
        self._worker.done.connect(self._on_vrpn_done)
        self._worker.start()

    def _on_vrpn_done(self, payload, err: str):
        self.v_btn.setEnabled(True)
        if payload is None:
            self.v_status.setText(f"[ERROR] {err}")
            return
        items = payload.get("items") or []
        drone = payload.get("drone")
        if not items and drone is None:
            self.v_status.setText(f"[WARN] {err}")
            return
        msg = []
        if items:
            manual = [it for it in self.items if it.get("source") != "vrpn"]
            self.items = manual + list(items)
            self._refresh_list(len(manual))
            self._load_item(self.items[len(manual)])
            msg.append(f"箱子 {len(items)} 個: " + ", ".join(str(it.get("name")) for it in items))
        else:
            msg.append("沒有箱子 rigid body")
        if drone is not None:
            dx, dy, dz = drone
            if self.v_take.isChecked():
                if self.set_takeoff_point(dx, dy):
                    msg.append(f"起飛點已更新為 {payload.get('drone_name') or '飛機'} 的位置 "
                               f"({dx:+.2f}, {dy:+.2f}) m, 高 {dz:.2f} m")
                else:
                    msg.append(f"[!] 飛機座標 ({dx}, {dy}) 不合理 (非有限值或超出範圍), 起飛點未變 "
                               f"—— 請檢查 Motive 的座標軸 / 單位")
            else:
                msg.append(f"{payload.get('drone_name') or '飛機'} 在 ({dx:+.2f}, {dy:+.2f}) m "
                           f"(未套用: 未勾選更新起飛點)")
        else:
            msg.append(f"沒讀到飛機 rigid body「{payload.get('drone_name')}」, 起飛點未變")
        self.v_status.setText("[OK] " + "; ".join(msg))
        self.changed.emit()

    # ---- 輸出 ----
    def to_config(self) -> Dict:
        return {
            "enabled": bool(self.enabled.isChecked()),
            "drone_size": [float(self.drone_sx.value()), float(self.drone_sy.value()),
                           float(self.drone_sz.value())],
            "marker_height": float(self.marker_h.value()),
            "clearance": "auto" if self.clr_auto.isChecked() else float(self.clearance.value()),
            "corner_segments": int(self.corner.value()),
            "low_mode": self.low_mode.currentText(),
            "over_max_top": float(self.over_max_top.value()),
            "vertical_clearance": float(self.vclear.value()),
            "items": [copy.deepcopy(it) for it in self.items],
            "vrpn": self.vrpn_config(),
        }

    def apply_to(self, cfg: Dict) -> Dict:
        return deep_update(cfg, {
            "obstacles": self.to_config(),
            "waypoints": {"takeoff_point": self.takeoff_point()},
        })
