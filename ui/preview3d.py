"""嵌入式 3D 預覽畫布 (matplotlib on PyQt6)。"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6 import QtWidgets

import viz


class Preview3D(QtWidgets.QWidget):
    def __init__(self, parent=None, dpi: int = 120):
        super().__init__(parent)
        self.fig = Figure(figsize=(9, 7), dpi=dpi, facecolor="#0E1318")
        self.canvas = FigureCanvasQTAgg(self.fig)
        # 讓畫布盡量撐滿中央面板
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.canvas.setMinimumSize(560, 460)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        self.fig.set_facecolor("#0E1318")
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0E1318")
        ax.axis("off")
        ax.text(0.5, 0.5, "選擇左側軌跡並按「產生 / 預覽」",
                ha="center", va="center", fontsize=13, color="#9AA7B3")
        self.canvas.draw_idle()

    def show_plan(self, plan, cfg):
        viz.draw_trajectory(self.fig, plan, cfg)
        self.canvas.draw_idle()
