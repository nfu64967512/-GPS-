"""
3D 軌跡視覺化 (純 matplotlib，不依賴 Qt)。
CLI 用來存 PNG；GUI 的 preview3d 也呼叫 draw_trajectory 畫到嵌入的畫布。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.geometry import SafeBox, takeoff_point
from core.planner import PlanResult, basis_duration, duration_basis


def _draw_box(ax, box: SafeBox):
    """畫安全盒線框。"""
    xs = [box.x_min, box.x_max]
    ys = [box.y_min, box.y_max]
    zs = [box.z_min, box.z_max]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color="#FF6A00", lw=0.8, alpha=0.5,
        )


def _draw_obstacles(ax, field, floor_z: float = 0.0):
    """畫障礙物 (箱子 = 半透明柱體, 光球 = 小點) 與地板上的外擴禁區虛線。"""
    if field is None or not getattr(field, "obstacles", None):
        return
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    enabled = bool(getattr(field, "enabled", True))
    over = {z.index: z for z in getattr(field, "over", [])} if enabled else {}
    for i, o in enumerate(field.obstacles):
        is_over = i in over
        face = ("#F39C12" if is_over else "#E74C3C") if enabled else "#7F8C8D"
        edge = "#FAD7A0" if is_over else "#F5B7B1"
        fp = np.asarray(o.footprint, dtype=float)
        if len(fp) < 3:                       # 退化 (單點 / 線段): 用一圈小方框表示
            c = fp.mean(axis=0)
            fp = c + 0.05 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        z0, z1 = float(o.z_bottom), float(o.z_top)
        bottom = [(x, y, z0) for x, y in fp]
        top = [(x, y, z1) for x, y in fp]
        faces = [bottom, top]
        n = len(fp)
        for i in range(n):
            j = (i + 1) % n
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        coll = Poly3DCollection(faces, alpha=0.35, facecolor=face, edgecolor=edge, linewidths=0.8)
        ax.add_collection3d(coll)
        cx, cy = o.center
        label = f"{o.name} (越過)" if is_over else o.name
        ax.text(cx, cy, z1 + 0.05, label, color=edge, fontsize=8, ha="center")
        if is_over:
            # 越過時要維持的高度: 在箱頂 + 垂直安全距離的高度畫一圈外擴禁區 (點線)
            zz = np.vstack([over[i].poly, over[i].poly[:1]])
            ax.plot(zz[:, 0], zz[:, 1], np.full(len(zz), over[i].z_req), color="#F39C12", lw=0.9,
                    ls=":", alpha=0.9)
        if o.markers is not None and len(o.markers):
            m = np.asarray(o.markers, dtype=float)
            ax.scatter(m[:, 0], m[:, 1], m[:, 2], color="#F9E79F", s=18, marker="o",
                       edgecolors="#B7950B", linewidths=0.5)
    # 外擴禁區 (裁到安全盒) 的地板輪廓
    for z in getattr(field, "zones_clipped", []):
        zz = np.vstack([z, z[:1]])
        ax.plot(zz[:, 0], zz[:, 1], np.full(len(zz), floor_z + 0.01),
                color="#F39C12" if enabled else "#7F8C8D", lw=0.9, ls="--", alpha=0.8)
    ax.scatter([], [], [], color="#E74C3C", s=40, marker="s", label="obstacle (繞開)")
    if over:
        ax.scatter([], [], [], color="#F39C12", s=40, marker="s", label="obstacle (越過)")


def _draw_takeoff(ax, cfg: Dict):
    """畫起飛點：地面上的十字 + 到起飛高度的虛線 (飛機原地爬升的那一段)。"""
    try:
        tx, ty = takeoff_point(cfg)
    except ValueError:
        return
    alt = float((cfg.get("waypoints", {}) or {}).get("takeoff_alt", 1.0))
    ax.plot([tx, tx], [ty, ty], [0.0, alt], color="#58D68D", lw=1.2, ls=":", alpha=0.9)
    ax.scatter([tx], [ty], [0.0], color="#58D68D", s=60, marker="P", label="takeoff")


def _style_dark(fig, ax):
    """深色主題 + 把 3D 立方體放到最大 (減少 matplotlib 3D 預設大量留白)。"""
    bg = "#0E1318"
    fg = "#C9D4DE"
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    # 讓 3D 軸幾乎填滿整個 figure (預設四周留白很大 -> 立方體看起來很小)
    # 右側留一點給 z 軸刻度、底部留一點給軸標題，避免被裁切
    ax.set_position([0.0, 0.01, 0.95, 0.93])
    pane = (1, 1, 1, 0.03)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis.label.set_color(fg)
        axis.line.set_color("#33424F")
    ax.tick_params(colors=fg, labelsize=8)
    for g in (ax.xaxis, ax.yaxis, ax.zaxis):
        g._axinfo["grid"]["color"] = (0.4, 0.46, 0.52, 0.25)


def draw_trajectory(fig, plan: PlanResult, cfg: Dict, title: Optional[str] = None):
    """把軌跡畫進給定的 matplotlib Figure。"""
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    t = plan.trajectory
    box = plan.box

    _draw_box(ax, box)
    _draw_obstacles(ax, getattr(plan, "obstacles", None))
    _draw_takeoff(ax, cfg)

    # 依時間上色的軌跡
    ax.scatter(t.x, t.y, t.z, c=t.t, cmap="viridis", s=4, alpha=0.9)
    ax.plot(t.x, t.y, t.z, color="#2E86C1", lw=0.4, alpha=0.4)

    # 起點 / 終點
    ax.scatter([t.x[0]], [t.y[0]], [t.z[0]], color="#2ECC71", s=70, marker="o", label="start")
    ax.scatter([t.x[-1]], [t.y[-1]], [t.z[-1]], color="#E74C3C", s=70, marker="X", label="end")

    # 原地轉頭點 (waypoints.turn_in_place): AUTO 任務在這些點停下、轉頭朝下一段再前進
    tp = ((t.meta.get("auto_estimate") or {}).get("compact") or {}).get("turn_points") or []
    if tp:
        p = np.asarray(tp, dtype=float)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color="#F5B041", s=45, marker="^",
                   edgecolors="#7E5109", linewidths=0.6, label="turn in place")

    ax.set_xlabel("X 東 (m)")
    ax.set_ylabel("Y 北 (m)")
    ax.set_zlabel("Z 高 (m)")
    ax.set_xlim(box.x_min - 0.2, box.x_max + 0.2)
    ax.set_ylim(box.y_min - 0.2, box.y_max + 0.2)
    ax.set_zlim(0, cfg["volume"]["size_z"])
    try:
        ax.set_box_aspect(
            (box.x_max - box.x_min, box.y_max - box.y_min, cfg["volume"]["size_z"]),
            zoom=1.25,  # 放大立方體在畫面中的占比
        )
    except TypeError:
        # 舊版 matplotlib 沒有 zoom 參數
        ax.set_box_aspect((box.x_max - box.x_min, box.y_max - box.y_min, cfg["volume"]["size_z"]))
    except Exception:
        pass

    _style_dark(fig, ax)

    # 工時依 flight.duration_basis: auto -> AUTO 航線時間 (只飛 AUTO 的基準); guided -> 軌跡時間
    dur_txt = f"{basis_duration(t, cfg):.0f}s"
    if duration_basis(cfg) == "auto" and (t.meta.get("auto_estimate") or {}).get("compact"):
        dur_txt = f"AUTO {dur_txt}"
    ttl = title or (
        f"{plan.pattern.display_name}  |  {plan.laps} 圈  |  "
        f"{dur_txt}  |  長度 {t.path_length_3d:.0f}m  |  "
        f"vmax {t.max_speed:.2f} m/s"
    )
    ax.set_title(ttl, fontsize=11, color="#E6EDF3", pad=2)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.2, labelcolor="#C9D4DE")
    return ax


def save_png(plan: PlanResult, cfg: Dict, path: str, dpi: int = 130):
    """存一張 3D 預覽 PNG (headless)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    draw_trajectory(fig, plan, cfg)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _setup_cjk_font():
    """盡量讓中文標籤能顯示 (找不到也不致命)。匯入時執行一次。"""
    import matplotlib

    for name in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC",
                 "Noto Sans CJK TC", "Arial Unicode MS"]:
        try:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue


_setup_cjk_font()
