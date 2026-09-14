"""障礙物 / 避障 / VRPN 客戶端 smoke test。
用法: python tests/obstacles_smoke.py
"""

import math
import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:  # Windows 主控台預設非 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

from core import patterns  # noqa: E402
from core.config import deep_update, load_config  # noqa: E402
from core.geometry import (  # noqa: E402
    SafeBox, enu_to_latlon, safe_box_from_config, takeoff_is_origin, takeoff_point,
)
from core.obstacles import (  # noqa: E402
    ObstacleField, convex_hull, inflate, obstacle_from_item, points_inside,
    segments_enter, signed_distance,
)
from core.planner import plan  # noqa: E402
from core.trajectory import mission_polyline  # noqa: E402
from core import vrpn_client as vc  # noqa: E402


def _densify(pts, step=0.02):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def _min_dist(lap, obstacle):
    return float(signed_distance(_densify(np.asarray(lap, float)), obstacle.footprint).min())


def test_geometry() -> int:
    fails = 0
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    if len(convex_hull(np.vstack([sq, [[0.5, 0.5], [0.5, 0.0]]]))) != 4:
        print("!! convex_hull 未去掉內點 / 共線點"); fails += 1
    inside = points_inside(np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 0.5]]), sq)
    if inside.tolist() != [True, False, False]:
        print(f"!! points_inside 邊界應不算內部: {inside.tolist()}"); fails += 1
    # 穿越 / 沿邊 / 擦頂點
    P = np.array([[-1, 0.5], [0, 0], [-1, 1], [0.5, 0.5]], float)
    Q = np.array([[2, 0.5], [1, 0], [1, 2], [0.5, 2.0]], float)
    ent = segments_enter(P, Q, sq).tolist()
    if ent != [True, False, False, True]:
        print(f"!! segments_enter 穿越/沿邊/擦角判斷錯: {ent}"); fails += 1
    # 外擴: 對齊的直邊恰為 r, 圓角 <= r/cos(pi/8)
    z = inflate(sq, 0.5, 8, 0.0)
    d = signed_distance(z, sq)
    if d.min() < 0.5 - 1e-9 or d.max() > 0.5 / math.cos(math.pi / 8) + 1e-9:
        print(f"!! inflate 距離範圍 {d.min():.4f}~{d.max():.4f} 不在 [0.5, 0.541]"); fails += 1
    mid = np.array([[0.5, 1.5 - 1e-6]])              # 上邊中點的外擴邊 (應剛好在邊界上)
    if points_inside(mid, z)[0] is False and signed_distance(mid, z)[0] > 1e-5:
        print("!! inflate 對齊邊的外擴距離不是 r"); fails += 1
    if not fails:
        print("幾何工具 (凸包 / 內外 / 穿越 / 外擴) OK")
    return fails


def test_avoid_lap() -> int:
    fails = 0
    box = SafeBox(-2, 2, -2, 2, 0.6, 2.4)
    o = obstacle_from_item({"name": "b1", "x": 0.0, "y": 0.0, "size": [0.6, 0.4, 0.5]})
    f = ObstacleField([o], box, 0.5, 2)
    # 往返線穿箱 -> 改道 2 段, 離箱 >= 0.5, 仍閉合
    lap = np.array([[-2, 0], [2, 0], [-2, 0]], float)
    lap2, info = f.avoid_lap(lap, False)
    if info.detours != 2 or not np.allclose(lap2[0], lap2[-1]):
        print(f"!! 往返改道: {info.as_dict()}"); fails += 1
    if _min_dist(lap2, o) < 0.5 - 1e-6:
        print(f"!! 往返改道後離箱 {_min_dist(lap2, o):.3f} < 0.5"); fails += 1
    if len(lap2) > 9:
        print(f"!! 往返改道航點過多 ({len(lap2)})"); fails += 1
    if not all(box.contains(x, y, 1.0) for x, y in lap2):
        print("!! 往返改道出安全盒"); fails += 1
    # 矩形角落在箱子 (貼牆角) 裡 -> 角被推到禁區邊界, 不可貼牆滑過
    o2 = obstacle_from_item({"name": "b2", "x": 1.9, "y": 1.9, "size": [0.6, 0.6, 0.5]})
    f2 = ObstacleField([o2], box, 0.5, 2)
    rect = np.array([[-2, 2], [2, 2], [2, -2], [-2, -2], [-2, 2]], float)
    r2, info2 = f2.avoid_lap(rect, False)
    if info2.moved != 1 or _min_dist(r2, o2) < 0.5 - 1e-6 or not all(box.contains(x, y, 1.0) for x, y in r2):
        print(f"!! 矩形角在箱內: {info2.as_dict()} 離箱 {_min_dist(r2, o2):.3f}"); fails += 1
    # 圓 (密集曲線) 起點在箱內: 起點被推出, 內部密集點拿掉, 離箱 >= 0.5, 閉合
    th = np.linspace(0, 2 * np.pi, 721)
    circ = np.column_stack([2 * np.cos(th), 2 * np.sin(th)])
    o3 = obstacle_from_item({"name": "b3", "x": 2.0, "y": 0.0, "size": [0.5, 0.5, 0.6]})
    f3 = ObstacleField([o3], box, 0.5, 2)
    c3, info3 = f3.avoid_lap(circ, True)
    if info3.moved != 1 or info3.removed == 0 or not np.allclose(c3[0], c3[-1]):
        print(f"!! 圓起點在箱內: {info3.as_dict()}"); fails += 1
    if _min_dist(c3, o3) < 0.5 - 1e-6:
        print(f"!! 圓改道後離箱 {_min_dist(c3, o3):.3f} < 0.5"); fails += 1
    # 兩箱相鄰 -> 禁區合併成 1 塊, 路徑不從縫中穿
    o4 = obstacle_from_item({"name": "b4", "x": 0.0, "y": 0.5, "size": [0.5, 0.5, 0.5]})
    o5 = obstacle_from_item({"name": "b5", "x": 0.0, "y": -0.5, "size": [0.5, 0.5, 0.5]})
    f45 = ObstacleField([o4, o5], box, 0.5, 2)
    l45, _ = f45.avoid_lap(lap, False)
    if len(f45.zones) != 1 or min(_min_dist(l45, o4), _min_dist(l45, o5)) < 0.5 - 1e-6:
        print(f"!! 相鄰兩箱未合併 / 從縫中穿過 (zones={len(f45.zones)})"); fails += 1
    # 牆到牆的障礙把房間切成兩半 -> 另一半從 HOME 到不了: 該側頂點移到可達側的禁區邊界 (不丟例外,
    # unreachable 回報給安全檢查警告), 結果整段路徑都在同一側且不穿禁區
    o7 = obstacle_from_item({"name": "wall", "x": 0, "y": 0, "size": [0.2, 3.5, 1.0]})
    f7 = ObstacleField([o7], box, 0.5, 2)
    l7, info7 = f7.avoid_lap(lap, False)
    if info7.failed != 0 or info7.unreachable != 1 or info7.moved != 1:
        print(f"!! 牆到牆障礙處理錯: {info7.as_dict()}"); fails += 1
    if len(set(np.sign(l7[:, 0]).tolist())) != 1 or _min_dist(l7, o7) < 0.5 - 1e-6:
        print(f"!! 牆到牆障礙: 路徑跨到另一側 / 穿禁區: {np.round(l7, 2).tolist()}"); fails += 1
    # 光球座標 (points) 種類 + 進場路徑 (從禁區內出發也要有路)
    o6 = obstacle_from_item({"name": "m", "kind": "points",
                             "points": [[1, 1, 0.5], [1.4, 1, 0.5], [1.4, 1.3, 0.52], [1, 1.3, 0.5]]})
    if abs(o6.z_top - 0.52) > 1e-9 or len(o6.footprint) != 4:
        print(f"!! points 障礙物解析錯: top={o6.z_top} fp={len(o6.footprint)}"); fails += 1
    p = f3.free_path((2.0, 0.0), (0.0, 0.0))
    if p is None or f3.zone_of(p[1]) >= 0 or len(p) < 3:
        print(f"!! 從禁區內出發的 free_path 錯: {p}"); fails += 1
    # 避障停用 -> 不改道
    f_off = ObstacleField([o], box, 0.5, 2, enabled=False)
    l_off, info_off = f_off.avoid_lap(lap, False)
    if info_off.touched or len(l_off) != 3:
        print("!! enabled=False 仍改道"); fails += 1
    if not fails:
        print("單圈避障 (改道 / 推頂點 / 合併 / 繞不開回報) OK")
    return fails


def test_planner(cfg) -> int:
    fails = 0
    items = [
        {"name": "box1", "kind": "box", "x": 0.9, "y": 0.0, "size": [0.6, 0.4, 0.5]},
        {"name": "box2", "kind": "box", "x": -1.3, "y": 1.3, "size": [0.5, 0.5, 0.6], "yaw_deg": 30},
        {"name": "m1", "kind": "points",
         "points": [[-1.5, -1.0, 0.4], [-1.1, -1.0, 0.4], [-1.1, -0.7, 0.42], [-1.5, -0.7, 0.4]]},
    ]
    # 這一段驗證「水平繞開」: 一律 around (箱子都不高, 預設 over 會改成拉高越過 -> 見 test_over)
    cfg_o = deep_update(cfg, {"obstacles": {"items": items, "low_mode": "around"}})
    box = safe_box_from_config(cfg_o)
    for key, _ in patterns.list_patterns():
        pr = plan(key, cfg_o)
        t = pr.trajectory
        if pr.obstacles is None or len(pr.obstacles.obstacles) != 3:
            print(f"!! {key}: PlanResult 缺障礙物"); fails += 1
            continue
        # GUIDED 軌跡與 AUTO 航點折線 (含進場) 都不可進障礙物
        d_traj = pr.obstacles.distances(np.column_stack([t.x, t.y]))
        if min(d_traj) < 0.5 - 0.02:
            print(f"!! {key}: 軌跡離障礙物 {min(d_traj):.3f} < 0.48"); fails += 1
        mp = mission_polyline(t, cfg_o, pr.pattern.is_smooth, lap_xy=pr.pattern.lap_xy, laps=pr.laps,
                              mode="compact", repeatable=pr.pattern.repeatable, field=pr.obstacles)
        mx, my, mz = mp.flown_with_approach(float(cfg_o["waypoints"]["takeoff_alt"]))
        d_mis = pr.obstacles.distances(_densify(np.column_stack([mx, my]), 0.05))
        if min(d_mis) < 0:
            print(f"!! {key}: AUTO 航點折線穿過障礙物 ({min(d_mis):.3f})"); fails += 1
        if any(e for e in pr.report.errors):
            print(f"!! {key}: 安全檢查 ERROR: {pr.report.errors}"); fails += 1
        if not all(box.contains(x, y, z) for x, y, z in zip(t.x, t.y, t.z)):
            print(f"!! {key}: 有點超出安全盒"); fails += 1
        if not (abs(t.x[0] - t.x[-1]) < 0.3 and abs(t.y[0] - t.y[-1]) < 0.3):
            print(f"!! {key}: 起終點未閉合"); fails += 1
        est = t.meta["auto_estimate"]["compact"]
        if est["n_items"] + 4 > int(cfg_o["waypoints"]["fc_budget"]):
            print(f"!! {key}: compact 超過預算 ({est['n_items']} + 4)"); fails += 1
        if key in ("reciprocate", "rectangle", "zigzag") and not (pr.avoid and pr.avoid.touched):
            print(f"!! {key}: 直線型 pattern 應有改道 / 推頂點"); fails += 1
        ob = t.meta.get("obstacles") or {}
        if ob.get("count") != 3:
            print(f"!! {key}: meta.obstacles 缺"); fails += 1
    # 匯出: 進場航點在 DO_JUMP block 之前, DO_JUMP 回跳到 block 第一點
    from io_export.waypoints import trajectory_to_waypoints, CMD_DO_JUMP
    cfg_a = deep_update(cfg_o, {"obstacles": {"items": [
        {"name": "front", "kind": "box", "x": 0.0, "y": 1.2, "size": [0.8, 0.3, 0.5]}]}})
    pr = plan("rectangle", cfg_a)
    lines, n_nav = trajectory_to_waypoints(pr, cfg_a, mode="compact")
    est = pr.trajectory.meta["auto_estimate"]["compact"]
    rows = [ln.split("\t") for ln in lines[1:]]
    jumps = [r for r in rows if int(r[3]) == CMD_DO_JUMP]
    if est["n_approach"] < 1:
        print(f"!! 進場段應插入繞障航點 (n_approach={est['n_approach']})"); fails += 1
    if not jumps:
        print("!! 匯出缺 DO_JUMP"); fails += 1
    else:
        first_seq = int(float(jumps[0][4]))
        if first_seq != 3 + est["n_approach"]:
            print(f"!! DO_JUMP 回跳 seq {first_seq} != 固定 3 + 進場 {est['n_approach']}"); fails += 1
    if n_nav != est["n_wp"]:
        print(f"!! 匯出 nav 數 {n_nav} != 估時 n_wp {est['n_wp']}"); fails += 1
    # 沒有障礙物時完全不受影響 (迴歸)
    pr0 = plan("circle", cfg)
    if pr0.avoid is not None or pr0.trajectory.meta["auto_estimate"]["compact"]["n_approach"] != 0:
        print("!! 無障礙物時仍有避障 / 進場點"); fails += 1
    # 避障停用 -> 穿過時 ERROR
    pr_off = plan("reciprocate", deep_update(cfg_o, {"obstacles": {"enabled": False}}))
    if pr_off.report.ok or not any("穿過障礙物" in e for e in pr_off.report.errors):
        print(f"!! 避障停用且穿過障礙物應 ERROR: {pr_off.report.errors}"); fails += 1
    # 起飛點在障礙物內 -> ERROR
    pr_h = plan("circle", deep_update(cfg, {"obstacles": {"items": [
        {"name": "home_box", "kind": "box", "x": 0.0, "y": 0.0, "size": [0.4, 0.4, 0.4]}]}}))
    if not any("HOME" in e for e in pr_h.report.errors):
        print(f"!! HOME 在障礙物內應 ERROR: {pr_h.report.errors}"); fails += 1
    if not fails:
        print("planner 整合 (各 pattern 避障 / 進場航點 / 匯出 / 安全檢查) OK")
    return fails


def _densify_xyz(x, y, z, step=0.05):
    pts = np.column_stack([x, y, z]); out = [pts[:1]]
    for i in range(len(pts) - 1):
        n = max(1, int(np.ceil(np.hypot(*(pts[i + 1, :2] - pts[i, :2])) / step)))
        out.append(pts[i] + (pts[i + 1] - pts[i]) * (np.arange(1, n + 1) / n)[:, None])
    return np.vstack(out)


def test_over(cfg) -> int:
    """低矮箱子 (箱頂 <= 0.6 m) 用局部拉高越過, 高的仍繞開。"""
    fails = 0
    from core.obstacles import points_inside
    low = {"name": "low", "kind": "box", "x": 0.9, "y": 0.0, "z_bottom": 0.0, "size": [0.5, 0.4, 0.5]}
    tall = {"name": "tall", "kind": "box", "x": -1.3, "y": 1.3, "z_bottom": 0.0, "size": [0.4, 0.4, 1.0]}
    cfg_o = deep_update(cfg, {"obstacles": {"items": [low, tall]}})     # 預設 low_mode: over, 門檻 0.6
    f = cfg_o["flight"]
    for key, _ in patterns.list_patterns():
        pr = plan(key, cfg_o)
        t = pr.trajectory
        fld = pr.obstacles
        if [fld.obstacles[z.index].name for z in fld.over] != ["low"] or len(fld.zones) != 1:
            print(f"!! {key}: 越過/繞開分類錯 over={[z.index for z in fld.over]} zones={len(fld.zones)}"); fails += 1
            continue
        z = fld.over[0]
        if abs(z.z_req - 1.0) > 1e-9:
            print(f"!! {key}: z_req {z.z_req} != 箱頂 0.5 + 0.5"); fails += 1
        inside = points_inside(np.column_stack([t.x, t.y]), z.poly)
        mp = mission_polyline(t, cfg_o, pr.pattern.is_smooth, lap_xy=pr.pattern.lap_xy, laps=pr.laps,
                              mode="compact", repeatable=pr.pattern.repeatable, field=fld)
        mx, my, mz = mp.flown_with_approach(float(cfg_o["waypoints"]["takeoff_alt"]))
        d = _densify_xyz(mx, my, mz)
        ins_m = points_inside(d[:, :2], z.poly)
        if inside.any():
            zmin = float(t.z[inside].min())
            if zmin < z.z_req - 0.02:
                print(f"!! {key}: 軌跡在越箱禁區內最低 {zmin:.3f} < {z.z_req}"); fails += 1
            if not t.meta.get("z_anchors_s"):
                print(f"!! {key}: 有越箱平台卻沒有航點錨點"); fails += 1
            if not ins_m.any():
                print(f"!! {key}: AUTO 航點折線沒經過越箱禁區 (軌跡有)"); fails += 1
            elif float(d[ins_m, 2].min()) < z.z_req - 0.10:
                print(f"!! {key}: AUTO 航點折線在越箱禁區內最低 {float(d[ins_m, 2].min()):.3f} < {z.z_req}-0.10"); fails += 1
        # 高的箱子仍水平繞開 (軌跡離 tall >= 0.48)
        d_tall = fld.distances(np.column_stack([t.x, t.y]))[1]
        if d_tall < 0.5 - 0.02:
            print(f"!! {key}: 高箱子未繞開 (距離 {d_tall:.3f})"); fails += 1
        # 垂直速度 / 3D 速度仍在限制內 (斜坡用 auto 速度)
        if float(t.vz.max()) > float(f.get("speed_up", 1.0)) + 0.05 or -float(t.vz.min()) > float(f.get("speed_down", 0.6)) + 0.05:
            print(f"!! {key}: 越箱斜坡垂直速度超限 {t.vz.max():.2f}/{t.vz.min():.2f}"); fails += 1
        if t.max_speed > float(f.get("max_speed", 1.0)) + 1e-3:
            print(f"!! {key}: 越箱斜坡 3D 速度超限 {t.max_speed:.2f}"); fails += 1
        if not pr.report.ok:
            print(f"!! {key}: 安全檢查 ERROR: {pr.report.errors}"); fails += 1
        if not all(pr.box.contains(x, y, zz) for x, y, zz in zip(t.x, t.y, t.z)):
            print(f"!! {key}: 有點超出安全盒"); fails += 1
    # 往返線直接穿過低箱: 不改道, 平台區高度 = 1.0, 匯出航點含平台起訖 (z 恰為 1.0)
    pr = plan("reciprocate", cfg_o)
    if pr.avoid is None or pr.avoid.detours != 0:
        print(f"!! 往返穿低箱應不改道: {pr.avoid.as_dict() if pr.avoid else None}"); fails += 1
    from io_export.waypoints import trajectory_to_waypoints
    lines, _ = trajectory_to_waypoints(pr, cfg_o, mode="compact")
    rows = [ln.split("\t") for ln in lines[1:]]
    alts = [float(r[10]) for r in rows if int(r[3]) == 16]
    if sum(1 for a in alts if abs(a - 1.0) < 1e-6) < 2:
        print(f"!! 匯出航點缺平台起訖 (高度 1.00 的航點 < 2): {alts}"); fails += 1
    # 門檻: 箱頂 0.7 > 0.6 -> 繞開; 拉高會超過天花板邊界 -> 繞開; around 模式 -> 全繞
    pr_h = plan("reciprocate", deep_update(cfg, {"obstacles": {"items": [dict(low, size=[0.5, 0.4, 0.7])]}}))
    if pr_h.obstacles.over or not (pr_h.avoid and pr_h.avoid.detours):
        print("!! 箱頂 0.7 m 應改為繞開"); fails += 1
    pr_c = plan("reciprocate", deep_update(cfg, {"obstacles": {"items": [low], "vertical_clearance": 2.0}}))
    if pr_c.obstacles.over:
        print("!! 拉高後超過天花板邊界應改為繞開"); fails += 1
    pr_a = plan("reciprocate", deep_update(cfg_o, {"obstacles": {"low_mode": "around"}}))
    if pr_a.obstacles.over or not (pr_a.avoid and pr_a.avoid.detours):
        print("!! low_mode=around 應全部繞開"); fails += 1
    # 進場段: 從 HOME 到第一個航點也要繞開低箱 (進場高度低)
    lowh = {"name": "lowh", "kind": "box", "x": 1.0, "y": 0.0, "z_bottom": 0.0, "size": [0.4, 0.4, 0.5]}
    pr_p = plan("rectangle", deep_update(cfg, {"obstacles": {"items": [lowh]}}))
    est = pr_p.trajectory.meta["auto_estimate"]["compact"]
    if pr_p.obstacles.approach_field is pr_p.obstacles or not pr_p.obstacles.approach_field.zones:
        print("!! approach_field 應把低箱當水平禁區"); fails += 1
    if not fails:
        print("越過低矮箱子 (局部拉高 / 錨點 / 分類 / 進場) OK")
    return fails


# ---------------------------------------------------------------------------
# 假 VRPN 伺服器: 依 vrpn_Connection 協定送 cookie + 描述 + Pos_Quat, 驗證客戶端解析
# ---------------------------------------------------------------------------
def _fake_vrpn_server(port_holder, bodies, n_msgs=5, sensor=0):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_holder.append(srv.getsockname()[1])
    srv.settimeout(5.0)
    try:
        conn, _ = srv.accept()
        conn.settimeout(3.0)
        conn.sendall(vc.make_cookie())
        cookie = b""
        while len(cookie) < vc.VRPN_COOKIE_SIZE:
            chunk = conn.recv(vc.VRPN_COOKIE_SIZE - len(cookie))
            if not chunk:
                return
            cookie += chunk
        port_holder.append(cookie)
        # 型別描述 (id 3 = Pos_Quat, 另外幾個無關型別), sender 描述 (每個 rigid body 一個)
        conn.sendall(vc.pack_description(vc.MSG_TYPE_DESCRIPTION, 0, "vrpn_Base ping_message"))
        conn.sendall(vc.pack_description(vc.MSG_TYPE_DESCRIPTION, 3, vc.TRACKER_POS_QUAT))
        conn.sendall(vc.pack_description(vc.MSG_TYPE_DESCRIPTION, 4, "vrpn_Tracker Velocity"))
        for i, (name, _, _) in enumerate(bodies):
            conn.sendall(vc.pack_description(vc.MSG_SENDER_DESCRIPTION, 10 + i, name))
        for k in range(n_msgs):
            for i, (name, pos, quat) in enumerate(bodies):
                p = [pos[0] + 0.001 * k, pos[1], pos[2]]
                conn.sendall(vc.pack_message(3, 10 + i, vc.pack_pos_quat(sensor, p, quat),
                                             stamp=100.0 + k))
                conn.sendall(vc.pack_message(4, 10 + i, b"\0" * 64, stamp=100.0 + k))   # 無關訊息
            time.sleep(0.05)
        time.sleep(0.3)
        conn.close()
    except Exception as e:  # noqa: BLE001
        port_holder.append(e)
    finally:
        srv.close()


def test_vrpn() -> int:
    fails = 0
    # 座標轉換: Motive Y-up (x, y_up, z) -> ENU (x, -z, y)
    M = vc.axes_matrix("y_up")
    enu = vc.to_enu([1.0, 2.0, 3.0], M)
    if not np.allclose(enu, [1.0, -3.0, 2.0]):
        print(f"!! y_up 轉換錯: {enu}"); fails += 1
    if not np.allclose(vc.to_enu([1, 2, 3], vc.axes_matrix("z_up")), [1, 2, 3]):
        print("!! z_up 應為恆等"); fails += 1
    # 偏航: 在 Y-up 中繞 Y 轉 +30° 的四元數 -> ENU 中繞 z 轉 +30°
    a = math.radians(30)
    q = [0.0, math.sin(a / 2), 0.0, math.cos(a / 2)]
    yaw = vc.yaw_enu(q, M)
    if abs(math.degrees(yaw) - 30) > 1e-6:
        print(f"!! 偏航轉換錯: {math.degrees(yaw):.2f} != 30"); fails += 1
    # 假伺服器 -> read_trackers
    bodies = [("box1", [1.0, 0.52, -0.5], q), ("box2", [-1.2, 0.5, 0.8], [0, 0, 0, 1])]
    holder = []
    th = threading.Thread(target=_fake_vrpn_server, args=(holder, bodies), daemon=True)
    th.start()
    t0 = time.time()
    while not holder and time.time() - t0 < 3:
        time.sleep(0.01)
    port = holder[0]
    samples = vc.read_trackers(f"127.0.0.1:{port}", seconds=1.0)
    th.join(timeout=5)
    if len(holder) < 2 or not isinstance(holder[1], (bytes, bytearray)) or not holder[1].startswith(b"vrpn: ver. 07"):
        print(f"!! 客戶端 cookie 錯: {holder[1:] if len(holder) > 1 else holder}"); fails += 1
    if set(samples) != {"box1", "box2"}:
        print(f"!! read_trackers 取到 {sorted(samples)}"); fails += 1
    else:
        s1 = samples["box1"]
        if s1.count != 5 or not np.allclose(s1.pos, [1.004, 0.52, -0.5]) or not np.allclose(s1.quat, q):
            print(f"!! box1 樣本錯: count={s1.count} pos={s1.pos} quat={s1.quat}"); fails += 1
        if abs(s1.stamp - 104.0) > 1e-3:
            print(f"!! 時間戳錯: {s1.stamp}"); fails += 1
        items = vc.trackers_to_items(samples, {"axes": "y_up", "mode": "each_box",
                                               "box_size": [0.6, 0.4, 0.5], "pivot": "top",
                                               "orientation": "fixed"})
        b1 = next(it for it in items if it["name"] == "box1")
        if not (abs(b1["x"] - 1.004) < 1e-6 and abs(b1["y"] - 0.5) < 1e-6
                and abs(b1["z_bottom"] - 0.02) < 1e-6 and abs(b1["yaw_deg"] - 30) < 1e-3
                and b1["source"] == "vrpn" and b1["size"] == [0.6, 0.4, 0.5]):
            print(f"!! trackers_to_items (each_box, fixed) 錯: {b1}"); fails += 1
        o = obstacle_from_item(b1)
        if abs(o.z_top - 0.52) > 1e-6 or abs(o.z_bottom - 0.02) > 1e-6:
            print(f"!! pivot=top: 頂 z {o.z_top} != 0.52 (量到的樞紐高) / 底 {o.z_bottom} != 0.02"); fails += 1
        # 箱底低於地板時: 保留量到的頂面高度, 高度縮短
        low = vc.trackers_to_items({"box2": samples["box2"]}, {"axes": "y_up", "mode": "each_box",
                                                              "box_size": [0.6, 0.4, 0.5], "pivot": "top",
                                                              "orientation": "fixed"})[0]
        if not (abs(low["z_bottom"]) < 1e-9 and abs(low["size"][2] - 0.5) < 1e-6):   # box2 頂 z = 0.5
            print(f"!! 箱底夾到地板錯: {low}"); fails += 1
        # 同一種箱子有躺有立: 用頂面高度判斷垂直邊 (35×40×50 cm 箱)
        dims = [0.35, 0.40, 0.50]
        sx, sy, sz, _ = vc.box_dims_from_height(dims, 0.366)              # 躺: 35 垂直
        if not (abs(sz - 0.35) < 1e-9 and abs(sx - 0.5) < 1e-9 and abs(sy - 0.5) < 1e-9):
            print(f"!! 躺放 auto_square 錯: {sx, sy, sz}"); fails += 1
        sx, sy, sz, _ = vc.box_dims_from_height(dims, 0.51)               # 立: 50 垂直
        if not (abs(sz - 0.5) < 1e-9 and abs(sx - 0.4) < 1e-9 and abs(sy - 0.4) < 1e-9):
            print(f"!! 立放 auto_square 錯: {sx, sy, sz}"); fails += 1
        sx, sy, sz, _ = vc.box_dims_from_height(dims, 0.41, "auto_long_y")  # 40 垂直, 長邊 0.5 沿 y
        if not (abs(sz - 0.4) < 1e-9 and abs(sx - 0.35) < 1e-9 and abs(sy - 0.5) < 1e-9):
            print(f"!! auto_long_y 錯: {sx, sy, sz}"); fails += 1
        sx, sy, sz, note = vc.box_dims_from_height(dims, 1.42)            # 不像任一邊 -> 視為立放
        if not (abs(sz - 0.5) < 1e-9 and "不像" in note):
            print(f"!! 高度不合理時應退回立放: {sx, sy, sz, note}"); fails += 1
        auto_items = vc.trackers_to_items(samples, {"axes": "y_up", "mode": "each_box", "box_size": dims,
                                                    "orientation": "auto_square", "pivot": "top",
                                                    "sizes": {"box2": [0.1, 0.2, 0.3]}})
        a1 = next(it for it in auto_items if it["name"] == "box1")       # 頂面 z 0.52 -> 50 垂直 -> 底面 0.4²
        a2 = next(it for it in auto_items if it["name"] == "box2")       # 個別指定
        if not (a1["size"] == [0.4, 0.4, 0.5] and abs(a1["z_bottom"] - 0.02) < 1e-6 and "立放" in a1["note"]
                and a2["size"] == [0.1, 0.2, 0.3] and "指定" in a2["note"]):
            print(f"!! auto orientation / sizes 覆寫錯: {a1['size']} {a2['size']}"); fails += 1
        hull = vc.trackers_to_items(samples, {"axes": "y_up", "mode": "hull"})
        if len(hull) != 1 or hull[0]["kind"] != "points" or len(hull[0]["points"]) != 2:
            print(f"!! trackers_to_items (hull) 錯: {hull}"); fails += 1
        # 只讀指定名稱
        holder2 = []
        th2 = threading.Thread(target=_fake_vrpn_server, args=(holder2, bodies, 2), daemon=True)
        th2.start()
        while not holder2:
            time.sleep(0.01)
        only = vc.read_trackers(f"127.0.0.1:{holder2[0]}", seconds=0.8, names=["box2"])
        th2.join(timeout=5)
        if set(only) != {"box2"}:
            print(f"!! names 過濾錯: {sorted(only)}"); fails += 1
    # 連不上 -> VRPNError
    try:
        vc.read_trackers("127.0.0.1:1", seconds=0.2, connect_timeout=0.5)
        print("!! 連不上應丟 VRPNError"); fails += 1
    except vc.VRPNError:
        pass
    if not fails:
        print("VRPN 客戶端 (協定解析 / 座標轉換 / 障礙物轉換) OK")
    return fails


def test_takeoff_point(cfg) -> int:
    """起飛點: 解析 / 進場段 / 估時 / 匯出 HOME / 爬升檢查 / 可達性錨點 / VRPN 分離飛機。"""
    fails = 0
    from core.obstacles import ObstacleField, obstacle_from_item
    from io_export.waypoints import trajectory_to_waypoints

    # 1) 解析器
    def tp(v):
        return takeoff_point({"waypoints": {"takeoff_point": v}})
    for v, want in (("origin", (0.0, 0.0)), (None, (0.0, 0.0)), ([0.4, -0.3], (0.4, -0.3)),
                    ((1, 2), (1.0, 2.0)), ({"x": 1, "y": 2}, (1.0, 2.0)), ("0.5 -0.2", (0.5, -0.2))):
        if tp(v) != want:
            print(f"!! takeoff_point({v!r}) = {tp(v)} != {want}"); fails += 1
    for bad in ("nope", [1], "a b", [1, 2, 3, 4]):
        try:
            tp(bad)
            if bad != [1, 2, 3, 4]:      # 多給的元素允許截斷, 其餘必須擋下
                print(f"!! takeoff_point({bad!r}) 應丟 ValueError"); fails += 1
        except ValueError:
            pass
    if not takeoff_is_origin(cfg) or takeoff_is_origin({"waypoints": {"takeoff_point": [1, 0]}}):
        print("!! takeoff_is_origin 判斷錯"); fails += 1

    # 2) 預設 (origin) 與「沒有這個鍵」完全等價 —— 舊設定檔不能有任何行為改變
    base = deep_update(cfg, {"obstacles": {"items": []}})
    no_key = deep_update(base, {})
    no_key["waypoints"] = {k: v for k, v in no_key["waypoints"].items() if k != "takeoff_point"}
    a, _ = trajectory_to_waypoints(plan("rectangle", no_key), no_key, mode="compact")
    b, _ = trajectory_to_waypoints(plan("rectangle", deep_update(base, {"waypoints": {"takeoff_point": "origin"}})),
                                   base, mode="compact")
    if a != b:
        print("!! takeoff_point=origin 與未設定的匯出結果不同"); fails += 1

    # 3) 起飛點與第一個航點之間有箱子 -> 進場段插點, 且第一段從起飛點算起
    blocker = {"name": "mid", "kind": "box", "x": 0.0, "y": 0.0, "z_bottom": 0.0, "size": [0.4, 1.6, 1.2]}
    cfg_t = deep_update(cfg, {"obstacles": {"items": [blocker]},
                              "waypoints": {"takeoff_point": [1.5, 0.0]}})
    pr = plan("rectangle", cfg_t)
    mp = mission_polyline(pr.trajectory, cfg_t, pr.pattern.is_smooth, lap_xy=pr.pattern.lap_xy,
                          laps=pr.laps, mode="compact", repeatable=pr.pattern.repeatable,
                          field=pr.obstacles)
    if tuple(np.round(mp.home, 6)) != (1.5, 0.0):
        print(f"!! MissionPath.home {mp.home} != (1.5, 0.0)"); fails += 1
    mx, my, mz = mp.flown_with_approach(float(cfg_t["waypoints"]["takeoff_alt"]))
    if abs(mx[0] - 1.5) > 1e-9 or abs(my[0]) > 1e-9:
        print(f"!! flown_with_approach 起點 ({mx[0]}, {my[0]}) != 起飛點"); fails += 1
    if mp.n_approach < 1:
        print("!! 起飛點與第一個航點被箱子擋住, 應插進場航點"); fails += 1
    d = pr.obstacles.approach_field.distances(_densify(np.column_stack([mx, my]), 0.05))
    if min(d) < 0:
        print(f"!! 實飛第一段穿過箱子 (最深 {min(d):.3f})"); fails += 1
    # 同一組箱子, 起飛點在原點時進場段應該不同 (證明真的用了起飛點)
    pr0 = plan("rectangle", deep_update(cfg, {"obstacles": {"items": [blocker]}}))
    mp0 = mission_polyline(pr0.trajectory, deep_update(cfg, {"obstacles": {"items": [blocker]}}),
                           pr0.pattern.is_smooth, lap_xy=pr0.pattern.lap_xy, laps=pr0.laps,
                           mode="compact", repeatable=pr0.pattern.repeatable, field=pr0.obstacles)
    if np.allclose(mp0.home, mp.home):
        print("!! 起飛點沒有進到 MissionPath"); fails += 1

    # 4) AUTO 估時用的是實際第一段 (把起飛點拉遠 -> 航線時間變長)
    far = deep_update(cfg, {"waypoints": {"takeoff_point": [1.9, 1.9]}, "flight": {"laps": 3}})
    near = deep_update(cfg, {"waypoints": {"takeoff_point": "origin"}, "flight": {"laps": 3}})
    nav_far = plan("circle", far).trajectory.meta["auto_estimate"]["compact"]["nav_s"]
    nav_near = plan("circle", near).trajectory.meta["auto_estimate"]["compact"]["nav_s"]
    if not (nav_far > nav_near + 0.5):
        print(f"!! 起飛點拉遠後 AUTO 航線時間沒變長 ({nav_near:.1f} -> {nav_far:.1f})"); fails += 1

    # 5) 匯出的 HOME 列 = 起飛點的經緯度; origin 時與假原點逐位相同
    w = cfg["waypoints"]
    pr_h = plan("circle", deep_update(cfg, {"waypoints": {"takeoff_point": [0.8, -0.6]}}))
    lines, _ = trajectory_to_waypoints(pr_h, deep_update(cfg, {"waypoints": {"takeoff_point": [0.8, -0.6]}}),
                                       mode="compact")
    row = lines[1].split("\t")
    lat, lon = enu_to_latlon(0.8, -0.6, float(w["origin_lat"]), float(w["origin_lon"]))
    if abs(float(row[8]) - float(lat)) > 1e-7 or abs(float(row[9]) - float(lon)) > 1e-7:
        print(f"!! HOME 列經緯度 {row[8]},{row[9]} != 起飛點換算 {float(lat)},{float(lon)}"); fails += 1
    lines0, _ = trajectory_to_waypoints(plan("circle", cfg), cfg, mode="compact")
    row0 = lines0[1].split("\t")
    if abs(float(row0[8]) - float(w["origin_lat"])) > 1e-9:
        print(f"!! origin 時 HOME 列應等於假原點 ({row0[8]})"); fails += 1

    # 6) 起飛垂直爬升檢查
    def climb(zb, tk="origin", alt=1.0):
        c = deep_update(cfg, {"obstacles": {"items": [
            {"name": "b", "kind": "box", "x": 0.0, "y": 0.0, "z_bottom": zb, "size": [0.4, 0.4, 0.4]}]},
            "waypoints": {"takeoff_point": tk, "takeoff_alt": alt}})
        return plan("circle", c).report
    r_low = climb(0.0)                    # 地上的箱子 -> 撞
    r_tab = climb(0.92)                   # 桌上的箱子, 底 0.92 < 1.0 -> 撞
    r_hi = climb(1.6)                     # 掛得很高 -> 只警告
    r_away = climb(0.92, tk=[-1.6, -1.6])  # 移開起飛點 -> 沒事
    if not any("原地爬到" in e for e in r_low.errors):
        print(f"!! 地面箱子上方起飛應 ERROR: {r_low.errors}"); fails += 1
    if not any("原地爬到" in e for e in r_tab.errors):
        print(f"!! 桌上箱子 (底 0.92 < 起飛高 1.0) 應 ERROR: {r_tab.errors}"); fails += 1
    if any("原地爬到" in e for e in r_hi.errors) or not any("不擋起飛" in w2 for w2 in r_hi.warnings):
        print(f"!! 高掛物應只警告: errors={r_hi.errors} warns={r_hi.warnings}"); fails += 1
    if any("原地爬到" in e for e in r_away.errors):
        print(f"!! 起飛點移開後不該再報爬升碰撞: {r_away.errors}"); fails += 1
    if not all("HOME" in m for m in (list(r_low.errors) + [w2 for w2 in r_hi.warnings if "不擋起飛" in w2])):
        print("!! 起飛點訊息應保留 HOME 字樣"); fails += 1

    # 7) 可達性錨點跟著起飛點: 牆把房間切兩半時, 路徑留在飛機那一半
    box = SafeBox(-2, 2, -2, 2, 0.6, 2.4)
    wall = obstacle_from_item({"name": "wall", "x": 0, "y": 0, "size": [0.2, 3.5, 1.0]})
    lap = np.array([[-2, -1.5], [2, 1.5], [-2, -1.5]], float)
    sides = []
    for home in ((-1.5, -1.5), (1.5, 1.5)):
        f = ObstacleField([wall], box, 0.5, 2, home=home)
        out, _ = f.avoid_lap(lap, False)
        sides.append(float(np.mean(out[:, 0])))
    if not (sides[0] < 0 < sides[1]):
        print(f"!! 錨點沒跟著起飛點: 兩側平均 x = {sides}"); fails += 1

    # 8) VRPN: 飛機用 sample.name 比對 (sensor != 0 時 key 有後綴, 用 key 比對會失敗)
    q = (0.0, 0.0, 0.0, 1.0)
    bodies = [("drone_01", [0.3, 0.1, -0.4], q), ("box1", [1.0, 0.52, -0.5], q)]
    for sensor in (0, 2):
        holder = []
        th = threading.Thread(target=_fake_vrpn_server, args=(holder, bodies, 3, sensor), daemon=True)
        th.start()
        t0 = time.time()
        while not holder and time.time() - t0 < 3:
            time.sleep(0.01)
        vcfg = {"axes": "y_up", "seconds": 1.0, "drone": "drone_01", "exclude": ["drone_01"],
                "mode": "each_box", "box_size": [0.35, 0.4, 0.5], "pivot": "top"}
        drone, boxes = vc.read_scene(f"127.0.0.1:{holder[0]}", vcfg)
        th.join(timeout=5)
        if drone is None or drone.name != "drone_01":
            print(f"!! sensor={sensor}: 沒分出飛機 ({drone})"); fails += 1
        if any(s.name == "drone_01" for s in boxes.values()):
            print(f"!! sensor={sensor}: 飛機被當成箱子"); fails += 1
        if set(s.name for s in boxes.values()) != {"box1"}:
            print(f"!! sensor={sensor}: 箱子集合錯 {sorted(boxes)}"); fails += 1
        if drone is not None:
            # 假伺服器每幀讓 x 漂 1 mm (模擬串流), 故 x 用 1 cm 容差; y/z 應精確
            xy = vc.sample_to_enu(drone, vcfg)
            if not (abs(xy[0] - 0.3) < 0.01 and abs(xy[1] - 0.4) < 1e-9 and abs(xy[2] - 0.1) < 1e-9):
                print(f"!! sample_to_enu 錯: {xy}"); fails += 1

    # 9) RTL: 飛控的 HOME 可能是解鎖位置(起飛點)也可能是橋接設的房間原點 -> 兩個目的地都要檢查。
    #    牆在 x≈0.9, 軌跡收在 x≈+2, 起飛點放在牆的另一側 -> 兩條回程線都被擋, 應各出一則警告。
    rtl_cfg = deep_update(cfg, {
        "obstacles": {"items": [{"name": "w", "kind": "box", "x": 0.9, "y": 0.0,
                                 "z_bottom": 0.0, "size": [0.3, 2.0, 1.2]}]},
        "waypoints": {"end_action": "rtl", "takeoff_point": [-1.8, 0.0]}})
    warns = [w2 for w2 in plan("circle", rtl_cfg).report.warnings if "RTL" in w2]
    if not any("起飛點" in w2 for w2 in warns) or not any("房間原點" in w2 for w2 in warns):
        print(f"!! RTL 應同時檢查起飛點與房間原點: {warns}"); fails += 1
    # 起飛點就是原點時只檢查一次 (不重複報)
    same = deep_update(cfg, {
        "obstacles": {"items": [{"name": "w", "kind": "box", "x": 0.9, "y": 0.0,
                                 "z_bottom": 0.0, "size": [0.3, 2.0, 1.2]}]},
        "waypoints": {"end_action": "rtl", "takeoff_point": "origin"}})
    if len([w2 for w2 in plan("circle", same).report.warnings if "RTL" in w2]) != 1:
        print("!! 起飛點 = 原點時 RTL 警告應只有一則"); fails += 1

    # 10) 非有限值必須擋下 (NaN 會一路傳進估時, 讓 plan() 空轉到記憶體耗盡)
    import math as _m
    for bad in (_m.nan, _m.inf, -_m.inf):
        for v in ([bad, 0.0], {"x": bad, "y": 0.0}, f"{bad} 0"):
            try:
                tp(v)
                print(f"!! 非有限起飛點未擋下: {v!r}"); fails += 1
            except ValueError:
                pass

    # 11) 房間外的起飛點 -> ERROR; 安全盒外但房間內 (飛機停牆邊) 仍合法
    out = plan("circle", deep_update(cfg, {"waypoints": {"takeoff_point": [16.0, -11.0]}}))
    if not any("在房間外" in e for e in out.report.errors):
        print(f"!! 房間外的起飛點應 ERROR: {out.report.errors}"); fails += 1
    edge = plan("circle", deep_update(cfg, {"waypoints": {"takeoff_point": [2.4, 0.0]}}))
    if not edge.report.ok:
        print(f"!! 安全盒外但房間內的起飛點應合法: {edge.report.errors}"); fails += 1
    # 沒有障礙物時也要檢查 (這個檢查不能塞在 check_obstacles 裡)
    if not any("在房間外" in e for e in
               plan("circle", deep_update(cfg, {"obstacles": {"items": []},
                                                "waypoints": {"takeoff_point": [9.0, 9.0]}})).report.errors):
        print("!! 無障礙物時房間外檢查沒跑"); fails += 1

    # 12) 起飛點在某個箱子裡, 不可讓「第一段穿過另一個箱子」被整段跳過
    hi_box = {"name": "shelf", "kind": "box", "x": 1.5, "y": 1.5, "z_bottom": 1.6, "size": [0.6, 0.6, 0.4]}
    # 放在第一段 (起飛點 -> 往返線起點 (-2,0)) 的中點, 但離往返線本身 0.6 m 以上
    mid_box = {"name": "mid", "kind": "box", "x": -0.25, "y": 0.75, "z_bottom": 0.0, "size": [0.3, 0.3, 1.0]}
    cfg_two = deep_update(cfg, {"obstacles": {"items": [hi_box, mid_box], "enabled": False,
                                              "clearance": 0.2, "low_mode": "around"},
                                "waypoints": {"takeoff_point": [1.5, 1.5]}})
    rep2 = plan("reciprocate", cfg_two).report
    if not any("mid" in e and "穿過障礙物" in e for e in rep2.errors):
        print(f"!! 第一段穿過另一個箱子未被報告: {rep2.errors}"); fails += 1

    # 13) RTL 的判定不可因為「避障停用」而消失 (幾何沒變)
    rtl_off = deep_update(cfg, {
        "obstacles": {"items": [{"name": "w", "kind": "box", "x": 0.9, "y": 0.0,
                                 "z_bottom": 0.0, "size": [0.3, 2.0, 1.2]}], "enabled": False},
        "waypoints": {"end_action": "rtl"}})
    if not any("RTL" in w2 for w2 in plan("circle", rtl_off).report.warnings):
        print("!! 避障停用時 RTL 警告不該消失"); fails += 1

    if not fails:
        print("起飛點 (解析 / 進場 / 估時 / HOME 列 / 爬升檢查 / 錨點 / VRPN 分離 / 範圍驗證) OK")
    return fails


def test_exact_penetration(cfg) -> int:
    """『穿過障礙物』必須用線段對多邊形的精確判定, 不能只靠取樣點 ——
    擦過箱角時取樣可能整段跳過, 但實飛就是會碰到。"""
    import core.safety as S
    from core.trajectory import MissionPath, Trajectory

    fails = 0
    box = safe_box_from_config(cfg)
    o = obstacle_from_item({"name": "b", "kind": "box", "x": 1.0, "y": 1.0, "size": [0.4, 0.4, 0.5]})
    field = ObstacleField([o], box, 0.0, 2)          # clearance 0: 只驗「有沒有切進底面」
    # 軌跡遠離箱子; 航點折線用一條只切進箱角約 7 mm 的長線段
    n = 50
    t = np.linspace(0.0, 1.0, n)
    zero = np.zeros(n)
    traj = Trajectory(name="t", t=t, x=np.full(n, -1.8), y=np.full(n, -1.8), z=np.full(n, 1.0),
                      yaw=zero, vx=zero, vy=zero, vz=zero, s=np.linspace(0, 1, n))
    graze = np.array([[0.0, 2.39], [2.39, 0.0]])
    mp = MissionPath(graze[:, 0], graze[:, 1], np.array([1.0, 1.0]),
                     graze[:, 0], graze[:, 1], np.array([1.0, 1.0]),
                     use_spline=False, repeat="unroll", block_len=0, jump_repeat=0,
                     home=(0.0, 2.39))
    orig_step = S.DISTANCE_SAMPLE_STEP
    try:
        for step, label in ((orig_step, "設定的取樣步長"), (1.0, "刻意調粗的取樣步長")):
            S.DISTANCE_SAMPLE_STEP = step
            r = S.SafetyReport()
            S.check_obstacles(r, traj, cfg, field, mission=mp)
            if not any("穿過障礙物" in e for e in r.errors):
                print(f"!! {label} ({step} m): 擦過箱角未被判定為穿過: {r.errors} {r.warnings}")
                fails += 1
    finally:
        S.DISTANCE_SAMPLE_STEP = orig_step
    # 反面: 完全不碰到箱子的折線不可誤報
    clear = np.array([[0.0, 3.2], [3.2, 0.0]])
    mp2 = MissionPath(clear[:, 0], clear[:, 1], np.array([1.0, 1.0]),
                      clear[:, 0], clear[:, 1], np.array([1.0, 1.0]),
                      use_spline=False, repeat="unroll", block_len=0, jump_repeat=0,
                      home=(0.0, 3.2))
    r2 = S.SafetyReport()
    S.check_obstacles(r2, traj, cfg, field, mission=mp2)
    if any("穿過障礙物" in e for e in r2.errors):
        print(f"!! 沒碰到箱子卻誤報穿過: {r2.errors}"); fails += 1
    if not fails:
        print("穿過障礙物 = 精確線段判定 (擦角也擋得下, 不誤報) OK")
    return fails


def test_airframe(cfg) -> int:
    """機身尺寸: 半對角 (不是半寬) 決定會不會碰到; 安全距離與離牆邊界都要大於它。"""
    fails = 0
    from core.obstacles import drone_radius, drone_size, recommended_clearance, resolve_clearance

    if drone_size(cfg) != (0.5, 0.5, 0.15):
        print(f"!! 預設機身尺寸應為 50×50×15 cm: {drone_size(cfg)}"); fails += 1
    # 只給兩個值時高度用預設 (舊設定檔相容)
    if drone_size(deep_update(cfg, {"obstacles": {"drone_size": [0.4, 0.4]}}))[2] != 0.15:
        print("!! 兩元素的 drone_size 應補上預設高度"); fails += 1
    rad = drone_radius(cfg)
    if abs(rad - math.hypot(0.25, 0.25)) > 1e-9:
        print(f"!! 機身半徑應為半對角 {math.hypot(0.25, 0.25):.4f}, 得到 {rad:.4f}"); fails += 1
    if abs(rad - 0.25) < 1e-9:
        print("!! 機身半徑不可用半寬 (機頭方向會轉, 最壞是對角)"); fails += 1
    rec = recommended_clearance(cfg)
    if abs(rec - (rad + float(cfg["waypoints"]["accept_radius"]) + 0.10)) > 1e-9:
        print(f"!! 建議安全距離公式不對: {rec:.4f}"); fails += 1
    # clearance: auto
    auto = deep_update(cfg, {"obstacles": {"clearance": "auto"}})
    if abs(resolve_clearance(auto) - rec) > 1e-9:
        print(f"!! clearance=auto 應等於建議值: {resolve_clearance(auto):.4f} != {rec:.4f}"); fails += 1
    if abs(resolve_clearance(cfg) - 0.5) > 1e-9:
        print(f"!! 數值 clearance 應原樣採用: {resolve_clearance(cfg)}"); fails += 1
    # 錯誤格式
    for bad in ([0.0, 0.5], [-1, 1], ["a", "b"], [float("nan"), 0.5]):
        try:
            drone_size(deep_update(cfg, {"obstacles": {"drone_size": bad}}))
            print(f"!! 不合理的機身尺寸未擋下: {bad}"); fails += 1
        except Exception:
            pass
    box = {"name": "b", "kind": "box", "x": 1.2, "y": 0.0, "z_bottom": 0.0, "size": [0.4, 0.4, 1.0]}
    with_box = deep_update(cfg, {"obstacles": {"items": [box]}})
    # 預設 (50 cm + clearance 0.5) 不該嘮叨
    base = plan("circle", with_box).report
    if any("機身" in m for m in base.errors + base.warnings):
        print(f"!! 預設機身/安全距離組合不該有訊息: {base.errors} {base.warnings}"); fails += 1
    # 安全距離小於機身半徑 -> ERROR
    tight = plan("circle", deep_update(with_box, {"obstacles": {"clearance": 0.3}})).report
    if not any("小於機身" in e for e in tight.errors):
        print(f"!! 安全距離小於機身半徑應 ERROR: {tight.errors}"); fails += 1
    # 大機身 -> 離牆邊界不足也要 ERROR (即使沒有障礙物)
    big = plan("circle", deep_update(cfg, {"obstacles": {"drone_size": [0.8, 0.8]}})).report
    if not any("比離牆邊界" in e for e in big.errors):
        print(f"!! 機身比離牆邊界寬應 ERROR (無障礙物時也要): {big.errors}"); fails += 1
    # 稍微不足 -> 只警告
    mild = plan("circle", deep_update(with_box, {"obstacles": {"clearance": 0.42}})).report
    if any("小於機身" in e for e in mild.errors) or not any("低於建議" in w2 for w2 in mild.warnings):
        print(f"!! 安全距離略低於建議應只警告: {mild.errors} {mild.warnings}"); fails += 1
    # 垂直: 光球貼頂板 -> 參考點在機身上緣, 機體整個掛在規劃高度下方
    from core.obstacles import marker_height, pivot_above, pivot_below, recommended_vertical_clearance
    below, above = pivot_below(cfg), pivot_above(cfg)
    if abs(marker_height(cfg) - 0.15) > 1e-9 or abs(below - 0.15) > 1e-9 or abs(above - 0.03) > 1e-9:
        print(f"!! 參考點上下機體應為 0.15 / 0.03 m: {below} / {above}"); fails += 1
    if abs(below - 0.075) < 1e-9:
        print("!! 不可退回半高 (光球貼頂板時機體整個在參考點下方)"); fails += 1
    if abs(recommended_vertical_clearance(cfg) - (below + 0.15)) > 1e-9:
        print("!! 建議垂直安全距離公式不對"); fails += 1
    # 未設定 marker_height -> 退回「參考點在機身中心」
    centre = deep_update(cfg, {"obstacles": {"marker_height": None}})
    if abs(pivot_below(centre) - 0.075) > 1e-9 or abs(pivot_above(centre) - 0.075) > 1e-9:
        print(f"!! 未設 marker_height 應退回上下各半高: {pivot_below(centre)} / {pivot_above(centre)}")
        fails += 1
    low = {"name": "low", "kind": "box", "x": 0.9, "y": 0.0, "z_bottom": 0.0, "size": [0.5, 0.4, 0.5]}
    with_low = deep_update(cfg, {"obstacles": {"items": [low]}})
    if any("垂直" in m for m in plan("reciprocate", with_low).report.errors +
           plan("reciprocate", with_low).report.warnings):
        print("!! 預設垂直設定不該有訊息"); fails += 1
    tight_v = plan("reciprocate", deep_update(with_low, {"obstacles": {"vertical_clearance": 0.05}})).report
    if not any("小於規劃高度下方的機體" in e for e in tight_v.errors):
        print(f"!! 垂直安全距離小於半高應 ERROR: {tight_v.errors}"); fails += 1
    mild_v = plan("reciprocate", deep_update(with_low, {"obstacles": {"vertical_clearance": 0.15}})).report
    if any("小於規劃高度下方" in e for e in mild_v.errors) or not any("低於建議" in w2 for w2 in mild_v.warnings):
        print(f"!! 垂直安全距離略低於建議應只警告: {mild_v.errors} {mild_v.warnings}"); fails += 1
    # 離地看『下方』(0.15), 離天花板看『上方』(0.03) —— 兩邊門檻不同
    for key, bad, ok_val in (("floor", 0.05, 0.60), ("ceiling", 0.01, 0.60)):
        rep = plan("circle", deep_update(cfg, {"margin": {key: bad}})).report
        if not any(f"margin.{key}" in e for e in rep.errors):
            print(f"!! margin.{key}={bad} 小於機體應 ERROR: {rep.errors}"); fails += 1
        rep = plan("circle", deep_update(cfg, {"margin": {key: ok_val}})).report
        if any(f"margin.{key}" in e for e in rep.errors):
            print(f"!! margin.{key}={ok_val} 不該 ERROR: {rep.errors}"); fails += 1
    # 離天花板 0.05 對「光球貼頂板」是夠的 (上方只有 0.03), 對「參考點在中心」則不夠
    if any("margin.ceiling" in e for e in plan("circle", deep_update(cfg, {"margin": {"ceiling": 0.05}})).report.errors):
        print("!! 光球貼頂板時 margin.ceiling=0.05 應可接受"); fails += 1
    if not any("margin.ceiling" in e for e in
               plan("circle", deep_update(centre, {"margin": {"ceiling": 0.05}})).report.errors):
        print("!! 參考點在中心時 margin.ceiling=0.05 應 ERROR"); fails += 1
    # 起飛爬升柱頂端 = 起飛高度 + 參考點上方 + 超調 = 1.0 + 0.03 + 0.05 = 1.08
    for zb, want_err in ((1.05, True), (1.30, False)):
        box_t = {"name": "t", "kind": "box", "x": 0.0, "y": 0.0, "z_bottom": zb, "size": [0.4, 0.4, 0.3]}
        rep = plan("circle", deep_update(cfg, {"obstacles": {"items": [box_t]}})).report
        got = any("原地爬到" in e for e in rep.errors)
        if got != want_err:
            print(f"!! 箱底 {zb} m 的爬升柱判定錯 (預期 ERROR={want_err}): {rep.errors}"); fails += 1
    if not fails:
        print("機身尺寸 (半對角 / 參考點上下機體 / 建議距離 / auto / 離牆離地 / 爬升柱) OK")
    return fails


def main() -> int:
    cfg = load_config()
    fails = test_geometry()
    fails += test_avoid_lap()
    fails += test_planner(cfg)
    fails += test_over(cfg)
    fails += test_takeoff_point(cfg)
    fails += test_exact_penetration(cfg)
    fails += test_airframe(cfg)
    fails += test_vrpn()
    print()
    if fails:
        print(f"FAILED: {fails} 項問題")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
