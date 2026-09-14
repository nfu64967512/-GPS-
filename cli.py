"""
命令列介面 (headless)。由 main.py --cli 進入。

  python main.py --cli list
  python main.py --cli preview circle [--out out.png]
  python main.py --cli export circle [--out output] [--formats waypoints,csv,plan,png]
  python main.py --cli export all
  python main.py --cli fly circle [--connect tcp:127.0.0.1:5760] [--confirm]
  python main.py --cli vrpn [--server localhost:3883] [--seconds 2] [--save obstacles.yaml]
      列出 Motive VRPN 串流上的 rigid body: 箱子轉成障礙物, 飛機 (drone) 的位置轉成起飛點;
      --save 存成可用 --config 疊加的 YAML (含 obstacles.items 與 waypoints.takeoff_point)
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from core import patterns
from core.config import deep_update, load_config
from core.geometry import describe_takeoff_point, safe_box_from_config, takeoff_is_origin
from core.planner import duration_basis, plan as plan_pattern
from core.safety import DEFAULT_MAX_WAYPOINTS, box_summary


def _all_keys():
    return [k for k, _ in patterns.list_patterns()]


def _resolve_keys(name: str):
    if name == "all":
        return _all_keys()
    if name not in dict(patterns.list_patterns()):
        raise SystemExit(f"未知 pattern: {name}; 可用: {_all_keys()} 或 all")
    return [name]


def _print_plan_summary(pr, cfg):
    t = pr.trajectory
    m = t.meta
    extra = ""
    if m.get("speed_profile") == "dynamic":
        extra = (f" (定速舊估 {m.get('naive_duration_s', 0):.0f}s, "
                 f"硬轉角 {m.get('hard_corners_total', 0)} 處)")
    auto = (m.get("auto_estimate") or {}).get("compact")
    if auto and auto.get("repeat") == "do_jump" and auto.get("jump_repeat", 0) > 0:
        wp_txt = f"一圈 {auto['block_len']} wp + DO_JUMP×{auto['jump_repeat']} + 收尾"
    elif auto:
        wp_txt = f"{auto['n_wp']} wp"
    auto_line = (f"AUTO(精簡) 任務: {auto['total_s']:.0f}s = 起飛 {auto['takeoff_s']:.0f} + "
                 f"航線 {auto['nav_s']:.0f} + 降落 {auto['land_s']:.0f}s ({wp_txt})"
                 if auto else "")
    if auto and auto.get("n_turns"):
        auto_line += (f"; 原地轉頭 {auto['n_turns']} 點 (停 {auto['turn_stops']} 次, "
                      f"懸停共 {auto['turn_s']:.0f}s)")
    per_lap = m.get("altitude_cycles_per_lap", 0)
    if m.get("mission_repeat") == "do_jump" and per_lap:
        auto_line += f"; 高度起伏每圈 {per_lap} 次"
    if duration_basis(cfg) == "auto" and auto:
        # 工時基準 = AUTO：航線工時為主, GUIDED 退為參考
        print(f"  {pr.pattern.display_name}: {pr.laps} 圈, AUTO 航線工時 {auto['nav_s']:.0f}s, "
              f"長度 {t.path_length_3d:.0f}m, vmax {t.max_speed:.2f} m/s")
        print(f"   {auto_line}")
        print(f"   GUIDED 串流工時 (參考) {t.duration:.0f}s{extra}")
    else:
        print(f"  {pr.pattern.display_name}: {pr.laps} 圈, 工時 {t.duration:.0f}s{extra}, "
              f"長度 {t.path_length_3d:.0f}m, vmax {t.max_speed:.2f} m/s")
        if auto_line:
            print(f"   {auto_line}")
    ob = m.get("obstacles") or {}
    n_ap = int((auto or {}).get("n_approach", 0) or 0)
    if not takeoff_is_origin(cfg) or ob.get("count"):
        print(f"   起飛點 {describe_takeoff_point(cfg)}"
              + (f", 進場繞障 {n_ap} 點" if n_ap else ", 進場直線可達"))
    if ob.get("count"):
        av = ob.get("avoid")
        if av and (av.get("moved") or av.get("removed") or av.get("detours") or av.get("failed")):
            txt = (f"改道 {av['detours']} 段, 頂點推到禁區邊界 {av['moved']} 個, 單圈路徑 "
                   f"{av['length_before_m']:.1f} -> {av['length_after_m']:.1f} m"
                   + (f", 繞不開 {av['failed']} 段!" if av.get("failed") else ""))
        elif av:
            txt = "路徑沒碰到障礙物, 未改道"
        else:
            txt = "避障已停用"
        print(f"   {ob.get('description', '')}; 避障: {txt}")
    for line in pr.report.as_lines():
        print("   " + line)


def cmd_list(args, cfg):
    print(box_summary(safe_box_from_config(cfg)))
    print("可用軌跡 patterns:")
    for k, name in patterns.list_patterns():
        print(f"  {k:<14} {name}")
    from core.obstacles import (drone_radius, drone_size, field_from_config, pivot_above,
                                pivot_below, recommended_clearance, recommended_vertical_clearance)
    sx, sy, sz = drone_size(cfg)
    print(f"機身: {sx*100:.0f}×{sy*100:.0f}×{sz*100:.0f} cm (半對角 {drone_radius(cfg):.2f} m); "
          f"規劃高度下方 {pivot_below(cfg):.2f} m / 上方 {pivot_above(cfg):.2f} m; "
          f"建議 水平安全距離 >= {recommended_clearance(cfg):.2f} m, "
          f"越箱垂直距離 >= {recommended_vertical_clearance(cfg):.2f} m")
    print(f"起飛點: {describe_takeoff_point(cfg)}")
    field = field_from_config(cfg, safe_box_from_config(cfg))
    if field.obstacles:
        print(field.describe() + ":")
        for o in field.obstacles:
            print(f"  {o.summary()}")
    else:
        print("障礙物: 無 (config obstacles.items 或 --cli vrpn --save)")


def cmd_vrpn(args, cfg):
    """連 VRPN 列出 rigid body -> 箱子當障礙物、飛機當起飛點 (可存成 YAML)。"""
    import math

    from core.vrpn_client import (axes_matrix, read_scene, sample_to_enu, to_enu,
                                  trackers_to_items, yaw_enu)
    v = dict((cfg.get("obstacles", {}) or {}).get("vrpn", {}) or {})
    if args.server:
        v["server"] = args.server
    if args.seconds is not None:
        v["seconds"] = float(args.seconds)
    if args.axes:
        v["axes"] = args.axes
    if args.mode:
        v["mode"] = args.mode
    if args.drone is not None:
        v["drone"] = args.drone
    if args.exclude is not None:
        v["exclude"] = [t.strip() for t in args.exclude.split(",") if t.strip()]
    if args.no_takeoff:
        v["takeoff_from_drone"] = False
    drone, boxes = read_scene(v.get("server", "localhost:3883"), v, log=print)
    if drone is None and not boxes:
        print("沒有收到任何 rigid body (確認 Motive Data Streaming -> VRPN 已開, rigid body 已啟用)")
        return
    M = axes_matrix(v.get("axes", "y_up"))
    print(f"{'tracker':<20}{'Motive pos (m)':<34}{'ENU (x,y,z)':<30}{'yaw':>6}{'筆數':>6}  角色")
    for key, s in sorted(list(boxes.items()) + ([(drone.key, drone)] if drone else [])):
        p = to_enu(s.pos, M)
        role = "飛機 (起飛點)" if drone is not None and s is drone else "箱子 (障礙物)"
        print(f"{key:<20}{str(tuple(round(float(x), 3) for x in s.pos)):<34}"
              f"{str(tuple(round(float(x), 3) for x in p)):<30}"
              f"{math.degrees(yaw_enu(s.quat, M)):>6.0f}{s.count:>6}  {role}")
    items = trackers_to_items(boxes, v)
    print(f"-> {len(items)} 個障礙物 ({v.get('mode', 'each_box')}, axes={v.get('axes', 'y_up')})")
    for it in items:
        print("   ", {k: val for k, val in it.items() if k != "note"}, "|", it.get("note", ""))
    take = None
    if drone is not None:
        dx, dy, dz = sample_to_enu(drone, v)
        use = bool(v.get("takeoff_from_drone", True))
        print(f"-> 起飛點 ({dx:+.3f}, {dy:+.3f}) m, 機體高 {dz:.3f} m"
              f"{'' if use else '  [未套用: takeoff_from_drone=false]'}")
        if use:
            take = [round(dx, 4), round(dy, 4)]
    else:
        print(f"-> 沒讀到飛機 rigid body「{v.get('drone', 'drone_01')}」, 起飛點不變")
    if args.save:
        import yaml
        out = {"obstacles": {"items": items}}
        if take is not None:
            out["waypoints"] = {"takeoff_point": take}
        with open(args.save, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
        print(f"-> 已存 {args.save} (之後: python main.py --cli --config {args.save} export all)")


def cmd_preview(args, cfg):
    import viz
    pr = plan_pattern(args.pattern, cfg)
    out = args.out or os.path.join("output", f"{args.pattern}_preview.png")
    viz.save_png(pr, cfg, out)
    _print_plan_summary(pr, cfg)
    print(f"  -> 已存預覽: {out}")


def cmd_export(args, cfg):
    from io_export import export_waypoints_dual, export_csv
    import viz

    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_dir = args.out or "output"
    os.makedirs(out_dir, exist_ok=True)

    for key in _resolve_keys(args.pattern):
        pr = plan_pattern(key, cfg)
        _print_plan_summary(pr, cfg)
        base = os.path.join(out_dir, key)
        if "waypoints" in fmts:
            res = export_waypoints_dual(pr, cfg, base + ".waypoints")
            pinfo, cinfo = res["precision"], res["compact"]
            print(f"   -> {pinfo['path']} [高精度] 總項 {pinfo['total_items']} "
                  f"({pinfo['nav']} nav wp)")
            tag = "OK <=預算" if cinfo["within_budget"] else "仍超過!"
            print(f"   -> {cinfo['path']} [精簡/上飛控] 總項 {cinfo['total_items']} "
                  f"({cinfo['nav']} nav wp)  預算 {res['budget']} [{tag}]")
            max_wp = int(cfg.get("waypoints", {}).get("max_waypoints", DEFAULT_MAX_WAYPOINTS))
            if pinfo["total_items"] > max_wp:
                print(f"   [WARN] 高精度版 {pinfo['total_items']} 項 > {max_wp}; "
                      f"上飛控請用 _compact 版")
        if "csv" in fmts:
            n = export_csv(pr, cfg, base + ".csv")
            print(f"   -> {base}.csv ({n} 列)")
        if "png" in fmts:
            viz.save_png(pr, cfg, base + "_preview.png")
            print(f"   -> {base}_preview.png")


def cmd_fly(args, cfg):
    from runner import GuidedRunner
    pr = plan_pattern(args.pattern, cfg)
    _print_plan_summary(pr, cfg)
    conn = args.connect or cfg["guided"]["connection"]
    runner = GuidedRunner(conn)
    runner.run(pr, cfg, confirm=args.confirm)


def run_cli(argv):
    ap = argparse.ArgumentParser(prog="indoor_traj_studio", description="室內 3D 軌跡規劃 (CLI)")
    ap.add_argument("--config", default=None, help="自訂 YAML 設定檔")
    ap.add_argument("--turn-in-place", action="store_true",
                    help="開啟 waypoints.turn_in_place: 到航點停下原地轉頭 (CONDITION_YAW + NAV_DELAY) 再前進")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出 patterns 與可用盒")

    p = sub.add_parser("preview", help="存 3D 預覽 PNG")
    p.add_argument("pattern")
    p.add_argument("--out", default=None)

    p = sub.add_parser("export", help="匯出 .waypoints / .csv / png")
    p.add_argument("pattern", help="pattern key 或 all")
    p.add_argument("--out", default=None, help="輸出資料夾 (預設 output)")
    p.add_argument("--formats", default="waypoints,csv", help="逗號分隔: waypoints,csv,png")

    p = sub.add_parser("fly", help="GUIDED 即時串流 (預設乾跑, 加 --confirm 才真飛)")
    p.add_argument("pattern")
    p.add_argument("--connect", default=None)
    p.add_argument("--confirm", action="store_true", help="真的解鎖起飛 (謹慎!)")

    p = sub.add_parser("vrpn", help="從 Motive VRPN 讀 rigid body (箱子光球) -> 障礙物")
    p.add_argument("--server", default=None, help="host[:port] (預設 config obstacles.vrpn.server)")
    p.add_argument("--seconds", type=float, default=None, help="收幾秒 (預設 2)")
    p.add_argument("--axes", default=None, help="y_up | z_up | 'x,-z,y'")
    p.add_argument("--mode", default=None, choices=["each_box", "hull"])
    p.add_argument("--drone", default=None, help="飛機的 rigid body 名稱 (預設 config obstacles.vrpn.drone)")
    p.add_argument("--no-takeoff", action="store_true", help="不要用飛機位置更新起飛點")
    p.add_argument("--exclude", default=None, help="略過這些 rigid body (逗號分隔; 預設 config obstacles.vrpn.exclude); 給空字串 = 不略過")
    p.add_argument("--save", default=None, help="存成 YAML (obstacles.items), 可用 --config 疊加")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.turn_in_place:
        cfg = deep_update(cfg, {"waypoints": {"turn_in_place": {"enabled": True}}})

    {
        "list": cmd_list,
        "preview": cmd_preview,
        "export": cmd_export,
        "fly": cmd_fly,
        "vrpn": cmd_vrpn,
    }[args.cmd](args, cfg)
