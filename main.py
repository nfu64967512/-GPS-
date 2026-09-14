"""
Indoor Trajectory Studio 進入點。

  python main.py             # 啟動 PyQt6 GUI
  python main.py --obstacles # 啟動 GUI 並直接切到「障礙物」頁
  python main.py --cli ...   # 命令列 (見 cli.py)
"""

from __future__ import annotations

import os
import sys

# 確保可從專案根目錄匯入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--cli":
        from cli import run_cli
        run_cli(argv[1:])
    else:
        from ui.main_window import launch
        launch(argv)


if __name__ == "__main__":
    main()
