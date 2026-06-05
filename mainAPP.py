"""
main.py — YOLO 物件計數器程式入口

執行方式：
    python main.py

依賴：
    pip install ultralytics opencv-python PyQt5 numpy
"""

import sys
import os

# 將專案根目錄加入 sys.path，確保相對 import 正常運作
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from gui.main_window import MainWindow


def main() -> None:
    # 高 DPI 支援
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO 物件計數器")
    app.setOrganizationName("YoloCounter")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
