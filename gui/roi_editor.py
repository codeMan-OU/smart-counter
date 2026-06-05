"""
gui/roi_editor.py — ROI / TripWire 編輯控制面板

提供按鈕讓使用者切換進入 ROI 多邊形編輯模式或 Trip Wire 編輯模式。
與 VideoWidget 協作，本身只負責按鈕狀態管理。
"""

from __future__ import annotations
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from config import CountMode


class RoiEditorPanel(QGroupBox):
    """
    ROI / Wire 編輯控制面板。

    訊號：
    - edit_roi_requested()  : 使用者請求進入 ROI 多邊形編輯模式
    - edit_wire_requested() : 使用者請求進入 Trip Wire 編輯模式
    - edit_cancelled()      : 取消編輯
    - clear_roi_requested() : 清除 ROI
    - clear_wire_requested(): 清除 Wire
    """

    edit_roi_requested  = pyqtSignal()
    edit_wire_requested = pyqtSignal()
    edit_cancelled      = pyqtSignal()
    clear_roi_requested = pyqtSignal()
    clear_wire_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__("🔲  ROI / 絆線設定", parent)
        self._in_edit = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── ROI 多邊形 ─────────────────────────────────────────────────
        roi_label = QLabel("ROI 多邊形（進入計數模式使用）")
        roi_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(roi_label)

        roi_btn_row = QHBoxLayout()
        self._btn_edit_roi = QPushButton("繪製 ROI")
        self._btn_edit_roi.setCheckable(True)
        self._btn_edit_roi.setMinimumHeight(34)
        self._btn_edit_roi.setMinimumWidth(130)
        self._btn_edit_roi.setToolTip(
            "點擊進入 ROI 繪製模式\n左鍵新增頂點，右鍵封閉多邊形，雙擊重置"
        )
        self._btn_clear_roi = QPushButton("清除 ROI")
        self._btn_clear_roi.setObjectName("danger")
        self._btn_clear_roi.setMinimumHeight(34)
        self._btn_clear_roi.setMinimumWidth(110)

        roi_btn_row.addWidget(self._btn_edit_roi)
        roi_btn_row.addWidget(self._btn_clear_roi)
        layout.addLayout(roi_btn_row)

        # ── Trip Wire ──────────────────────────────────────────────────
        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.HLine)
        self._separator.setStyleSheet("color: #333;")
        layout.addWidget(self._separator)

        wire_label = QLabel("Trip Wire（穿越計數模式使用）")
        wire_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(wire_label)

        wire_btn_row = QHBoxLayout()
        self._btn_edit_wire = QPushButton("繪製絆線")
        self._btn_edit_wire.setCheckable(True)
        self._btn_edit_wire.setMinimumHeight(34)
        self._btn_edit_wire.setMinimumWidth(130)
        self._btn_edit_wire.setToolTip(
            "點擊進入絆線繪製模式\n在影像上依序點擊兩點定義線段"
        )
        self._btn_clear_wire = QPushButton("清除絆線")
        self._btn_clear_wire.setObjectName("danger")
        self._btn_clear_wire.setMinimumHeight(34)
        self._btn_clear_wire.setMinimumWidth(110)

        wire_btn_row.addWidget(self._btn_edit_wire)
        wire_btn_row.addWidget(self._btn_clear_wire)
        layout.addLayout(wire_btn_row)

        # ── 狀態提示 ───────────────────────────────────────────────────
        self._status_label = QLabel("尚未設定 ROI / 絆線")
        self._status_label.setStyleSheet("color: #ffaa44; font-size: 11px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # ── 連接訊號 ───────────────────────────────────────────────────
        self._btn_edit_roi.clicked.connect(self._on_edit_roi_clicked)
        self._btn_edit_wire.clicked.connect(self._on_edit_wire_clicked)
        self._btn_clear_roi.clicked.connect(self.clear_roi_requested)
        self._btn_clear_wire.clicked.connect(self.clear_wire_requested)

    # ── 槽函數 ────────────────────────────────────────────────────────────────

    def _on_edit_roi_clicked(self, checked: bool) -> None:
        if checked:
            self._btn_edit_wire.setChecked(False)
            self._set_status("🟢 ROI 繪製模式：左鍵新增頂點，右鍵封閉，雙擊重置")
            self.edit_roi_requested.emit()
        else:
            self._set_status_idle()
            self.edit_cancelled.emit()

    def _on_edit_wire_clicked(self, checked: bool) -> None:
        if checked:
            self._btn_edit_roi.setChecked(False)
            self._set_status("🟢 絆線模式：在影像上點擊兩點定義線段")
            self.edit_wire_requested.emit()
        else:
            self._set_status_idle()
            self.edit_cancelled.emit()

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet("color: #44ff88; font-size: 11px;")

    def _set_status_idle(self) -> None:
        self._status_label.setText("編輯已取消")
        self._status_label.setStyleSheet("color: #aaa; font-size: 11px;")

    # ── 公開更新方法 ──────────────────────────────────────────────────────────

    def on_roi_set(self, point_count: int) -> None:
        """ROI 設定完成後呼叫，更新狀態顯示。"""
        self._btn_edit_roi.setChecked(False)
        msg = f"✅ ROI 已設定（{point_count} 個頂點）"
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("color: #44ff88; font-size: 11px;")

    def on_wire_set(self) -> None:
        """Wire 設定完成後呼叫。"""
        self._btn_edit_wire.setChecked(False)
        self._status_label.setText("✅ 絆線已設定")
        self._status_label.setStyleSheet("color: #44ff88; font-size: 11px;")

    def on_roi_cleared(self) -> None:
        self._status_label.setText("ROI 已清除")
        self._status_label.setStyleSheet("color: #ffaa44; font-size: 11px;")

    def on_wire_cleared(self) -> None:
        self._status_label.setText("絆線已清除")
        self._status_label.setStyleSheet("color: #ffaa44; font-size: 11px;")
