"""
gui/dashboard.py — 計數儀表板元件

顯示：
- 大型計數數字（A 總數）
- 含 B 計數（綠色）/ 無 B 計數（黃色）子統計
- 計數事件歷史清單（ID / 時間 / 模式 / 含B）
- 重置按鈕 / CSV 檔案路徑顯示
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy,
    QAbstractItemView, QFrame, QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from core.counter import CountEvent


class Dashboard(QGroupBox):
    """
    計數儀表板。

    訊號：
    - reset_requested() : 使用者點擊重置計數
    """

    reset_requested = pyqtSignal()
    label_names_changed = pyqtSignal(str, str)

    _MAX_TABLE_ROWS = 500  # 最多顯示幾筆歷史

    def __init__(self, parent=None) -> None:
        super().__init__("📊  計數儀表板", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── 標題列（累計計數 + 重置按鈕）────────────────────────────────
        count_row = QHBoxLayout()
        count_label_title = QLabel("累計計數")
        count_label_title.setStyleSheet("color: #888; font-size: 13px;")
        count_row.addWidget(count_label_title)
        count_row.addStretch()

        self._btn_reset = QPushButton("重置")
        self._btn_reset.setObjectName("danger")
        self._btn_reset.setFixedWidth(60)
        self._btn_reset.setToolTip("重置計數（不影響 ID 追蹤記錄）")
        self._btn_reset.clicked.connect(self.reset_requested)
        count_row.addWidget(self._btn_reset)
        layout.addLayout(count_row)

        # ── 大型總計數數字 ────────────────────────────────────────────────
        self._count_display = QLabel("0")
        self._count_display.setAlignment(Qt.AlignCenter)
        self._count_display.setStyleSheet(
            "font-size: 72px; font-weight: bold; color: #ffffff;"
            "letter-spacing: 4px; padding: 4px 0 0 0;"
        )
        layout.addWidget(self._count_display)

        # ── 含 B / 無 B 子統計 ────────────────────────────────────────────
        #
        # 視覺結構：
        #   總計數  [ 5 ]
        #     ├ 含 B  [ 3 ]   ← 綠色
        #     └ 無 B  [ 2 ]   ← 黃色
        #
        sub_frame = QFrame()
        sub_frame.setStyleSheet(
            "QFrame { background-color: #1a1a2e; border: 1px solid #2a2a4a;"
            " border-radius: 6px; padding: 4px; }"
        )
        sub_layout = QVBoxLayout(sub_frame)
        sub_layout.setSpacing(4)
        sub_layout.setContentsMargins(10, 6, 10, 6)

        # 含 B 列
        with_b_row = QHBoxLayout()
        with_b_prefix = QLabel("├ 含 B")
        with_b_prefix.setStyleSheet("color: #4aff8c; font-size: 13px; font-weight: bold;")
        with_b_row.addWidget(with_b_prefix)
        self._with_b_name_edit = QLineEdit()
        self._with_b_name_edit.setPlaceholderText("自訂名稱")
        self._with_b_name_edit.setMaxLength(20)
        self._with_b_name_edit.setFixedHeight(28)
        self._with_b_name_edit.editingFinished.connect(self._emit_label_names_changed)
        with_b_row.addWidget(self._with_b_name_edit, stretch=1)
        self._count_with_b_label = QLabel("0")
        self._count_with_b_label.setStyleSheet(
            "color: #4aff8c; font-size: 22px; font-weight: bold; min-width: 40px;"
        )
        self._count_with_b_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        with_b_row.addWidget(self._count_with_b_label)
        sub_layout.addLayout(with_b_row)

        # 分隔線
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2a2a4a;")
        sub_layout.addWidget(sep)

        # 無 B 列
        no_b_row = QHBoxLayout()
        no_b_prefix = QLabel("└ 無 B")
        no_b_prefix.setStyleSheet("color: #4a8aff; font-size: 13px; font-weight: bold;")
        no_b_row.addWidget(no_b_prefix)
        self._no_b_name_edit = QLineEdit()
        self._no_b_name_edit.setPlaceholderText("自訂名稱")
        self._no_b_name_edit.setMaxLength(20)
        self._no_b_name_edit.setFixedHeight(28)
        self._no_b_name_edit.editingFinished.connect(self._emit_label_names_changed)
        no_b_row.addWidget(self._no_b_name_edit, stretch=1)
        self._count_no_b_label = QLabel("0")
        self._count_no_b_label.setStyleSheet(
            "color: #4a8aff; font-size: 22px; font-weight: bold; min-width: 40px;"
        )
        self._count_no_b_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        no_b_row.addWidget(self._count_no_b_label)
        sub_layout.addLayout(no_b_row)

        layout.addWidget(sub_frame)

        # ── FPS 與模式 ─────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("color: #888; font-size: 11px;")
        self._mode_label = QLabel("模式: --")
        self._mode_label.setStyleSheet("color: #888; font-size: 11px;")
        info_row.addWidget(self._fps_label)
        info_row.addStretch()
        info_row.addWidget(self._mode_label)
        layout.addLayout(info_row)

        # ── 歷史清單表格 ───────────────────────────────────────────────
        table_label = QLabel("計數歷史")
        table_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(table_label)

        # 欄位：計數 / ID / 時間 / 模式 / 含B
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["計數", "ID", "時間", "模式", "含B"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setStyleSheet(
            "QHeaderView::section { background-color: #2a2a4a; color: #aaa; "
            "font-size: 11px; border: none; padding: 4px; }"
        )
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(180)
        self._table.setStyleSheet(
            "QTableWidget { background-color: #12121e; color: #ccc; "
            "font-size: 11px; border: 1px solid #2a2a4a; }"
            "QTableWidget::item:alternate { background-color: #1a1a2e; }"
        )
        layout.addWidget(self._table)

        # ── CSV 路徑 ───────────────────────────────────────────────────
        self._csv_label = QLabel("CSV：尚未記錄")
        self._csv_label.setStyleSheet(
            "color: #555; font-size: 10px;"
        )
        self._csv_label.setWordWrap(True)
        layout.addWidget(self._csv_label)

    # ── 公開更新介面 ──────────────────────────────────────────────────────────

    def update_count(self, count: int, count_with_b: int = 0, count_no_b: int = 0) -> None:
        """更新計數大數字與子統計顯示。"""
        self._count_display.setText(str(count))
        self._count_with_b_label.setText(str(count_with_b))
        self._count_no_b_label.setText(str(count_no_b))

    def set_label_names(self, count_with_b_name: str, count_no_b_name: str) -> None:
        """套用子統計自訂名稱。"""
        self._with_b_name_edit.blockSignals(True)
        self._no_b_name_edit.blockSignals(True)
        self._with_b_name_edit.setText(count_with_b_name or "")
        self._no_b_name_edit.setText(count_no_b_name or "")
        self._with_b_name_edit.blockSignals(False)
        self._no_b_name_edit.blockSignals(False)

    def label_names(self) -> tuple[str, str]:
        """回傳目前輸入欄位中的自訂名稱。"""
        return (
            self._with_b_name_edit.text().strip(),
            self._no_b_name_edit.text().strip(),
        )

    def _emit_label_names_changed(self) -> None:
        self.label_names_changed.emit(*self.label_names())

    def set_interaction_locked(self, locked: bool) -> None:
        """Lock user-editable dashboard controls while keeping live values visible."""
        self._btn_reset.setEnabled(not locked)
        self._with_b_name_edit.setEnabled(not locked)
        self._no_b_name_edit.setEnabled(not locked)

    def update_fps(self, fps: float) -> None:
        self._fps_label.setText(f"FPS: {fps:.1f}")

    def update_mode(self, mode: str) -> None:
        mode_map = {
            "trip_wire": "絆線穿越",
            "roi_enter": "ROI 進入",
        }
        self._mode_label.setText(f"模式：{mode_map.get(mode, mode)}")

    def add_event(self, event: CountEvent) -> None:
        """新增單筆計數事件到歷史表格。"""
        if self._table.rowCount() >= self._MAX_TABLE_ROWS:
            self._table.removeRow(0)

        row = self._table.rowCount()
        self._table.insertRow(row)

        dt = datetime.fromtimestamp(event.timestamp)
        time_str = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

        mode_map = {"trip_wire": "絆線", "roi_enter": "ROI"}
        has_b_str = "✔" if event.has_b else "—"

        items = [
            str(event.count_at),
            str(event.track_id),
            time_str,
            mode_map.get(event.mode, event.mode),
            has_b_str,
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)

            # 「含B」欄位依狀態著色
            if col == 4:
                if event.has_b:
                    item.setForeground(QColor("#4aff8c"))   # 綠色
                else:
                    item.setForeground(QColor("#4a8aff"))   # 藍色

            self._table.setItem(row, col, item)

        self._table.scrollToBottom()

    def clear_table(self) -> None:
        """清除歷史清單。"""
        self._table.setRowCount(0)

    def update_csv_path(self, path: Optional[str]) -> None:
        if path:
            self._csv_label.setText(f"CSV：{path}")
            self._csv_label.setStyleSheet("color: #4aff8c; font-size: 10px;")
        else:
            self._csv_label.setText("CSV：尚未記錄")
            self._csv_label.setStyleSheet("color: #555; font-size: 10px;")
