"""
gui/control_panel.py — 來源設定、模型選擇、計數模式切換控制面板

包含：
- 影像來源選擇（影片 / RTSP / USB）
- 模型路徑選擇
- 計數模式切換（Trip Wire / ROI Enter）
- 開始 / 暫停 / 停止按鈕
"""

from __future__ import annotations
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QRadioButton, QButtonGroup, QFileDialog,
    QSpinBox, QFrame, QCheckBox,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from config import AppSettings, SourceType, CountMode, BASE_DIR, DEFAULT_CSV_DIR


class ControlPanel(QWidget):
    """
    控制面板元件。

    訊號：
    - start_requested(AppSettings) : 請求開始推論
    - pause_requested()            : 暫停 / 繼續
    - stop_requested()             : 停止並釋放資源
    - mode_changed(str)            : 計數模式切換
    - settings_changed(AppSettings): 設定更新（不含開始/停止）
    """

    start_requested   = pyqtSignal(object)   # AppSettings
    pause_requested   = pyqtSignal()
    stop_requested    = pyqtSignal()
    mode_changed      = pyqtSignal(str)
    settings_changed  = pyqtSignal(object)   # AppSettings
    speed_changed     = pyqtSignal(float)    # 播放速度倍率（0.25 ~ 4.0）
    file_selected     = pyqtSignal(str)      # 影片檔路徑（選檔後立即發出）
    preview_requested = pyqtSignal(object)   # AppSettings（預覽按鈕點擊）
    show_id_changed   = pyqtSignal(bool)     # 顯示/隱藏 Track ID
    voice_prompt_changed = pyqtSignal(bool)  # 啟用/停用語音提示
    confidence_threshold_changed = pyqtSignal(float)  # 物件框信心值門檻

    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._running = False
        self._paused = False
        self._locked = False
        self._previewing = False   # 預覽執行緒是否正在運行
        self._build_ui()
        self._load_settings(settings)

    # ── UI 建構 ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        self._main_layout = main_layout
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── 影像來源 ─────────────────────────────────────────────────────
        src_group = QGroupBox("📹  影像來源")
        src_layout = QVBoxLayout(src_group)

        self._source_combo = QComboBox()
        self._source_combo.addItems(["本地影片檔", "RTSP 串流", "USB 攝影機"])
        self._source_combo.currentIndexChanged.connect(self._on_source_type_changed)
        src_layout.addWidget(self._source_combo)

        # 影片檔路徑
        self._file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("選擇影片檔…")
        self._btn_browse = QPushButton("📂 瀏覽")
        self._btn_browse.setFixedWidth(72)
        self._btn_browse.clicked.connect(self._browse_file)
        self._file_row.addWidget(self._file_edit)
        self._file_row.addWidget(self._btn_browse)
        src_layout.addLayout(self._file_row)

        # RTSP URL
        self._rtsp_row = QHBoxLayout()
        self._rtsp_edit = QLineEdit()
        self._rtsp_edit.setPlaceholderText("rtsp://192.168.1.1/stream")
        self._rtsp_row.addWidget(QLabel("URL："))
        self._rtsp_row.addWidget(self._rtsp_edit)
        src_layout.addLayout(self._rtsp_row)

        # USB 編號
        self._usb_row = QHBoxLayout()
        self._usb_spin = QSpinBox()
        self._usb_spin.setRange(0, 10)
        self._usb_spin.setValue(0)
        self._usb_row.addWidget(QLabel("攝影機編號："))
        self._usb_row.addWidget(self._usb_spin)
        self._usb_row.addStretch()
        src_layout.addLayout(self._usb_row)

        main_layout.addWidget(src_group)
        self._on_source_type_changed(0)  # 初始化顯示狀態

        # ── 模型設定 ─────────────────────────────────────────────────────
        model_group = QGroupBox("🧠  模型設定")
        model_layout = QHBoxLayout(model_group)
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("yolov8n.pt")
        self._btn_model = QPushButton("📂")
        self._btn_model.setFixedWidth(36)
        self._btn_model.clicked.connect(self._browse_model)
        model_layout.addWidget(QLabel("權重："))
        model_layout.addWidget(self._model_edit)
        model_layout.addWidget(self._btn_model)
        main_layout.addWidget(model_group)

        # ── 計數模式 ─────────────────────────────────────────────────────
        mode_group = QGroupBox("🔢  計數模式")
        mode_layout = QVBoxLayout(mode_group)

        self._mode_btn_group = QButtonGroup(self)
        self._radio_wire = QRadioButton("模式 A — Trip Wire（穿越絆線）")
        self._radio_roi  = QRadioButton("模式 B — ROI 進入（進入多邊形區域）")
        self._radio_wire.setChecked(True)
        self._mode_btn_group.addButton(self._radio_wire, 0)
        self._mode_btn_group.addButton(self._radio_roi, 1)
        self._mode_btn_group.buttonClicked.connect(self._on_mode_changed)

        mode_layout.addWidget(self._radio_wire)
        mode_layout.addWidget(self._radio_roi)
        main_layout.addWidget(mode_group)

        # ── Log 輸出目錄 ──────────────────────────────────────────────────
        csv_group = QGroupBox("💾  Log 檔存放位置")
        csv_layout = QVBoxLayout(csv_group)
        csv_layout.setSpacing(6)

        csv_path_row = QHBoxLayout()
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText("logs/")
        self._btn_csv = QPushButton("📂")
        self._btn_csv.setFixedWidth(36)
        self._btn_csv.setToolTip("選擇 Log 檔存放資料夾")
        self._btn_csv.clicked.connect(self._browse_csv_dir)
        csv_path_row.addWidget(self._csv_edit)
        csv_path_row.addWidget(self._btn_csv)
        csv_layout.addLayout(csv_path_row)

        self._btn_log_dir = QPushButton("選擇 Log 存放位置")
        self._btn_log_dir.setFixedHeight(30)
        self._btn_log_dir.clicked.connect(self._browse_csv_dir)
        csv_layout.addWidget(self._btn_log_dir)
        main_layout.addWidget(csv_group)

        # ── 播放速度 ─────────────────────────────────────────────────────
        speed_group = QGroupBox("⏩  播放速度（僅限影片檔）")
        speed_layout = QVBoxLayout(speed_group)
        speed_layout.setSpacing(6)

        self._speed_combo = QComboBox()
        self._speed_options = [
            (0.25, "0.25x"),
            (0.5, "0.5x"),
            (0.75, "0.75x"),
            (1.0, "1x"),
            (1.25, "1.25x"),
            (1.5, "1.5x"),
            (1.75, "1.75x"),
            (2.0, "2x"),
            (3.0, "3x"),
            (4.0, "4x"),
        ]
        for factor, label in self._speed_options:
            self._speed_combo.addItem(label, factor)
        self._speed_combo.setCurrentIndex(3)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self._speed_combo)

        main_layout.addWidget(speed_group)

        # ── 輪播模式（僅影片檔有效）──────────────────────────────────────
        loop_group = QGroupBox("🔁  輪播設定")
        loop_layout = QVBoxLayout(loop_group)
        loop_layout.setSpacing(4)

        self._chk_loop = QCheckBox("影片播完後自動重播（計數持續累計）")
        self._chk_loop.setToolTip(
            "僅適用於本地影片檔。\n"
            "開啟後影片播完會自動從頭重播，計數不重置，每輪累計。"
        )
        loop_layout.addWidget(self._chk_loop)

        self._loop_hint = QLabel("📌 每輪計數累計，不重置")
        self._loop_hint.setStyleSheet("color: #888; font-size: 10px;")
        self._loop_hint.setVisible(False)
        loop_layout.addWidget(self._loop_hint)

        self._chk_loop.toggled.connect(
            lambda checked: self._loop_hint.setVisible(checked)
        )

        main_layout.addWidget(loop_group)

        # ── 其他設定 ──────────────────────────────────────────────────────
        display_group = QGroupBox("🖥️  其他設定")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(10)

        self._chk_show_id = QCheckBox("顯示 Track ID")
        self._chk_show_id.setChecked(True)
        self._chk_show_id.setToolTip("勾選：標籤顯示 ID 編號\n取消：只顯示信心度，隱藏 ID")
        self._chk_show_id.toggled.connect(
            lambda checked: self.show_id_changed.emit(checked)
        )
        display_layout.addWidget(self._chk_show_id)

        self._chk_voice_prompt = QCheckBox("啟用語音提示")
        self._chk_voice_prompt.setToolTip(
            "勾選後，含 B 的物件通過時播放 voice-wav/B.wav；"
            "無 B 時播放 voice-wav/NO_B.wav"
        )
        self._chk_voice_prompt.toggled.connect(self._on_voice_prompt_toggled)
        display_layout.addWidget(self._chk_voice_prompt)

        confidence_row = QHBoxLayout()
        confidence_row.addWidget(QLabel("物件框信心值："))
        self._confidence_combo = QComboBox()
        for value in ( 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80,0.90):
            self._confidence_combo.addItem(f"{value:.2f}", value)
        self._confidence_combo.setToolTip("數值越高越嚴格，誤判較少；數值越低較容易偵測到物件")
        self._confidence_combo.currentIndexChanged.connect(self._on_confidence_changed)
        confidence_row.addWidget(self._confidence_combo, stretch=1)
        display_layout.addLayout(confidence_row)

        main_layout.addWidget(display_group)

        # ── 操作按鈕 ─────────────────────────────────────────────────────
        # 第一行：預覽按鈕
        self._btn_preview = QPushButton("👁  預覽")
        self._btn_preview.setFixedHeight(34)
        self._btn_preview.setCheckable(True)
        self._btn_preview.setToolTip(
            "顯示影片第一幀 / 開始 RTSP-USB 串流預覽\n再次點擊停止預覽"
        )
        main_layout.addWidget(self._btn_preview)

        # 第二行：開始 / 暫停 / 停止
        self._action_buttons_widget = QWidget()
        btn_layout = QHBoxLayout(self._action_buttons_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self._btn_start = QPushButton("▶  開始")
        self._btn_start.setObjectName("primary")
        self._btn_start.setFixedHeight(38)

        self._btn_pause = QPushButton("⏸  暫停")
        self._btn_pause.setEnabled(False)
        self._btn_pause.setFixedHeight(38)

        self._btn_stop = QPushButton("⏹  停止")
        self._btn_stop.setObjectName("danger")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setFixedHeight(38)

        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_pause)
        btn_layout.addWidget(self._btn_stop)
        main_layout.addWidget(self._action_buttons_widget)

        # ── 狀態列 ───────────────────────────────────────────────────────
        self._status_label = QLabel("就緒")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        main_layout.addWidget(self._status_label)

        main_layout.addStretch()

        # 連接按鈕
        self._btn_preview.clicked.connect(self._on_preview)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)

    def _on_voice_prompt_toggled(self, checked: bool) -> None:
        self._settings.voice_prompt_enabled = checked
        self.voice_prompt_changed.emit(checked)

    def _on_confidence_changed(self, index: int) -> None:
        threshold = float(self._confidence_combo.itemData(index))
        self._settings.confidence_threshold = threshold
        self.confidence_threshold_changed.emit(threshold)

    def detach_action_panel(self) -> QWidget:
        """
        Move preview/start/pause/stop controls into a standalone panel.

        MainWindow uses this to place operation controls in a second column
        while this widget keeps owning the signals and button state.
        """
        if hasattr(self, "_action_panel"):
            return self._action_panel

        self._main_layout.removeWidget(self._btn_preview)
        self._main_layout.removeWidget(self._action_buttons_widget)
        self._main_layout.removeWidget(self._status_label)

        self._action_panel = QWidget()
        layout = QVBoxLayout(self._action_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._btn_preview)
        layout.addWidget(self._action_buttons_widget)
        layout.addWidget(self._status_label)
        return self._action_panel

    # ── 槽函數 ────────────────────────────────────────────────────────────────

    def _on_source_type_changed(self, index: int) -> None:
        """切換影像來源類型時，顯示/隱藏對應的輸入欄位。"""
        is_file = index == 0
        is_rtsp = index == 1
        is_usb  = index == 2

        # 顯示控制
        self._file_edit.setVisible(is_file)
        self._btn_browse.setVisible(is_file)
        self._rtsp_edit.setVisible(is_rtsp)
        # rtsp_row label
        for i in range(self._rtsp_row.count()):
            w = self._rtsp_row.itemAt(i).widget()
            if w:
                w.setVisible(is_rtsp)
        self._usb_spin.setVisible(is_usb)
        for i in range(self._usb_row.count()):
            w = self._usb_row.itemAt(i).widget()
            if w:
                w.setVisible(is_usb)

    def _on_preview(self, checked: bool) -> None:
        """預覽按鈕切換。"""
        if self._locked:
            self._btn_preview.setChecked(self._previewing)
            return
        if checked:
            settings = self._collect_settings()
            if settings is None:
                # 驗證失敗，取消 checked 狀態
                self._btn_preview.setChecked(False)
                return
            self._previewing = True
            self._btn_preview.setText("⏹  停止預覽")
            self._set_status("預覽中…", "#4a9eff")
            self.preview_requested.emit(settings)
        else:
            self._previewing = False
            self._btn_preview.setText("👁  預覽")
            self._set_status("預覽已停止", "#888")
            # 停止預覽訊號：用 None 作為特殊標記
            self.preview_requested.emit(None)

    def _on_mode_changed(self, button) -> None:
        mode = CountMode.TRIP_WIRE if button == self._radio_wire else CountMode.ROI_ENTER
        self.mode_changed.emit(mode)

    def _on_speed_changed(self, index: int) -> None:
        """下拉選單選取倍速後發送訊號。"""
        factor = float(self._speed_combo.itemData(index))
        self.speed_changed.emit(factor)

    def _on_start(self) -> None:
        if self._locked:
            return
        settings = self._collect_settings()
        if settings is None:
            return
        self._settings = settings
        self._running = True
        self._paused = False
        # 停止預覽狀態（推論接管畫面）
        self._previewing = False
        self._btn_preview.setChecked(False)
        self._btn_preview.setText("👁  預覽")
        self._set_pause_button_primary(True)
        self._sync_action_button_state()
        self._set_status("推論中…", "#4aff8c")
        self.start_requested.emit(settings)

    def _on_pause(self) -> None:
        if self._locked:
            return
        self._paused = not self._paused
        self._btn_pause.setText("▶  繼續" if self._paused else "⏸  暫停")
        self._set_status("已暫停" if self._paused else "推論中…",
                         "#ffaa44" if self._paused else "#4aff8c")
        self.pause_requested.emit()

    def _on_stop(self) -> None:
        if self._locked:
            return
        self._set_stopped_state()
        self.stop_requested.emit()

    def _set_stopped_state(self) -> None:
        self._running = False
        self._paused = False
        self._btn_pause.setText("⏸  暫停")
        self._set_pause_button_primary(False)
        self._sync_action_button_state()
        self._set_status("已停止", "#ff6666")

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇影片檔", "",
            "影片檔 (*.mp4 *.avi *.mov *.mkv *.ts *.m4v);;所有檔案 (*)"
        )
        if path:
            self._file_edit.setText(path)
            # 選完影片後自動觸發預覽（若目前未在推論中）
            if not self._running:
                self._btn_preview.setChecked(True)
                self._on_preview(True)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 YOLO 模型", "",
            "YOLO 模型 (*.pt);;所有檔案 (*)"
        )
        if path:
            self._model_edit.setText(path)

    def _browse_csv_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇 Log 檔存放位置")
        if path:
            self._csv_edit.setText(path)
            self._settings.csv_dir = path

    # ── 私有工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_project_path(path: str) -> str:
        """Resolve relative paths from the project root."""
        if not path or os.path.isabs(path):
            return path
        return os.path.join(BASE_DIR, path)

    def _collect_settings(self) -> "AppSettings | None":
        """從 UI 收集設定並回傳 AppSettings；驗證失敗回傳 None。"""
        from config import AppSettings, SourceType, CountMode
        import os

        s = AppSettings()
        idx = self._source_combo.currentIndex()
        if idx == 0:
            s.source_type = SourceType.FILE
            s.source_path = self._file_edit.text().strip()
            if not s.source_path:
                self._set_status("請選擇影片檔", "#ff6666")
                return None
            if not os.path.isfile(self._resolve_project_path(s.source_path)):
                self._set_status("影片檔不存在", "#ff6666")
                return None
        elif idx == 1:
            s.source_type = SourceType.RTSP
            s.source_path = self._rtsp_edit.text().strip()
            if not s.source_path:
                self._set_status("請輸入 RTSP URL", "#ff6666")
                return None
        elif idx == 2:
            s.source_type = SourceType.USB
            s.usb_index = self._usb_spin.value()

        model = self._model_edit.text().strip() or "yolov8n.pt"
        s.model_path = model

        s.count_mode = (
            CountMode.TRIP_WIRE if self._radio_wire.isChecked()
            else CountMode.ROI_ENTER
        )

        csv_dir = self._csv_edit.text().strip()
        if csv_dir and not (os.name != "nt" and len(csv_dir) >= 3 and csv_dir[1:3] in (":/", ":\\")):
            s.csv_dir = csv_dir
        else:
            s.csv_dir = DEFAULT_CSV_DIR

        # 保留已設定的 ROI / Wire
        s.roi_points = self._settings.roi_points
        s.wire_points = self._settings.wire_points
        s.last_count = getattr(self._settings, "last_count", 0)
        s.last_count_with_b = getattr(self._settings, "last_count_with_b", 0)
        s.last_count_no_b = getattr(self._settings, "last_count_no_b", 0)

        # 輪播模式（僅 FILE 來源有意義）
        s.loop_video = self._chk_loop.isChecked() and s.source_type == "file"
        s.voice_prompt_enabled = self._chk_voice_prompt.isChecked()
        s.confidence_threshold = float(self._confidence_combo.currentData())

        return s

    def _load_settings(self, s: AppSettings) -> None:
        """將 AppSettings 填入 UI 欄位。"""
        self._settings = s

        type_map = {
            SourceType.FILE: 0,
            SourceType.RTSP: 1,
            SourceType.USB:  2,
        }
        self._source_combo.setCurrentIndex(type_map.get(s.source_type, 0))
        self._file_edit.setText(s.source_path if s.source_type == SourceType.FILE else "")
        self._rtsp_edit.setText(s.source_path if s.source_type == SourceType.RTSP else "")
        self._usb_spin.setValue(s.usb_index)
        self._model_edit.setText(s.model_path)

        if s.count_mode == CountMode.TRIP_WIRE:
            self._radio_wire.setChecked(True)
        else:
            self._radio_roi.setChecked(True)

        self._csv_edit.setText(s.csv_dir)

        # 輪播模式
        self._chk_loop.setChecked(getattr(s, "loop_video", False))
        self._chk_voice_prompt.setChecked(getattr(s, "voice_prompt_enabled", False))
        confidence = float(getattr(s, "confidence_threshold", 0.4))
        closest_index = min(
            range(self._confidence_combo.count()),
            key=lambda i: abs(float(self._confidence_combo.itemData(i)) - confidence),
        )
        self._confidence_combo.setCurrentIndex(closest_index)

    def _set_status(self, text: str, color: str = "#888") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _set_pause_button_primary(self, enabled: bool) -> None:
        """Make pause/continue visually primary while inference is active."""
        self._btn_pause.setObjectName("primary" if enabled else "")
        self._btn_pause.style().unpolish(self._btn_pause)
        self._btn_pause.style().polish(self._btn_pause)

    def _sync_action_button_state(self) -> None:
        """Apply running/locked state to controls that may live outside this panel."""
        unlocked = not self._locked
        self._btn_preview.setEnabled(unlocked and not self._running)
        self._btn_start.setEnabled(unlocked and not self._running)
        self._btn_pause.setEnabled(unlocked and self._running)
        self._btn_stop.setEnabled(unlocked and self._running)

    # ── 公開介面 ──────────────────────────────────────────────────────────────

    def set_stopped(self) -> None:
        """外部呼叫（例如影片播放結束時）強制停止 UI 狀態。"""
        self._set_stopped_state()

    def set_preview_stopped(self) -> None:
        """外部呼叫：預覽執行緒結束後重置預覽按鈕狀態。"""
        self._previewing = False
        self._btn_preview.setChecked(False)
        self._btn_preview.setText("👁  預覽")
        self._sync_action_button_state()
        self._set_status("預覽結束", "#888")

    def set_controls_locked(self, locked: bool) -> None:
        """Lock controls, including detached preview/start/pause/stop buttons."""
        self._locked = locked
        self.setEnabled(not locked)
        if locked:
            self._btn_preview.setEnabled(False)
            self._btn_start.setEnabled(False)
            self._btn_pause.setEnabled(False)
            self._btn_stop.setEnabled(False)
        else:
            self._sync_action_button_state()

    def update_roi_wire(self, roi_points, wire_points) -> None:
        """外部更新 ROI / Wire 設定到內部 settings 快取。"""
        self._settings.roi_points = roi_points
        self._settings.wire_points = wire_points

    @property
    def speed_factor(self) -> float:
        """當前播放速度倍率（0.25 ~ 4.0）。"""
        return float(self._speed_combo.currentData())

    @property
    def current_settings(self) -> AppSettings:
        return self._settings
