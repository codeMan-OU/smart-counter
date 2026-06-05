"""
gui/main_window.py — Qt5 主視窗

職責：
- 整合 VideoWidget / ControlPanel / RoiEditorPanel / Dashboard
- 管理 VideoThread 生命週期
- 處理設定的儲存與載入（JSON）
- 協調各元件之間的訊號/槽連接
"""

from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QScrollArea, QMessageBox, QAction, QMenuBar,
    QStatusBar, QLabel, QInputDialog, QLineEdit, QProgressDialog,
    QPushButton,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QIcon, QCloseEvent, QPixmap, QFont, QFontDatabase

from config import AppSettings, BASE_DIR, SETTINGS_PATH, CountMode, DEFAULT_CSV_DIR
from core.video_source import VideoSource
from core.inference import InferenceEngine, DetectionResult
from core.counter import CounterEngine, CountEvent
from core.csv_logger import CsvLogger

from gui.video_widget import VideoWidget
from gui.control_panel import ControlPanel
from gui.roi_editor import RoiEditorPanel
from gui.dashboard import Dashboard


class ClickableLabel(QLabel):
    """QLabel with a clicked signal."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# 影像推論執行緒
# ──────────────────────────────────────────────────────────────────────────────

class FrameData:
    """執行緒間傳遞的幀資料包。"""

    __slots__ = (
        "frame", "a_detections", "b_detections",
        "counted_ids", "counted_id_has_b",
        "fps", "count", "count_with_b", "count_no_b", "new_event",
    )

    def __init__(
        self,
        frame: np.ndarray,
        a_detections: List[DetectionResult],
        b_detections: List[DetectionResult],
        counted_ids: set,
        counted_id_has_b: Dict[int, bool],
        fps: float,
        count: int,
        count_with_b: int,
        count_no_b: int,
        new_event: Optional[CountEvent],
    ) -> None:
        self.frame = frame
        self.a_detections = a_detections
        self.b_detections = b_detections
        self.counted_ids = counted_ids
        self.counted_id_has_b = counted_id_has_b
        self.fps = fps
        self.count = count
        self.count_with_b = count_with_b
        self.count_no_b = count_no_b
        self.new_event = new_event


class VideoThread(QThread):
    """
    影像擷取 + 推論執行緒。

    執行緒中的工作：
    1. 從 VideoSource 讀取幀
    2. 呼叫 InferenceEngine 推論（同時輸出 A、B 兩類）
    3. 以 B 偵測結果輔助 CounterEngine 更新計數
    4. 透過 frame_ready 訊號將結果送回主執行緒

    輪播模式（loop_video=True，僅 FILE 來源）：
    - 影片播完後自動重新開啟同一檔案，繼續推論
    - CounterEngine 不重置（計數持續累計）
    - InferenceEngine 的 ID remapper 不重置（ID 繼續遞增，不重新從 1）
    - ByteTrack 的 persist 狀態因重建 source 而自然重置，新一輪物件
      會取得新的 raw tracker ID，再由 remapper 映射為連續新 ID

    注意：CounterEngine 與 CsvLogger 操作均在此執行緒，
    GUI 更新只透過訊號/槽進行，保證執行緒安全。
    """

    frame_ready        = pyqtSignal(object)   # FrameData
    error_occurred     = pyqtSignal(str)
    finished_naturally = pyqtSignal()          # 影片播放結束（非輪播時）

    def __init__(
        self,
        settings: AppSettings,
        counter: CounterEngine,
        engine: InferenceEngine,
        logger: CsvLogger,
        speed_factor: float = 1.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._counter = counter
        self._engine = engine          # InferenceEngine 由外部傳入（保持 remapper 狀態）
        self._logger = logger
        self._paused = False
        self._stop_flag = False
        self._speed_factor: float = max(0.25, min(4.0, speed_factor))

    def set_speed(self, factor: float) -> None:
        """動態調整播放速度倍率（執行緒安全，直接賦值 float 為原子操作）。"""
        self._speed_factor = max(0.25, min(4.0, factor))

    def run(self) -> None:
        """執行緒主迴圈，支援輪播模式。"""
        is_file_source = (self._settings.source_type == "file")
        loop_video = self._settings.loop_video and is_file_source

        try:
            # 同步 Counter 尺寸（需先開一次 source 取得解析度）
            probe_source = VideoSource.from_settings(self._settings)
            fw, fh = probe_source.width, probe_source.height
            source_fps = probe_source.fps if probe_source.fps > 0 else 25.0
            probe_source.release()

            self._counter.set_roi(self._settings.roi_points, fw, fh)
            self._counter.set_wire(self._settings.wire_points, fw, fh)
            self._counter.set_mode(self._settings.count_mode)

            fps_timer = time.perf_counter()
            fps_count = 0
            current_fps = 0.0

            while not self._stop_flag:
                # ── 開啟（或重開）影像來源 ────────────────────────────────
                source = VideoSource.from_settings(self._settings)

                # ── 單輪播放迴圈 ──────────────────────────────────────────
                round_ended = False
                while not self._stop_flag:
                    if self._paused:
                        self.msleep(30)
                        continue

                    frame_start = time.perf_counter()

                    ok, frame = source.read()
                    if not ok:
                        round_ended = True
                        break

                    # 推論：同時取得 A 與 B 的偵測結果
                    ignore_a_right_of_x = None
                    if (
                        self._settings.count_mode == CountMode.TRIP_WIRE
                        and len(self._settings.wire_points) == 4
                    ):
                        wire_x1 = int(self._settings.wire_points[0] * fw)
                        wire_x2 = int(self._settings.wire_points[2] * fw)
                        ignore_a_right_of_x = max(wire_x1, wire_x2)

                    a_detections, b_detections = self._engine.infer(
                        frame,
                        ignore_a_right_of_x=ignore_a_right_of_x,
                    )

                    # 計數更新
                    new_event: Optional[CountEvent] = None
                    for det in a_detections:
                        triggered = self._counter.update(det, b_detections)
                        if triggered:
                            new_event = self._counter.history[-1]
                            self._logger.log(new_event)

                    # FPS 計算
                    fps_count += 1
                    elapsed = time.perf_counter() - fps_timer
                    if elapsed >= 1.0:
                        current_fps = fps_count / elapsed
                        fps_count = 0
                        fps_timer = time.perf_counter()

                    # 打包並發送
                    data = FrameData(
                        frame=frame,
                        a_detections=a_detections,
                        b_detections=b_detections,
                        counted_ids=set(self._counter.counted_ids),
                        counted_id_has_b=self._counter.counted_id_has_b,
                        fps=current_fps,
                        count=self._counter.count,
                        count_with_b=self._counter.count_with_b,
                        count_no_b=self._counter.count_no_b,
                        new_event=new_event,
                    )
                    self.frame_ready.emit(data)

                    # 節流 sleep
                    if is_file_source:
                        target_interval = 1.0 / source_fps / self._speed_factor
                        elapsed_frame = time.perf_counter() - frame_start
                        sleep_sec = target_interval - elapsed_frame
                        if sleep_sec > 0.001:
                            self.msleep(int(sleep_sec * 1000))

                source.release()

                # ── 輪播判斷 ──────────────────────────────────────────────
                if round_ended and loop_video and not self._stop_flag:
                    # 輪播：清除 CounterEngine 的 prev_centers 避免殘留
                    # 跨輪首幀無法形成 cross-product，不影響計數正確性
                    self._counter._prev_centers.clear()
                    # counted_ids 保留，避免跨輪重複計數同一物件（理論上不會，但防禦）
                    continue   # 重新進入外層 while，重開影片來源
                else:
                    # 非輪播或已停止：結束
                    if round_ended:
                        self.finished_naturally.emit()
                    break

        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def pause(self) -> None:
        self._paused = not self._paused

    def stop(self) -> None:
        self._stop_flag = True
        self.wait(3000)


# ──────────────────────────────────────────────────────────────────────────────
# 預覽執行緒（不做推論，純讀幀顯示）
# ──────────────────────────────────────────────────────────────────────────────

class PreviewThread(QThread):
    """
    預覽執行緒：讀取影像來源並傳送幀，不執行推論。

    - 影片檔（FILE）：只讀第一幀後停止，靜止預覽
    - RTSP / USB    ：持續讀幀，以來源 FPS 節流，直到外部呼叫 stop()
    """

    preview_frame_ready = pyqtSignal(object)  # np.ndarray
    error_occurred      = pyqtSignal(str)

    _LIVE_PREVIEW_FPS = 15

    def __init__(
        self,
        settings: AppSettings,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._stop_flag = False

    def run(self) -> None:
        source: Optional[VideoSource] = None
        try:
            source = VideoSource.from_settings(self._settings)

            if self._settings.source_type == "file":
                ok, frame = source.read()
                if ok and frame is not None:
                    self.preview_frame_ready.emit(frame)
                return

            interval_ms = int(1000 / self._LIVE_PREVIEW_FPS)
            while not self._stop_flag:
                loop_start = time.perf_counter()
                ok, frame = source.read()
                if ok and frame is not None:
                    self.preview_frame_ready.emit(frame)
                elapsed_ms = int((time.perf_counter() - loop_start) * 1000)
                sleep_ms = max(1, interval_ms - elapsed_ms)
                self.msleep(sleep_ms)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if source:
                source.release()

    def stop(self) -> None:
        self._stop_flag = True
        self.wait(3000)


# ──────────────────────────────────────────────────────────────────────────────
# 模型載入執行緒
# ──────────────────────────────────────────────────────────────────────────────

class ModelLoadThread(QThread):
    """Background loader for YOLO model initialization."""

    model_loaded = pyqtSignal(object)  # InferenceEngine
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._conf_threshold = conf_threshold

    def run(self) -> None:
        try:
            self.model_loaded.emit(InferenceEngine(
                self._model_path,
                conf_threshold=self._conf_threshold,
            ))
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# 主視窗
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    YOLO 物件計數器主視窗。

    佈局：
    ┌─────────────────────────────┬──────────────┐
    │                             │ ControlPanel │
    │      VideoWidget            │ RoiEditor    │
    │                             │ Dashboard    │
    └─────────────────────────────┴──────────────┘
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI智捷方舟-WeiLiang")
        self.resize(1280, 720)

        self._settings: AppSettings = AppSettings()
        self._counter = CounterEngine()
        self._inference_engine: Optional[InferenceEngine] = None  # 延遲建立，換片時保留
        self._logger = CsvLogger(self._settings.csv_dir)
        self._thread: Optional[VideoThread] = None
        self._preview_thread: Optional[PreviewThread] = None
        self._model_loader_thread: Optional[ModelLoadThread] = None
        self._loading_dialog: Optional[QProgressDialog] = None
        self._start_cancelled: bool = False
        self._missing_audio_warnings: set[str] = set()
        self._audio_processes: List[subprocess.Popen] = []
        self._loading_settings: bool = False
        self._controls_locked: bool = False

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._load_settings()
        self._apply_stylesheet()
        QTimer.singleShot(0, self._ask_startup_count_policy)

    # ── UI 建構 ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── 左側：Logo 橫條 + 影像顯示 ───────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Logo 橫條
        header_bar = QWidget()
        header_bar.setFixedHeight(108)
        header_bar.setStyleSheet(
            "background-color: #0e0e1e;"
            "border: 1px solid #2a2a4a;"
            "border-radius: 6px;"
        )
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(12, 8, 18, 8)
        header_layout.setSpacing(18)

        # Logo 圖片
        logo_label = ClickableLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "gui", "logo.png")
        # __file__ 在 main_window.py 時是 gui/ 底下，需往上一層找 gui/logo.png
        # 同時相容從專案根目錄執行的情況
        _candidate_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "logo.png"),
        ]
        logo_pixmap = None
        for _p in _candidate_paths:
            if os.path.isfile(_p):
                _pm = QPixmap(_p)
                if not _pm.isNull():
                    logo_pixmap = _pm
                    break
        if logo_pixmap is not None:
            logo_label.setPixmap(
                logo_pixmap.scaledToHeight(72, Qt.SmoothTransformation)
            )
        else:
            logo_label.setText("🏭")
            logo_label.setStyleSheet("font-size: 42px;")
        logo_label.setFixedWidth(104)
        logo_label.setMinimumHeight(86)
        logo_label.setAlignment(Qt.AlignCenter)
        # 加入這行，改成你想要的底色
        logo_label.setStyleSheet("background-color: #ffffff; border-radius: 6px; padding: 4px;")
        logo_label.setCursor(Qt.PointingHandCursor)
        logo_label.setToolTip("點擊鎖定 / 解鎖操作")
        logo_label.clicked.connect(self._on_logo_clicked)
        header_layout.addWidget(logo_label)

        # 分隔線
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #2a2a4a;")
        header_layout.addWidget(sep)

        # 標題文字
        title_layout = QVBoxLayout()
        title_layout.setSpacing(6)
        title_main = QLabel("AI智捷方舟- 智能計數系統")
        font_main = QFont()
        font_main.setPixelSize(31)   # pixel size 跨平台一致
        font_main.setBold(True)
        title_main.setFont(font_main)
        title_main.setStyleSheet(
            "color: #e0e0ff;"
            "font-size: 31px;"
            "font-weight: 700;"
            "letter-spacing: 2px;"
            "background: transparent;"
            "border: none;"
            "padding: 0;"
        )
        title_sub = QLabel("測試版  ·  YOLO v8 + ByteTrack")
        title_sub.setStyleSheet(
            "color: #6060aa;"
            "font-size: 16px;"
            "letter-spacing: 1px;"
            "background: transparent;"
            "border: none;"
            "padding: 0;"
        )
        title_layout.addWidget(title_main)
        title_layout.addWidget(title_sub)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        left_layout.addWidget(header_bar)

        self._video_widget = VideoWidget()
        self._video_widget.show_placeholder()
        left_layout.addWidget(self._video_widget, stretch=1)

        self._control_panel = ControlPanel(self._settings)
        self._roi_editor = RoiEditorPanel()
        self._dashboard = Dashboard()

        self._settings_panel_expanded = True
        right_panel = QWidget()
        right_panel.setFixedWidth(660)
        right_layout = QHBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._settings_column = QWidget()
        self._settings_column.setFixedWidth(320)
        settings_layout = QVBoxLayout(self._settings_column)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)
        self._btn_collapse_settings = QPushButton("收合設定")
        self._btn_collapse_settings.setFixedHeight(32)
        self._btn_collapse_settings.setToolTip("收合右側設定欄")
        self._btn_collapse_settings.clicked.connect(
            lambda: self._set_settings_column_collapsed(True)
        )
        settings_layout.addWidget(self._btn_collapse_settings)
        settings_layout.addWidget(self._control_panel)
        settings_layout.addStretch()

        self._settings_strip = QWidget()
        self._settings_strip.setFixedWidth(44)
        self._settings_strip.setVisible(False)
        strip_layout = QVBoxLayout(self._settings_strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.setSpacing(0)
        self._btn_expand_settings = QPushButton("設定")
        self._btn_expand_settings.setFixedWidth(44)
        self._btn_expand_settings.setMinimumHeight(96)
        self._btn_expand_settings.setToolTip("展開右側設定欄")
        self._btn_expand_settings.clicked.connect(
            lambda: self._set_settings_column_collapsed(False)
        )
        strip_layout.addWidget(self._btn_expand_settings)
        strip_layout.addStretch()

        operations_column = QWidget()
        operations_column.setMinimumWidth(320)
        operations_layout = QVBoxLayout(operations_column)
        operations_layout.setContentsMargins(0, 0, 0, 0)
        operations_layout.setSpacing(8)
        operations_layout.addWidget(self._control_panel.detach_action_panel())
        operations_layout.addWidget(self._roi_editor)
        operations_layout.addWidget(self._dashboard)
        operations_layout.addStretch()

        right_layout.addWidget(operations_column, stretch=1)
        right_layout.addWidget(self._settings_column)
        right_layout.addWidget(self._settings_strip)

        scroll = QScrollArea()
        scroll.setWidget(right_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(676)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(scroll)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就緒")

    def _set_settings_column_collapsed(self, collapsed: bool) -> None:
        self._settings_panel_expanded = not collapsed
        self._settings_column.setVisible(not collapsed)
        self._settings_strip.setVisible(collapsed)
        self._status_bar.showMessage("設定欄已收合" if collapsed else "設定欄已展開")

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("檔案")
        save_action = QAction("儲存設定", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_settings)
        file_menu.addAction(save_action)

        load_action = QAction("載入設定", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_settings)
        file_menu.addAction(load_action)

        file_menu.addSeparator()
        exit_action = QAction("離開", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("說明")
        about_action = QAction("關於", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ── 訊號連接 ──────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._control_panel.start_requested.connect(self._on_start)
        self._control_panel.pause_requested.connect(self._on_pause)
        self._control_panel.stop_requested.connect(self._on_stop)
        self._control_panel.mode_changed.connect(self._on_mode_changed)
        self._control_panel.speed_changed.connect(self._on_speed_changed)
        self._control_panel.preview_requested.connect(self._on_preview_requested)
        self._control_panel.show_id_changed.connect(self._video_widget.set_show_id)
        self._control_panel.voice_prompt_changed.connect(self._on_voice_prompt_changed)
        self._control_panel.confidence_threshold_changed.connect(
            self._on_confidence_threshold_changed
        )

        self._roi_editor.edit_roi_requested.connect(
            lambda: self._video_widget.set_edit_mode(True, CountMode.ROI_ENTER)
        )
        self._roi_editor.edit_wire_requested.connect(
            lambda: self._video_widget.set_edit_mode(True, CountMode.TRIP_WIRE)
        )
        self._roi_editor.edit_cancelled.connect(
            lambda: self._video_widget.set_edit_mode(False)
        )
        self._roi_editor.clear_roi_requested.connect(self._on_clear_roi)
        self._roi_editor.clear_wire_requested.connect(self._on_clear_wire)

        self._video_widget.roi_updated.connect(self._on_roi_updated)
        self._video_widget.wire_updated.connect(self._on_wire_updated)

        self._dashboard.reset_requested.connect(self._on_reset_count)
        self._dashboard.label_names_changed.connect(self._on_label_names_changed)

    # ── 槽函數：推論控制 ──────────────────────────────────────────────────────

    def _on_start(self, settings: AppSettings) -> None:
        """開始推論執行緒（先停預覽執行緒）。"""
        if self._controls_locked:
            return
        if self._thread and self._thread.isRunning():
            return
        if self._model_loader_thread and self._model_loader_thread.isRunning():
            return

        self._stop_preview()
        settings.last_count = self._counter.count
        settings.last_count_with_b = self._counter.count_with_b
        settings.last_count_no_b = self._counter.count_no_b
        self._settings = settings
        self._start_cancelled = False

        # 換影片來源時，重建 InferenceEngine 並重置 ID remapper
        # 若模型路徑未改變，重建成本在可接受範圍內（避免跨影片 ID 污染）
        self._show_loading_dialog()

        self._model_loader_thread = ModelLoadThread(
            settings.model_path,
            getattr(settings, "confidence_threshold", 0.4),
        )
        self._model_loader_thread.model_loaded.connect(
            self._on_model_loaded, Qt.QueuedConnection
        )
        self._model_loader_thread.error_occurred.connect(
            self._on_model_load_error, Qt.QueuedConnection
        )
        self._model_loader_thread.finished.connect(
            self._on_model_loader_finished, Qt.QueuedConnection
        )
        self._model_loader_thread.start()

    def _on_model_loaded(self, engine: InferenceEngine) -> None:
        if self._start_cancelled:
            self._hide_loading_dialog()
            self._inference_engine = None
            self._status_bar.showMessage("已停止")
            return
        self._inference_engine = engine
        self._inference_engine.set_next_mapped_id(
            getattr(self._settings, "last_count", 0) + 1
        )
        if self._loading_dialog is not None:
            self._loading_dialog.setLabelText("模型已載入，正在啟動推論並等待第一個畫面…")

        settings = self._settings
        self._logger = CsvLogger(settings.csv_dir)
        self._counter.full_reset()
        self._counter.set_counts(
            getattr(settings, "last_count", 0),
            getattr(settings, "last_count_with_b", 0),
            getattr(settings, "last_count_no_b", 0),
        )
        self._dashboard.clear_table()
        self._dashboard.update_count(
            self._counter.count,
            self._counter.count_with_b,
            self._counter.count_no_b,
        )
        self._dashboard.update_mode(settings.count_mode)
        self._dashboard.update_csv_path(None)

        self._thread = VideoThread(
            settings,
            self._counter,
            self._inference_engine,
            self._logger,
            speed_factor=self._control_panel.speed_factor,
        )
        self._thread.frame_ready.connect(self._on_frame_ready, Qt.QueuedConnection)
        self._thread.error_occurred.connect(self._on_thread_error)
        self._thread.finished_naturally.connect(self._on_video_ended)
        self._thread.start()

        loop_status = "  🔁 輪播中" if settings.loop_video else ""
        self._status_bar.showMessage(f"推論中…{loop_status}")

    def _on_model_load_error(self, message: str) -> None:
        self._hide_loading_dialog()
        self._start_cancelled = False
        self._control_panel.set_stopped()
        self._status_bar.showMessage(f"模型載入失敗：{message}")
        QMessageBox.critical(self, "模型載入失敗", f"無法載入模型：\n{message}")

    def _on_model_loader_finished(self) -> None:
        self._model_loader_thread = None

    def _show_loading_dialog(self) -> None:
        self._loading_dialog = QProgressDialog(
            "正在載入模型並準備推論，請稍候…",
            "",
            0,
            0,
            self,
        )
        self._loading_dialog.setWindowTitle("準備中")
        self._loading_dialog.setWindowModality(Qt.ApplicationModal)
        self._loading_dialog.setCancelButton(None)
        self._loading_dialog.setMinimumDuration(0)
        self._loading_dialog.setAutoClose(False)
        self._loading_dialog.setAutoReset(False)
        self._loading_dialog.show()
        self._status_bar.showMessage("正在載入模型…")

    def _hide_loading_dialog(self) -> None:
        if self._loading_dialog is not None:
            self._loading_dialog.close()
            self._loading_dialog = None

    # ── 槽函數：預覽 ──────────────────────────────────────────────────────────

    def _on_preview_requested(self, settings: Optional[AppSettings]) -> None:
        if self._controls_locked:
            return
        if settings is None:
            self._stop_preview()
            self._video_widget.show_placeholder()
            self._status_bar.showMessage("預覽已停止")
            return

        self._stop_preview()

        self._settings = settings
        self._preview_thread = PreviewThread(settings)
        self._preview_thread.preview_frame_ready.connect(
            self._on_preview_frame, Qt.QueuedConnection
        )
        self._preview_thread.error_occurred.connect(self._on_preview_error)
        self._preview_thread.finished.connect(self._on_preview_finished)
        self._preview_thread.start()
        self._status_bar.showMessage("預覽中…")

    def _on_preview_frame(self, frame: np.ndarray) -> None:
        self._video_widget.update_preview_frame(frame, self._settings.count_mode)

    def _on_preview_error(self, message: str) -> None:
        self._control_panel.set_preview_stopped()
        self._status_bar.showMessage(f"預覽失敗：{message}")
        QMessageBox.warning(self, "預覽失敗", f"無法開啟影像來源：\n{message}")

    def _on_preview_finished(self) -> None:
        self._status_bar.showMessage("預覽就緒，可開始繪製 ROI / 絆線")

    def _stop_preview(self) -> None:
        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.stop()
        self._preview_thread = None

    def _on_pause(self) -> None:
        if self._controls_locked:
            return
        if self._thread and self._thread.isRunning():
            self._thread.pause()

    def _on_stop(self) -> None:
        if self._controls_locked:
            return
        if self._model_loader_thread and self._model_loader_thread.isRunning():
            self._start_cancelled = True
            self._hide_loading_dialog()
            self._status_bar.showMessage("正在取消啟動，請稍候…")
        elif self._loading_dialog is not None:
            self._hide_loading_dialog()
        if self._thread:
            self._thread.stop()
            self._thread = None
        self._logger.close()
        self._status_bar.showMessage("已停止")

    def _on_logo_clicked(self) -> None:
        if self._controls_locked:
            password, ok = QInputDialog.getText(
                self,
                "解除鎖定",
                "請輸入解鎖密碼：000",
                QLineEdit.Normal,
                "",
            )
            if not ok:
                return
            if password != "000":
                QMessageBox.warning(self, "密碼錯誤", "密碼不正確，無法解除鎖定。")
                return
            self._set_controls_locked(False)
            return

        confirm = QMessageBox.question(
            self,
            "確認鎖定",
            "鎖定後所有設定與操作按鈕都不能更動。\n確定要鎖定嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            self._set_controls_locked(True)

    def _set_controls_locked(self, locked: bool) -> None:
        self._controls_locked = locked
        self._control_panel.set_controls_locked(locked)
        self._roi_editor.setEnabled(not locked)
        self._dashboard.set_interaction_locked(locked)
        self._video_widget.set_edit_mode(False)
        self._status_bar.showMessage("操作已鎖定" if locked else "操作已解除鎖定")

    # ── 槽函數：幀資料接收 ────────────────────────────────────────────────────

    def _on_frame_ready(self, data: FrameData) -> None:
        """接收來自執行緒的幀資料，更新 UI（主執行緒）。"""
        self._hide_loading_dialog()

        roi_poly = self._counter.roi_polygon
        wire_p1, wire_p2 = self._counter.wire_points

        self._video_widget.update_frame(
            frame=data.frame,
            a_detections=data.a_detections,
            b_detections=data.b_detections,
            counted_ids=data.counted_ids,
            counted_id_has_b=data.counted_id_has_b,
            roi_polygon=roi_poly,
            wire_p1=wire_p1,
            wire_p2=wire_p2,
            mode=self._settings.count_mode,
        )

        self._dashboard.update_count(
            data.count,
            data.count_with_b,
            data.count_no_b,
        )
        self._dashboard.update_fps(data.fps)

        if data.new_event is not None:
            self._dashboard.add_event(data.new_event)
            self._dashboard.update_csv_path(self._logger.filepath)
            self._remember_current_counts()
            self._save_settings()

        loop_tag = "  🔁" if self._settings.loop_video else ""
        self._status_bar.showMessage(
            f"推論中{loop_tag}  FPS: {data.fps:.1f}  "
            f"計數: {data.count}  "
            f"含B: {data.count_with_b}  "
            f"無B: {data.count_no_b}"
        )

        if data.new_event is not None:
            self._play_voice_prompt(data.new_event.has_b)

    # ── 槽函數：ROI / Wire ────────────────────────────────────────────────────

    def _on_roi_updated(self, points: list) -> None:
        if self._controls_locked:
            return
        self._settings.roi_points = points
        self._control_panel.update_roi_wire(
            self._settings.roi_points, self._settings.wire_points
        )
        self._roi_editor.on_roi_set(len(points))
        self._video_widget.set_edit_mode(False)

        if self._thread and self._thread.isRunning():
            self._sync_counter_geometry()

        self._video_widget.repaint_with_current_settings(self._settings.count_mode)
        self._save_settings()

    def _on_wire_updated(self, wire: list) -> None:
        if self._controls_locked:
            return
        self._settings.wire_points = wire
        self._control_panel.update_roi_wire(
            self._settings.roi_points, self._settings.wire_points
        )
        self._roi_editor.on_wire_set()
        self._video_widget.set_edit_mode(False)

        if self._thread and self._thread.isRunning():
            self._sync_counter_geometry()

        self._video_widget.repaint_with_current_settings(self._settings.count_mode)
        self._save_settings()

    def _on_clear_roi(self) -> None:
        if self._controls_locked:
            return
        self._settings.roi_points = []
        self._counter.set_roi([], 1, 1)
        self._roi_editor.on_roi_cleared()
        self._video_widget.set_roi_norm([])
        self._video_widget.repaint_with_current_settings(self._settings.count_mode)
        self._save_settings()

    def _on_clear_wire(self) -> None:
        if self._controls_locked:
            return
        self._settings.wire_points = []
        self._counter.set_wire([], 1, 1)
        self._roi_editor.on_wire_cleared()
        self._video_widget.set_wire_norm([])
        self._video_widget.repaint_with_current_settings(self._settings.count_mode)
        self._save_settings()

    def _sync_counter_geometry(self) -> None:
        """將 settings 中的 ROI / Wire 正規化座標同步到 CounterEngine。"""
        fw = self._video_widget._frame_w
        fh = self._video_widget._frame_h
        if fw > 1 and fh > 1:
            self._counter.set_roi(self._settings.roi_points, fw, fh)
            self._counter.set_wire(self._settings.wire_points, fw, fh)

    # ── 槽函數：其他 ──────────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str) -> None:
        if self._controls_locked:
            return
        self._settings.count_mode = mode
        self._counter.set_mode(mode)
        self._dashboard.update_mode(mode)

    def _on_voice_prompt_changed(self, enabled: bool) -> None:
        if self._controls_locked:
            return
        self._settings.voice_prompt_enabled = enabled
        if self._loading_settings:
            return
        self._save_settings()

    def _on_confidence_threshold_changed(self, threshold: float) -> None:
        if self._controls_locked:
            return
        self._settings.confidence_threshold = threshold
        if self._inference_engine is not None:
            self._inference_engine.set_confidence_threshold(threshold)
        if self._loading_settings:
            return
        self._save_settings()

    def _on_label_names_changed(
        self,
        count_with_b_name: str,
        count_no_b_name: str,
    ) -> None:
        if self._controls_locked:
            return
        self._settings.count_with_b_name = count_with_b_name
        self._settings.count_no_b_name = count_no_b_name
        if self._loading_settings:
            return
        self._save_settings()

    def _on_speed_changed(self, factor: float) -> None:
        if self._controls_locked:
            return
        if self._thread and self._thread.isRunning():
            self._thread.set_speed(factor)

    def _play_voice_prompt(self, has_b: bool) -> None:
        """Play non-blocking voice prompt for a count event."""
        if not getattr(self._settings, "voice_prompt_enabled", False):
            return

        filename = "B.wav" if has_b else "NO_B.wav"
        filepath = os.path.join(BASE_DIR, "voice-wav", filename)
        if not os.path.isfile(filepath):
            if filename not in self._missing_audio_warnings:
                self._missing_audio_warnings.add(filename)
                self._status_bar.showMessage(f"找不到語音檔：{filepath}")
            return

        try:
            self._play_wav_file(filepath)
        except Exception as exc:
            self._status_bar.showMessage(f"語音提示播放失敗：{exc}")

    def _play_wav_file(self, filepath: str) -> None:
        """Play a WAV file asynchronously on Windows, macOS, and common Linux desktops."""
        system = platform.system().lower()
        if system == "windows":
            import winsound
            winsound.PlaySound(
                filepath,
                winsound.SND_FILENAME | winsound.SND_ASYNC,
            )
            return

        player = None
        args: List[str] = []
        if system == "darwin":
            player = shutil.which("afplay")
            if player:
                args = [player, filepath]
        else:
            for candidate in ("paplay", "aplay", "pw-play", "ffplay"):
                player = shutil.which(candidate)
                if not player:
                    continue
                if candidate == "ffplay":
                    args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", filepath]
                else:
                    args = [player, filepath]
                break

        if not args:
            raise RuntimeError("找不到可用的音訊播放程式")

        self._audio_processes = [
            proc for proc in self._audio_processes
            if proc.poll() is None
        ]
        self._audio_processes.append(
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    def _on_reset_count(self) -> None:
        if self._controls_locked:
            return
        confirm = QMessageBox.question(
            self,
            "確認重置",
            "確定要重置目前計數與計數歷史嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        password, ok = QInputDialog.getText(
            self,
            "重置密碼",
            "請輸入重置密碼：000",
            QLineEdit.Normal,
            "",
        )
        if not ok:
            return
        if password != "000":
            QMessageBox.warning(self, "密碼錯誤", "密碼不正確，未執行重置。")
            return

        self._counter.reset_count()
        self._dashboard.update_count(0, 0, 0)
        self._dashboard.clear_table()
        self._logger.new_session()
        self._dashboard.update_csv_path(None)
        self._remember_current_counts()
        self._save_settings()

    def _on_thread_error(self, message: str) -> None:
        self._hide_loading_dialog()
        QMessageBox.critical(self, "執行緒錯誤", f"推論執行緒發生錯誤：\n{message}")
        self._control_panel.set_stopped()
        self._status_bar.showMessage(f"錯誤：{message}")

    def _on_video_ended(self) -> None:
        """影片播放自然結束（非輪播情況下才會收到此訊號）。"""
        self._control_panel.set_stopped()
        self._status_bar.showMessage(
            f"影片播放結束  最終計數：{self._counter.count}  "
            f"含B：{self._counter.count_with_b}  "
            f"無B：{self._counter.count_no_b}"
        )

    # ── 設定儲存 / 載入 ───────────────────────────────────────────────────────

    def _save_settings(self) -> None:
        try:
            self._remember_current_counts()
            if hasattr(self, "_dashboard"):
                (
                    self._settings.count_with_b_name,
                    self._settings.count_no_b_name,
                ) = self._dashboard.label_names()
            self._settings.model_path = self._portable_project_path(
                self._settings.model_path
            )
            if self._settings.source_type == "file":
                self._settings.source_path = self._portable_project_path(
                    self._settings.source_path
                )
            data = asdict(self._settings)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._status_bar.showMessage(f"設定儲存失敗：{exc}")

    def _load_settings(self) -> None:
        if not os.path.isfile(SETTINGS_PATH):
            self._settings.model_path = self._project_file_or_fallback(
                "",
                "PTmodel",
                (".pt",),
            )
            self._settings.source_path = self._project_file_or_fallback(
                "",
                "video",
                (".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"),
            )
            self._control_panel._load_settings(self._settings)
            self._save_settings()
            return
        try:
            self._loading_settings = True
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            s = AppSettings(**{
                k: v for k, v in data.items()
                if k in AppSettings.__dataclass_fields__
            })
            s.csv_dir = self._portable_csv_dir(s.csv_dir)
            s.model_path = self._project_file_or_fallback(
                s.model_path,
                "PTmodel",
                (".pt",),
            )
            if s.source_type == "file":
                s.source_path = self._project_file_or_fallback(
                    s.source_path,
                    "video",
                    (".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"),
                )
            self._settings = s
            self._counter.set_counts(
                getattr(s, "last_count", 0),
                getattr(s, "last_count_with_b", 0),
                getattr(s, "last_count_no_b", 0),
            )
            self._dashboard.set_label_names(
                getattr(s, "count_with_b_name", ""),
                getattr(s, "count_no_b_name", ""),
            )
            self._dashboard.update_count(
                self._counter.count,
                self._counter.count_with_b,
                self._counter.count_no_b,
            )
            self._control_panel._load_settings(s)
            self._video_widget.set_roi_norm(s.roi_points)
            self._video_widget.set_wire_norm(s.wire_points)

            if s.roi_points:
                self._roi_editor.on_roi_set(len(s.roi_points))
            if s.wire_points:
                self._roi_editor.on_wire_set()

            self._sync_counter_geometry()
            self._video_widget.repaint_with_current_settings(s.count_mode)
            self._status_bar.showMessage("設定已載入")
        except Exception as exc:
            self._status_bar.showMessage(f"設定載入失敗：{exc}")
        finally:
            self._loading_settings = False

    def _remember_current_counts(self) -> None:
        """Store current counter totals in settings so app restart can restore them."""
        self._settings.last_count = self._counter.count
        self._settings.last_count_with_b = self._counter.count_with_b
        self._settings.last_count_no_b = self._counter.count_no_b

    def _resolve_project_path(self, path: str) -> str:
        """Resolve a relative project path to an absolute filesystem path."""
        if not path or os.path.isabs(path):
            return path
        return os.path.join(BASE_DIR, path)

    def _portable_project_path(self, path: str) -> str:
        """Store project-local files as relative paths for easier folder moves."""
        if not path:
            return path

        resolved = os.path.abspath(self._resolve_project_path(path))
        base = os.path.abspath(BASE_DIR)
        try:
            if os.path.commonpath([resolved, base]) == base:
                return os.path.relpath(resolved, base).replace("\\", "/")
        except ValueError:
            pass
        return path

    def _find_first_project_file(
        self,
        folder_name: str,
        extensions: tuple[str, ...],
    ) -> str:
        """Return the first matching project file as a portable relative path."""
        folder = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder):
            return ""

        normalized_exts = tuple(ext.lower() for ext in extensions)
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(normalized_exts):
                return self._portable_project_path(path)
        return ""

    def _project_file_or_fallback(
        self,
        path: str,
        folder_name: str,
        extensions: tuple[str, ...],
    ) -> str:
        """
        Keep an existing path, recover moved project files by basename, or choose
        the first file in the expected project folder.
        """
        if path and os.path.isfile(self._resolve_project_path(path)):
            return self._portable_project_path(path)

        basename = os.path.basename(path) if path else ""
        if basename:
            same_name = os.path.join(BASE_DIR, folder_name, basename)
            if os.path.isfile(same_name):
                return self._portable_project_path(same_name)

        fallback = self._find_first_project_file(folder_name, extensions)
        return fallback or path

    def _ask_startup_count_policy(self) -> None:
        """Ask whether to continue the previous totals or start from zero."""
        count = getattr(self._settings, "last_count", 0)
        count_with_b = getattr(self._settings, "last_count_with_b", 0)
        count_no_b = getattr(self._settings, "last_count_no_b", 0)

        box = QMessageBox(self)
        box.setWindowTitle("計數設定")
        box.setIcon(QMessageBox.Question)
        box.setText("是否要將本次計數歸零？")
        box.setInformativeText(
            f"目前保存計數：總計 {count}，含 B {count_with_b}，無 B {count_no_b}\n\n"
            "選擇「不歸零」：計數與顯示 ID 會延續上一次。\n"
            "選擇「歸零」：計數歸零，顯示 ID 從 1 開始。"
        )
        keep_button = box.addButton("不歸零，延續上次", QMessageBox.NoRole)
        reset_button = box.addButton("歸零，ID 從 1 開始", QMessageBox.YesRole)
        box.setDefaultButton(keep_button)
        box.exec_()

        if box.clickedButton() != reset_button:
            self._status_bar.showMessage("已延續上次計數與 ID")
            return

        confirm = QMessageBox.question(
            self,
            "再次確認歸零",
            "確定要將計數歸零，並讓 ID 從 1 開始嗎？\n\n"
            "歸零前的目前資訊會先自動寫入 txt 紀錄檔。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            self._status_bar.showMessage("已取消歸零，延續上次計數與 ID")
            return

        snapshot_path = self._write_reset_snapshot()

        self._counter.full_reset()
        self._settings.last_count = 0
        self._settings.last_count_with_b = 0
        self._settings.last_count_no_b = 0
        self._dashboard.update_count(0, 0, 0)
        self._dashboard.clear_table()
        self._dashboard.update_csv_path(None)
        if self._inference_engine is not None:
            self._inference_engine.reset_id_map()
        self._save_settings()
        self._status_bar.showMessage(f"計數已歸零，紀錄已保存：{snapshot_path}")

    def _write_reset_snapshot(self) -> str:
        """Write a text snapshot before startup reset."""
        now = datetime.now()
        logs_dir = os.path.join(BASE_DIR, DEFAULT_CSV_DIR)
        os.makedirs(logs_dir, exist_ok=True)
        filepath = os.path.join(
            logs_dir,
            f"reset_snapshot_{now.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        mode_name = {
            CountMode.TRIP_WIRE: "Trip Wire（穿越計數）",
            CountMode.ROI_ENTER: "ROI Enter（進入計數）",
        }.get(self._settings.count_mode, self._settings.count_mode)

        lines = [
            "歸零前資訊紀錄",
            "=" * 40,
            f"紀錄時間：{now.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "計數資訊",
            f"總計：{self._counter.count}",
            f"含 B：{self._counter.count_with_b}",
            f"無 B：{self._counter.count_no_b}",
            "",
            "設定資訊",
            f"影像來源類型：{self._settings.source_type}",
            f"影像來源：{self._settings.source_path}",
            f"USB 編號：{self._settings.usb_index}",
            f"模型路徑：{self._settings.model_path}",
            f"計數模式：{mode_name}",
            f"CSV 目錄：{self._settings.csv_dir}",
            f"輪播模式：{self._settings.loop_video}",
            f"語音提示：{self._settings.voice_prompt_enabled}",
            f"物件框信心值：{self._settings.confidence_threshold:.2f}",
            f"含 B 顯示名稱：{self._settings.count_with_b_name}",
            f"無 B 顯示名稱：{self._settings.count_no_b_name}",
            "",
            "ROI / 絆線",
            f"ROI 點位：{json.dumps(self._settings.roi_points, ensure_ascii=False)}",
            f"絆線點位：{json.dumps(self._settings.wire_points, ensure_ascii=False)}",
            "",
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def _portable_csv_dir(self, path: str) -> str:
        """Use the project logs folder when a Windows drive path is loaded on Unix."""
        if platform.system().lower() == "windows":
            return path
        if len(path) >= 3 and path[1:3] in (":/", ":\\"):
            return DEFAULT_CSV_DIR
        return path

    # ── 關閉事件 ──────────────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:
        confirm = QMessageBox.question(
            self,
            "確認離開",
            "確定要離開程式嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            event.ignore()
            return

        if self._model_loader_thread and self._model_loader_thread.isRunning():
            QMessageBox.information(
                self,
                "模型載入中",
                "模型仍在載入中，請稍候完成後再離開。",
            )
            event.ignore()
            return

        self._stop_preview()
        if self._thread and self._thread.isRunning():
            self._thread.stop()
        self._logger.close()
        self._save_settings()
        event.accept()

    # ── 說明 ──────────────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "關於 YOLO 物件計數器",
            "<b>YOLO 物件計數器 v1.0</b><br><br>"
            "基於 YOLOv8 + ByteTrack 的工業物件計數系統<br>"
            "支援多邊形 ROI、Trip Wire 穿越計數<br>"
            "支援 A / B 雙類別偵測，分別統計含B / 無B 計數<br>"
            "支援本地影片、RTSP 串流、USB 攝影機<br>"
            "支援輪播模式（影片播完自動重播，計數持續累計）<br><br>"
            "技術棧：Python / PyQt5 / OpenCV / Ultralytics",
        )

    # ── 全域樣式 ──────────────────────────────────────────────────────────────

    def _ui_font_family(self) -> str:
        preferred = [
            "Microsoft JhengHei",   # Windows
            "PingFang TC",          # macOS
            "Heiti TC",             # older macOS
            "Noto Sans CJK TC",     # Linux
            "Noto Sans TC",
            "WenQuanYi Zen Hei",
        ]
        families = set(QFontDatabase().families())
        for family in preferred:
            if family in families:
                return family
        return self.font().family()

    def _apply_stylesheet(self) -> None:
        ui_font = self._ui_font_family()
        stylesheet = """
            QMainWindow, QWidget {
                background-color: #12121e;
                color: #d0d0e0;
                font-family: "__UI_FONT__";
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px 6px 6px 6px;
                font-size: 12px;
                color: #8888aa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                color: #9090bb;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #1e1e36;
                border: 1px solid #3a3a5a;
                border-radius: 4px;
                padding: 4px 6px;
                color: #d0d0e0;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #4a6aff;
            }
            QPushButton {
                background-color: #2a2a4a;
                border: 1px solid #4a4a7a;
                border-radius: 4px;
                padding: 5px 10px;
                color: #d0d0e0;
            }
            QPushButton:hover {
                background-color: #3a3a6a;
            }
            QPushButton:pressed {
                background-color: #1a1a3a;
            }
            QPushButton:disabled {
                background-color: #1a1a2a;
                color: #555;
                border-color: #333;
            }
            QPushButton#primary {
                background-color: #1a4aff;
                border-color: #3a6aff;
                color: white;
                font-weight: bold;
            }
            QPushButton#primary:hover {
                background-color: #2a5aff;
            }
            QPushButton#primary:disabled {
                background-color: #1a1a2a;
                color: #555;
                border-color: #333;
                font-weight: bold;
            }
            QPushButton#danger {
                background-color: #4a1a1a;
                border-color: #7a2a2a;
                color: #ff6666;
            }
            QPushButton#danger:hover {
                background-color: #6a2a2a;
            }
            QPushButton:checkable:checked {
                background-color: #1a3a1a;
                border-color: #2a8a2a;
                color: #4aff8c;
            }
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #4a4a7a;
                border-radius: 3px;
                background: #1e1e36;
            }
            QCheckBox::indicator:checked {
                background: #4a6aff;
                border-color: #6a8aff;
            }
            QRadioButton {
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #4a4a7a;
                border-radius: 7px;
                background: #1e1e36;
            }
            QRadioButton::indicator:checked {
                background: #4a6aff;
                border-color: #6a8aff;
            }
            QScrollBar:vertical {
                width: 6px;
                background: #1a1a2e;
            }
            QScrollBar::handle:vertical {
                background: #3a3a6a;
                border-radius: 3px;
            }
            QMenuBar {
                background-color: #0e0e1e;
                color: #aaa;
            }
            QMenuBar::item:selected {
                background-color: #2a2a4a;
            }
            QMenu {
                background-color: #1a1a2e;
                border: 1px solid #2a2a4a;
            }
            QMenu::item:selected {
                background-color: #2a2a4a;
            }
            QStatusBar {
                background-color: #0e0e1e;
                color: #666;
                font-size: 11px;
            }
        """
        self.setStyleSheet(stylesheet.replace("__UI_FONT__", ui_font))
