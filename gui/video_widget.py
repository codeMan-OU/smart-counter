"""
gui/video_widget.py — 影像顯示 Qt 元件

職責：
- 將 OpenCV BGR frame 轉換為 QImage 顯示
- 疊加繪製 BBox、Track ID、ROI 多邊形、Trip Wire
- A 物件框依狀態顯示三種顏色：
    · 未計數              → BBOX_COLOR（黃色）
    · 已計數 + 含 B       → BBOX_COUNTED_WITH_B（綠色）
    · 已計數 + 無 B       → BBOX_COUNTED_NO_B（暗黃色）
- B 物件框：僅在中心點落在某 A 的 BBox 內時才繪製（青色）
- 在 ROI 編輯模式下處理滑鼠事件（多邊形頂點 / 線段端點定義）
- 預覽模式：在靜止幀上即時疊加 ROI / Wire 編輯中的線條

中文字型支援：
- 使用 Pillow 繪製中文文字，自動偵測 Windows / Linux 系統字型
- 找不到中文字型時自動 fallback 為英文提示
"""

from __future__ import annotations
import os
import sys
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap

from PIL import Image, ImageDraw, ImageFont

from config import (
    ROI_COLOR, ROI_ALPHA, ROI_LINE_THICKNESS, ROI_POINT_RADIUS,
    WIRE_COLOR, WIRE_THICKNESS,
    BBOX_COLOR, BBOX_COUNTED_WITH_B, BBOX_COUNTED_NO_B, BBOX_THICKNESS,
    BBOX_B_COLOR, BBOX_B_THICKNESS,
    TEXT_SCALE, TEXT_THICKNESS, CountMode,
)
from core.inference import DetectionResult


# ── 中文字型渲染器 ────────────────────────────────────────────────────────────

class CvTextRenderer:
    """
    在 OpenCV frame 上安全繪製含中文字元的文字。

    策略（依優先序）：
    1. Windows  → 微軟正黑體 / 新細明體
    2. Linux    → Noto Sans CJK TC / SC / JP（apt: fonts-noto-cjk）
    3. Fallback → 將中文字元替換為 ASCII，仍用 cv2.putText 渲染

    字型只在第一次使用時載入，後續從快取取得。
    """

    # 各平台字型候選路徑（依優先序排列）
    _FONT_CANDIDATES: List[str] = [
        # Windows
        "C:/Windows/Fonts/msjhbd.ttc",   # 微軟正黑體 Bold
        "C:/Windows/Fonts/msjh.ttc",     # 微軟正黑體
        "C:/Windows/Fonts/mingliu.ttc",  # 新細明體
        # Ubuntu / Debian (apt install fonts-noto-cjk)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # Jetson / Orin Nano (JetPack Ubuntu)
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]

    def __init__(self) -> None:
        # key: font_size(int) → ImageFont
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._font_path: Optional[str] = self._detect_font_path()
        self._has_cjk: bool = self._font_path is not None

    # ── 公開介面 ──────────────────────────────────────────────────────────────

    @property
    def has_cjk(self) -> bool:
        """是否成功載入 CJK 字型。"""
        return self._has_cjk

    def put_text(
        self,
        frame: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_size: int,
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """
        在 frame 上繪製文字（支援中文）。

        Args:
            frame     : BGR numpy array（in-place 修改）
            text      : 要繪製的文字（可含中文）
            org       : 文字左下角座標 (x, y)，與 cv2.putText 相同
            font_size : PIL 字型大小（像素）
            color     : BGR 顏色 tuple
            thickness : 僅 fallback 模式使用（cv2.putText 參數）
        """
        if self._has_cjk:
            self._pil_put_text(frame, text, org, font_size, color)
        else:
            # Fallback：將非 ASCII 字元替換後用 cv2.putText 繪製
            ascii_text = self._to_ascii_fallback(text)
            cv_scale = font_size / 28.0   # 28px ≈ cv2 scale 0.5 的參考基準
            cv2.putText(
                frame, ascii_text, org,
                cv2.FONT_HERSHEY_SIMPLEX,
                cv_scale, color, thickness, cv2.LINE_AA,
            )

    def get_text_size(
        self,
        text: str,
        font_size: int,
    ) -> Tuple[int, int]:
        """
        回傳文字的 (width, height)。
        有 CJK 字型時用 PIL 量測；fallback 時用 cv2.getTextSize。
        """
        if self._has_cjk:
            font = self._get_font(font_size)
            # PIL >= 9.2.0 使用 font.getbbox；舊版用 font.getsize
            try:
                bbox = font.getbbox(text)          # (left, top, right, bottom)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except AttributeError:
                return font.getsize(text)          # (width, height)
        else:
            ascii_text = self._to_ascii_fallback(text)
            cv_scale = font_size / 28.0
            (tw, th), _ = cv2.getTextSize(
                ascii_text, cv2.FONT_HERSHEY_SIMPLEX, cv_scale, 1
            )
            return tw, th

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_font_path() -> Optional[str]:
        """掃描候選清單，回傳第一個存在的字型路徑；全部找不到回傳 None。"""
        for path in CvTextRenderer._FONT_CANDIDATES:
            if os.path.isfile(path):
                return path
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """從快取取得指定大小的字型（若不存在則建立）。"""
        if size not in self._font_cache:
            self._font_cache[size] = ImageFont.truetype(self._font_path, size)
        return self._font_cache[size]

    def _pil_put_text(
        self,
        frame: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_size: int,
        color: Tuple[int, int, int],
    ) -> None:
        """
        使用 Pillow 在 BGR frame 上繪製文字（in-place）。

        流程：BGR → RGB PIL Image → ImageDraw.text → 轉回 BGR ndarray。
        color 為 BGR，轉換為 RGB 後傳給 Pillow。
        org 為左下角座標；PIL text() 接受左上角，需依字型高度調整。
        """
        font = self._get_font(font_size)

        # 量測高度，用於 org（左下角 → 左上角）換算
        try:
            bbox = font.getbbox(text)
            th = bbox[3] - bbox[1]
        except AttributeError:
            _, th = font.getsize(text)

        # BGR → RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        # PIL 使用左上角座標；cv2 org 是左下角 → y 軸上移 th
        x, y = org
        pil_color = (color[2], color[1], color[0])   # BGR → RGB
        draw.text((x, y - th), text, font=font, fill=pil_color)

        # RGB PIL → BGR ndarray（in-place 更新）
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        frame[:] = result

    @staticmethod
    def _to_ascii_fallback(text: str) -> str:
        """
        將文字中的非 ASCII 字元替換為可讀的英文提示。
        僅在無 CJK 字型時使用。
        """
        # 常見提示語對照表
        _REPLACEMENTS = {
            "左鍵點擊兩點定義絆線": "Click 2 pts to set wire",
            "左鍵新增頂點": "L-click: add vertex",
            "右鍵封閉多邊形": "R-click: close ROI",
            "雙擊重置": "Dbl-click: reset",
            "已選": "sel",
            "點": "pts",
        }
        result = text
        for zh, en in _REPLACEMENTS.items():
            result = result.replace(zh, en)
        # 剩餘非 ASCII 字元以 '?' 替代
        return result.encode("ascii", errors="replace").decode("ascii")


# ── 模組層級單例（字型只偵測一次）────────────────────────────────────────────
_text_renderer = CvTextRenderer()


class VideoWidget(QLabel):
    """
    影像顯示元件，繼承自 QLabel。

    三種顯示模式：
    1. Placeholder   : 尚未選擇來源（文字佔位）
    2. Preview       : 靜止幀預覽，疊加 ROI / Wire 編輯線條
    3. Inference     : 推論模式，疊加 BBox / Track ID / ROI / Wire

    編輯模式（edit_mode=True）可在模式 2 / 3 下同時啟用：
    - 左鍵點擊：新增 ROI 頂點 或 定義 Wire 端點
    - 右鍵點擊：完成 ROI 多邊形（需至少 3 點）
    - 雙擊左鍵：清除所有點重新開始

    訊號：
    - roi_updated(list)  : ROI 頂點更新（正規化座標）
    - wire_updated(list) : Wire 端點更新（正規化座標 [x1,y1,x2,y2]）
    """

    roi_updated  = pyqtSignal(list)   # List[List[float]]
    wire_updated = pyqtSignal(list)   # List[float] [x1,y1,x2,y2]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #1a1a2e;")

        # 目前顯示的原始幀（BGR）；None 表示尚無幀
        self._current_frame: Optional[np.ndarray] = None

        # 預覽靜止幀（BGR）；推論開始前從外部設定
        self._preview_frame: Optional[np.ndarray] = None

        # 編輯模式
        self._edit_mode: bool = False
        self._edit_target: str = CountMode.TRIP_WIRE

        # 編輯中的暫存點（像素座標，相對於原始/預覽幀）
        self._editing_points: List[Tuple[int, int]] = []

        # 已確認的正規化座標（持久存放，用於疊加顯示）
        self._roi_points_norm: List[List[float]] = []
        self._wire_norm: List[float] = []

        # 當前計數模式（預覽疊加需要）
        self._count_mode: str = CountMode.TRIP_WIRE

        # 已計數 ID 及其 has_b 狀態（推論模式使用）
        self._counted_ids: set = set()
        self._counted_id_has_b: Dict[int, bool] = {}

        # 幀尺寸快取（像素）
        self._frame_w: int = 1
        self._frame_h: int = 1

        # 顯示控制
        self._show_id: bool = True   # 是否在標籤中顯示 Track ID

    # ── 公開介面：幀更新 ──────────────────────────────────────────────────────

    def update_frame(
        self,
        frame: np.ndarray,
        a_detections: List[DetectionResult],
        b_detections: List[DetectionResult],
        counted_ids: set,
        counted_id_has_b: Dict[int, bool],
        roi_polygon: Optional[np.ndarray],
        wire_p1: Optional[Tuple[int, int]],
        wire_p2: Optional[Tuple[int, int]],
        mode: str,
    ) -> None:
        """
        推論模式：接收新幀與偵測結果，疊加後顯示。
        在主執行緒呼叫（由 Qt Signal 保證）。

        Args:
            frame            : BGR 影像幀
            a_detections     : A 類別偵測結果清單
            b_detections     : B 類別偵測結果清單（中心點不在任何 A 內的不繪製）
            counted_ids      : 已計數的 Track ID 集合
            counted_id_has_b : 已計數 ID → has_b 對應表
            roi_polygon      : ROI 像素座標多邊形（或 None）
            wire_p1, wire_p2 : Trip Wire 端點（或 None）
            mode             : 當前計數模式
        """
        self._frame_w = frame.shape[1]
        self._frame_h = frame.shape[0]
        self._current_frame = frame.copy()
        self._counted_ids = counted_ids
        self._counted_id_has_b = counted_id_has_b
        self._count_mode = mode

        display = frame.copy()
        self._draw_roi(display, roi_polygon)
        self._draw_wire(display, wire_p1, wire_p2, mode)
        self._draw_detections(display, a_detections, b_detections,
                              counted_ids, counted_id_has_b)

        if self._edit_mode:
            self._draw_editing_overlay(display)

        self._set_pixmap_from_bgr(display)

    def update_preview_frame(
        self,
        frame: np.ndarray,
        mode: str,
    ) -> None:
        """
        預覽模式：設定靜止底圖並立刻疊加已確認的 ROI / Wire 顯示。

        Args:
            frame : BGR 靜止幀（影片第一幀 或 RTSP/USB 即時幀）
            mode  : 當前計數模式（決定 Wire 顯示方式）
        """
        self._preview_frame = frame.copy()
        self._frame_w = frame.shape[1]
        self._frame_h = frame.shape[0]
        self._count_mode = mode
        # current_frame 清空，表示目前是預覽模式（非推論）
        self._current_frame = None
        self._repaint_preview()

    def repaint_with_current_settings(self, mode: str) -> None:
        """
        ROI / Wire 設定變更後，主動觸發預覽幀重繪。
        若目前是推論模式（_current_frame 有值），則不做任何事
        （推論幀會在下一幀自動帶入新設定）。
        若無任何底圖，則以靜態畫布顯示最新設定。
        """
        self._count_mode = mode
        if self._current_frame is not None:
            return
        if self._preview_frame is not None:
            self._repaint_preview()
        else:
            self._repaint_blank_canvas()

    # ── 公開介面：模式控制 ────────────────────────────────────────────────────

    def set_edit_mode(self, enabled: bool, target: str = CountMode.TRIP_WIRE) -> None:
        """
        切換編輯模式。

        Args:
            enabled: True 進入編輯模式
            target : 編輯目標（CountMode.TRIP_WIRE 或 CountMode.ROI_ENTER）
        """
        self._edit_mode = enabled
        self._edit_target = target
        self._editing_points.clear()

        if enabled:
            self.setCursor(Qt.CrossCursor)
            if self._preview_frame is None and self._current_frame is None:
                # 無底圖：顯示靜態畫布（而非純文字），讓使用者能即時看到線條
                self._repaint_blank_canvas()
            elif self._preview_frame is not None and self._current_frame is None:
                self._repaint_preview()
        else:
            self.setCursor(Qt.ArrowCursor)
            if self._preview_frame is not None and self._current_frame is None:
                self._repaint_preview()
            elif self._preview_frame is None and self._current_frame is None:
                self.show_placeholder()

    def set_roi_norm(self, points: List[List[float]]) -> None:
        """從外部設定 ROI 正規化點（載入設定 / ROI 確認後使用）。"""
        self._roi_points_norm = points

    def set_wire_norm(self, wire: List[float]) -> None:
        """從外部設定 Wire 正規化點（載入設定 / Wire 確認後使用）。"""
        self._wire_norm = wire

    def clear_preview(self) -> None:
        """清除預覽幀，回到 placeholder 狀態（推論開始時呼叫）。"""
        self._preview_frame = None
        self._current_frame = None

    def set_show_id(self, show: bool) -> None:
        """設定是否在標籤中顯示 Track ID。切換後下一幀自動生效。"""
        self._show_id = show

    def show_placeholder(self) -> None:
        """顯示等待影像來源的佔位畫面。"""
        self.clear()
        self.setText("📷  請選擇影像來源並點擊「預覽」")
        self.setStyleSheet(
            "background-color: #1a1a2e; color: #4a9eff; font-size: 18px;"
        )

    # ── 滑鼠事件（編輯模式） ──────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if not self._edit_mode:
            return

        frame_pt = self._label_to_frame(event.pos())
        if frame_pt is None:
            return

        if event.button() == Qt.LeftButton:
            self._editing_points.append(frame_pt)

            # Trip Wire：兩點即完成
            if self._edit_target == CountMode.TRIP_WIRE:
                if len(self._editing_points) >= 2:
                    p1, p2 = self._editing_points[0], self._editing_points[1]
                    wire = [
                        p1[0] / self._frame_w, p1[1] / self._frame_h,
                        p2[0] / self._frame_w, p2[1] / self._frame_h,
                    ]
                    self._wire_norm = wire
                    self._editing_points.clear()
                    self.wire_updated.emit(wire)
                    # 立刻重繪，讓使用者看到已確認的線段
                    self._repaint_preview_or_editing()
                else:
                    self._repaint_preview_or_editing()
            else:
                self._repaint_preview_or_editing()

        elif event.button() == Qt.RightButton:
            if self._edit_target == CountMode.ROI_ENTER and len(self._editing_points) >= 3:
                norm_pts = [
                    [p[0] / self._frame_w, p[1] / self._frame_h]
                    for p in self._editing_points
                ]
                self._roi_points_norm = norm_pts
                self._editing_points.clear()
                self.roi_updated.emit(norm_pts)
                # 立刻重繪，讓使用者看到已確認的 ROI
                self._repaint_preview_or_editing()

    def mouseDoubleClickEvent(self, event) -> None:
        """雙擊左鍵：清除編輯中的暫存點，重新開始。"""
        if self._edit_mode and event.button() == Qt.LeftButton:
            self._editing_points.clear()
            self._repaint_preview_or_editing()

    # ── 私有：重繪協調 ────────────────────────────────────────────────────────

    def _repaint_preview(self) -> None:
        """
        以 _preview_frame 為底圖，疊加已確認的 ROI / Wire 並重繪。
        若在編輯模式中，也一併疊加暫存點。
        """
        if self._preview_frame is None:
            return

        display = self._preview_frame.copy()

        roi_poly = self._norm_roi_to_pixel()
        self._draw_roi(display, roi_poly)

        w_p1, w_p2 = self._norm_wire_to_pixel()
        self._draw_wire(display, w_p1, w_p2, self._count_mode)

        if self._edit_mode:
            self._draw_editing_overlay(display)

        self._set_pixmap_from_bgr(display)

    def _repaint_blank_canvas(self) -> None:
        """
        無底圖（未預覽、未推論）時，以 widget 當前尺寸建立純色靜態畫布，
        疊加已確認的 ROI / Wire 以及編輯中的暫存點並顯示。

        底圖顏色與 placeholder 一致（#1a1a2e）。
        """
        w = max(self.width(), 1)
        h = max(self.height(), 1)

        # 虛擬幀尺寸同步到 widget 尺寸，確保正規化座標換算正確
        self._frame_w = w
        self._frame_h = h

        # 深藍黑色底圖（BGR: 46, 26, 18 → #1a1a2e）
        canvas = np.full((h, w, 3), (46, 26, 18), dtype=np.uint8)

        roi_poly = self._norm_roi_to_pixel()
        self._draw_roi(canvas, roi_poly)

        wp1, wp2 = self._norm_wire_to_pixel()
        self._draw_wire(canvas, wp1, wp2, self._count_mode)

        if self._edit_mode:
            self._draw_editing_overlay(canvas)

        self._set_pixmap_from_bgr(canvas)

    def _repaint_preview_or_editing(self) -> None:
        """
        在編輯模式下觸發重繪。

        優先使用預覽幀作為底圖；若無預覽幀，則產生純色靜態畫布，
        確保使用者在任何情況下都能即時看到正在定義的線條 / ROI。
        """
        if self._preview_frame is not None and self._current_frame is None:
            self._repaint_preview()
        elif self._current_frame is None:
            # 無任何底圖時，建立純色靜態畫布顯示編輯中的線條
            self._repaint_blank_canvas()

    # ── 私有：座標換算 ────────────────────────────────────────────────────────

    def _norm_roi_to_pixel(self) -> Optional[np.ndarray]:
        """將正規化 ROI 點換算為像素座標 numpy array。"""
        if len(self._roi_points_norm) < 3:
            return None
        pts = [
            [int(p[0] * self._frame_w), int(p[1] * self._frame_h)]
            for p in self._roi_points_norm
        ]
        return np.array(pts, dtype=np.int32)

    def _norm_wire_to_pixel(
        self,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """將正規化 Wire 點換算為像素座標 tuple pair。"""
        if len(self._wire_norm) != 4:
            return None, None
        x1 = int(self._wire_norm[0] * self._frame_w)
        y1 = int(self._wire_norm[1] * self._frame_h)
        x2 = int(self._wire_norm[2] * self._frame_w)
        y2 = int(self._wire_norm[3] * self._frame_h)
        return (x1, y1), (x2, y2)

    # ── 私有繪製方法 ──────────────────────────────────────────────────────────

    def _draw_roi(
        self,
        frame: np.ndarray,
        roi_polygon: Optional[np.ndarray],
    ) -> None:
        """繪製 ROI 多邊形（半透明填充 + 邊框 + 頂點）。"""
        if roi_polygon is None or len(roi_polygon) < 3:
            return

        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], ROI_COLOR)
        cv2.addWeighted(overlay, ROI_ALPHA, frame, 1 - ROI_ALPHA, 0, frame)
        cv2.polylines(frame, [roi_polygon], True, ROI_COLOR, ROI_LINE_THICKNESS)

        for pt in roi_polygon:
            cv2.circle(frame, tuple(pt), ROI_POINT_RADIUS, ROI_COLOR, -1)

    def _draw_wire(
        self,
        frame: np.ndarray,
        p1: Optional[Tuple[int, int]],
        p2: Optional[Tuple[int, int]],
        mode: str,
    ) -> None:
        """繪製 Trip Wire 線段與方向箭頭。"""
        if p1 is None or p2 is None:
            return

        if mode != CountMode.TRIP_WIRE:
            ep1, ep2 = self._extend_line_to_frame(p1, p2, frame.shape[1], frame.shape[0])
            if ep1 is not None and ep2 is not None:
                cv2.line(frame, ep1, ep2, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (140, 140, 140), 2, cv2.LINE_AA)
            return

        ep1, ep2 = self._extend_line_to_frame(p1, p2, frame.shape[1], frame.shape[0])
        if ep1 is not None and ep2 is not None:
            cv2.line(frame, ep1, ep2, WIRE_COLOR, 1, cv2.LINE_AA)

        cv2.line(frame, p1, p2, WIRE_COLOR, WIRE_THICKNESS, cv2.LINE_AA)

        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        cv2.arrowedLine(
            frame,
            (mx - 20, my),
            (mx + 20, my),
            WIRE_COLOR, WIRE_THICKNESS, cv2.LINE_AA,
            tipLength=0.5,
        )

        cv2.circle(frame, p1, 5, WIRE_COLOR, -1)
        cv2.circle(frame, p2, 5, WIRE_COLOR, -1)

    @staticmethod
    def _extend_line_to_frame(
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Return the intersections of the infinite wire line with frame bounds."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None, None

        candidates: List[Tuple[float, Tuple[int, int]]] = []

        if dx != 0:
            for x in (0, frame_w - 1):
                t = (x - x1) / dx
                y = y1 + t * dy
                if 0 <= y <= frame_h - 1:
                    candidates.append((t, (int(round(x)), int(round(y)))))

        if dy != 0:
            for y in (0, frame_h - 1):
                t = (y - y1) / dy
                x = x1 + t * dx
                if 0 <= x <= frame_w - 1:
                    candidates.append((t, (int(round(x)), int(round(y)))))

        unique = []
        seen = set()
        for t, pt in sorted(candidates, key=lambda item: item[0]):
            if pt not in seen:
                seen.add(pt)
                unique.append((t, pt))

        if len(unique) < 2:
            return None, None

        return unique[0][1], unique[-1][1]

    def _draw_detections(
        self,
        frame: np.ndarray,
        a_detections: List[DetectionResult],
        b_detections: List[DetectionResult],
        counted_ids: set,
        counted_id_has_b: Dict[int, bool],
    ) -> None:
        """
        繪製 A 物件框與 B 物件框。

        A 框顏色規則：
          - 未計數              → BBOX_COLOR（黃色）
          - 已計數 + 含 B       → BBOX_COUNTED_WITH_B（綠色）
          - 已計數 + 無 B       → BBOX_COUNTED_NO_B（暗黃色）

        B 框規則：
          - 僅當 B 的中心點落在任一 A 的 BBox 內才繪製
          - 顏色固定為 BBOX_B_COLOR（青色）
          - 不顯示 ID
        """
        # 先繪製 A 框
        for det in a_detections:
            x1, y1, x2, y2 = det.bbox
            is_counted = det.track_id in counted_ids

            if not is_counted:
                color = BBOX_COLOR
            elif counted_id_has_b.get(det.track_id, False):
                color = BBOX_COUNTED_WITH_B
            else:
                color = BBOX_COUNTED_NO_B

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

            if self._show_id:
                label = f"A  ID:{det.track_id}  {det.conf:.2f}"
            else:
                label = f"A  {det.conf:.2f}"
            self._draw_label(frame, label, x1, y1, color)

            cv2.circle(frame, det.center, 3, color, -1)

        # 收集所有 A 的 BBox，供 B 的過濾判斷
        a_bboxes = [det.bbox for det in a_detections]

        # 繪製 B 框（僅中心點在某個 A 的 BBox 內才繪製）
        for b in b_detections:
            bcx, bcy = b.cx, b.cy
            inside_any_a = any(
                ax1 <= bcx <= ax2 and ay1 <= bcy <= ay2
                for ax1, ay1, ax2, ay2 in a_bboxes
            )
            if not inside_any_a:
                continue

            bx1, by1, bx2, by2 = b.bbox
            cv2.rectangle(frame, (bx1, by1), (bx2, by2),
                          BBOX_B_COLOR, BBOX_B_THICKNESS)

            label = f"B  {b.conf:.2f}"
            self._draw_label(frame, label, bx1, by1, BBOX_B_COLOR)

            cv2.circle(frame, b.center, 3, BBOX_B_COLOR, -1)

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        x1: int,
        y1: int,
        color: Tuple[int, int, int],
    ) -> None:
        """在 BBox 左上角繪製背景填充的文字標籤。（僅含 ASCII，保留原實作）"""
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS
        )
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - th - baseline - 2),
            (x1 + tw + 4, label_y + 2),
            color, -1,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE, (0, 0, 0),
            TEXT_THICKNESS, cv2.LINE_AA,
        )

    def _draw_editing_overlay(self, frame: np.ndarray) -> None:
        """
        在編輯模式下，疊加正在定義中的暫存點與線條（尚未確認的部分）。

        提示文字使用 CvTextRenderer，自動支援中文；
        若無 CJK 字型則自動顯示英文 fallback。
        """
        pts = self._editing_points
        color = WIRE_COLOR if self._edit_target == CountMode.TRIP_WIRE else ROI_COLOR

        for pt in pts:
            cv2.circle(frame, pt, ROI_POINT_RADIUS, color, -1)

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], color, ROI_LINE_THICKNESS)

        if self._edit_target == CountMode.ROI_ENTER and len(pts) >= 3:
            cv2.line(frame, pts[-1], pts[0], color, 1, cv2.LINE_AA)

        # ── 提示文字（使用 CvTextRenderer 支援中文）─────────────────────────
        if _text_renderer.has_cjk:
            if self._edit_target == CountMode.TRIP_WIRE:
                hint = f"左鍵點擊兩點定義絆線（已選 {len(pts)}/2 點）"
            else:
                hint = f"左鍵新增頂點（已選 {len(pts)} 點），右鍵封閉多邊形，雙擊重置"
        else:
            # 無 CJK 字型時顯示純英文
            if self._edit_target == CountMode.TRIP_WIRE:
                hint = f"Click 2 pts to set wire (selected {len(pts)}/2)"
            else:
                hint = f"L-click: add vertex ({len(pts)} pts)  R-click: close  Dbl: reset"

        _HINT_FONT_SIZE = 18   # px，對應畫面底部提示

        # 繪製半透明背景條，提升可讀性
        tw, th = _text_renderer.get_text_size(hint, _HINT_FONT_SIZE)
        bar_y = frame.shape[0] - _HINT_FONT_SIZE - 10
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, bar_y - 4),
            (tw + 16, frame.shape[0]),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # 文字繪製（org 為左下角，與 cv2.putText 相同語義）
        _text_renderer.put_text(
            frame, hint,
            org=(8, frame.shape[0] - 8),
            font_size=_HINT_FONT_SIZE,
            color=(255, 255, 255),
        )

    def _show_edit_placeholder(self) -> None:
        """完全無底圖時進入編輯模式，顯示文字提示。"""
        hint = (
            "絆線模式：點擊兩點定義線段"
            if self._edit_target == CountMode.TRIP_WIRE
            else "ROI 模式：左鍵新增頂點，右鍵封閉，雙擊重置"
        )
        self.setText(f"✏️  {hint}")
        self.setStyleSheet(
            "background-color: #1a1a2e; color: #4aff8c; font-size: 15px;"
        )

    # ── 私有：座標轉換 ────────────────────────────────────────────────────────

    def _label_to_frame(self, label_pos: QPoint) -> Optional[Tuple[int, int]]:
        """
        將 QLabel 內的滑鼠座標轉換為原始幀像素座標。

        有底圖（_preview_frame 或 _current_frame）時：按照縮放比例換算。
        無底圖時：以 widget 尺寸作為虛擬座標系，正規化結果仍正確。
        """
        lw = self.width()
        lh = self.height()

        has_frame = (self._preview_frame is not None) or (self._current_frame is not None)

        if not has_frame:
            fw = lw if lw > 0 else 1
            fh = lh if lh > 0 else 1
            self._frame_w = fw
            self._frame_h = fh
            px = max(0, min(lw - 1, label_pos.x()))
            py = max(0, min(lh - 1, label_pos.y()))
            return (px, py)

        fw = self._frame_w
        fh = self._frame_h

        scale = min(lw / fw, lh / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        offset_x = (lw - disp_w) // 2
        offset_y = (lh - disp_h) // 2

        px = label_pos.x() - offset_x
        py = label_pos.y() - offset_y

        if px < 0 or py < 0 or px >= disp_w or py >= disp_h:
            return None

        fx = int(px / scale)
        fy = int(py / scale)
        fx = max(0, min(fw - 1, fx))
        fy = max(0, min(fh - 1, fy))
        return (fx, fy)

    # ── 私有：影像顯示 ────────────────────────────────────────────────────────

    def _set_pixmap_from_bgr(self, frame: np.ndarray) -> None:
        """將 BGR frame 轉換為 QPixmap 並顯示，同時清除文字 placeholder 樣式。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(bytes(rgb.data), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setStyleSheet("")
