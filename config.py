"""
config.py — 全域常數與設定資料結構

所有可調整的常數集中於此，避免魔法數字散落程式各處。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os

# ── 路徑 ────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH: str = os.path.join(BASE_DIR, "settings.json")
DEFAULT_CSV_DIR: str = "logs"
DEFAULT_MODEL_PATH: str = "yolov8n.pt"  # 預設模型，可於 GUI 更換

# ── 推論參數 ─────────────────────────────────────────────────────────────────
TARGET_CLASS_NAME: str   = "A"          # 主要目標類別（A）
TARGET_CLASS_B_NAME: str = "B"          # 次要目標類別（B），不追蹤 ID
INFERENCE_CONF: float    = 0.4          # 信心度門檻
INFERENCE_IOU: float     = 0.5          # NMS IOU 門檻
TRACKER_CONFIG: str      = "bytetrack.yaml"  # ultralytics 內建 tracker

# ── 影像顯示 ─────────────────────────────────────────────────────────────────
DISPLAY_WIDTH: int  = 960   # VideoWidget 預設寬度（像素）
DISPLAY_HEIGHT: int = 540   # VideoWidget 預設高度（像素）
VIDEO_FPS_TARGET: int = 30  # 影像執行緒目標 FPS

# ── ROI 繪製樣式 ──────────────────────────────────────────────────────────────
ROI_COLOR: Tuple[int, int, int] = (0, 255, 128)       # BGR
ROI_ALPHA: float = 0.25                                # 填充透明度
ROI_LINE_THICKNESS: int = 2
ROI_POINT_RADIUS: int = 6

# ── Trip Wire 樣式 ────────────────────────────────────────────────────────────
WIRE_COLOR: Tuple[int, int, int] = (0, 128, 255)      # BGR
WIRE_THICKNESS: int = 2
WIRE_ARROW_SIZE: int = 20

# ── BBox 標注樣式（A 物件）────────────────────────────────────────────────────
BBOX_COLOR: Tuple[int, int, int]          = (0, 200, 255)   # 未計數 A：黃色
BBOX_COUNTED_WITH_B: Tuple[int, int, int] = (0, 255, 80)    # 已計數 A 含 B：綠色
BBOX_COUNTED_NO_B: Tuple[int, int, int]   = (220, 100, 0)   # 已計數 A 無 B：藍色（BGR）
BBOX_THICKNESS: int = 2
TEXT_SCALE: float   = 0.55
TEXT_THICKNESS: int = 1

# ── BBox 標注樣式（B 物件）────────────────────────────────────────────────────
BBOX_B_COLOR: Tuple[int, int, int] = (255, 200, 0)    # 青色（BGR）
BBOX_B_THICKNESS: int = 2

# ── 計數模式 ─────────────────────────────────────────────────────────────────
class CountMode:
    TRIP_WIRE = "trip_wire"   # 模式 A：穿越絆線
    ROI_ENTER = "roi_enter"   # 模式 B：進入 ROI


# ── 影像來源類型 ──────────────────────────────────────────────────────────────
class SourceType:
    FILE = "file"
    RTSP = "rtsp"
    USB  = "usb"


# ── 設定資料結構 ──────────────────────────────────────────────────────────────
@dataclass
class AppSettings:
    """
    完整應用程式設定，可序列化為 JSON。
    所有座標均以 0~1 的正規化比例儲存，與解析度無關。
    """

    # 影像來源
    source_type: str = SourceType.FILE
    source_path: str = ""           # 影片路徑 或 RTSP URL
    usb_index: int = 0              # USB 攝影機編號

    # 模型
    model_path: str = DEFAULT_MODEL_PATH

    # 計數模式
    count_mode: str = CountMode.TRIP_WIRE

    # ROI 多邊形（正規化座標串列，每個元素 [x, y]）
    roi_points: List[List[float]] = field(default_factory=list)

    # Trip Wire（正規化座標，[x1, y1, x2, y2]）
    wire_points: List[float] = field(default_factory=list)

    # CSV 輸出目錄
    csv_dir: str = DEFAULT_CSV_DIR

    # 輪播模式（僅影片檔有效）：播完後自動重播，計數持續累計
    loop_video: bool = False

    # 語音提示：含 B 播放 B.wav，無 B 播放 NO_B.wav
    voice_prompt_enabled: bool = False

    # 物件框信心值門檻
    confidence_threshold: float = INFERENCE_CONF

    # Dashboard 子統計自訂顯示名稱
    count_with_b_name: str = ""
    count_no_b_name: str = ""

    # 上次關閉/執行後保留的計數
    last_count: int = 0
    last_count_with_b: int = 0
    last_count_no_b: int = 0
