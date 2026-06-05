"""
core/video_source.py — 統一影像來源介面

封裝本地影片檔、RTSP 串流、USB 攝影機三種輸入，
對外提供一致的 read() / release() 介面。
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Tuple, Optional
from config import BASE_DIR, SourceType


def _resolve_project_path(path: str) -> str:
    """Resolve relative media paths from the project root."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


class VideoSource:
    """
    統一影像來源封裝。

    使用方式：
        src = VideoSource.from_settings(settings)
        ok, frame = src.read()
        src.release()
    """

    def __init__(self, cap: cv2.VideoCapture, source_type: str) -> None:
        self._cap = cap
        self._source_type = source_type
        self._width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ── 工廠方法 ──────────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str) -> "VideoSource":
        """從本地影片檔建立來源。"""
        resolved_path = _resolve_project_path(path)
        cap = cv2.VideoCapture(resolved_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"無法開啟影片檔：{resolved_path}")
        return cls(cap, SourceType.FILE)

    @classmethod
    def from_rtsp(cls, url: str) -> "VideoSource":
        """從 RTSP 串流建立來源。"""
        # CAP_FFMPEG 確保使用 FFmpeg 後端以支援 RTSP
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ConnectionError(f"無法連線 RTSP 串流：{url}")
        return cls(cap, SourceType.RTSP)

    @classmethod
    def from_usb(cls, index: int = 0) -> "VideoSource":
        """從 USB 攝影機建立來源。"""
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise IOError(f"無法開啟 USB 攝影機（編號 {index}）")
        # 提高 USB 攝影機緩衝區設定，減少延遲
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cls(cap, SourceType.USB)

    @classmethod
    def from_settings(cls, settings) -> "VideoSource":
        """根據 AppSettings 自動選擇來源類型。"""
        if settings.source_type == SourceType.FILE:
            return cls.from_file(settings.source_path)
        elif settings.source_type == SourceType.RTSP:
            return cls.from_rtsp(settings.source_path)
        elif settings.source_type == SourceType.USB:
            return cls.from_usb(settings.usb_index)
        else:
            raise ValueError(f"未知的來源類型：{settings.source_type}")

    # ── 公開介面 ──────────────────────────────────────────────────────────────

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        讀取下一幀。
        回傳 (True, frame) 或 (False, None)。
        影片檔播放結束時回傳 (False, None)。
        """
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        """釋放影像來源資源。"""
        if self._cap.isOpened():
            self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    # ── 屬性 ─────────────────────────────────────────────────────────────────

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def frame_count(self) -> int:
        """影片總幀數（USB/RTSP 回傳 -1）。"""
        if self._source_type == SourceType.FILE:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1
