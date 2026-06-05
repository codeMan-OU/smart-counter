"""
core/csv_logger.py — CSV 計數事件記錄器

每次計數事件觸發時寫入一筆 CSV 記錄。
CSV 檔名包含啟動時間戳，避免覆蓋舊記錄。

欄位說明：
    count        : A 物件累計總計數
    track_id     : A 物件的 ByteTrack ID
    datetime     : 可讀時間（YYYY-MM-DD HH:MM:SS.mmm）
    mode         : 計數模式（trip_wire / roi_enter）
    has_b        : 計數當下 A 的 BBox 內是否含 B（True/False）
    count_with_b : 含 B 的 A 累計數
    count_no_b   : 無 B 的 A 累計數
"""

from __future__ import annotations
import csv
import os
import time
from datetime import datetime
from typing import Optional
from config import BASE_DIR
from core.counter import CountEvent


# CSV 欄位定義
CSV_FIELDS = [
    "count",
    "track_id",
    "datetime",
    "mode",
    "has_b",
    "count_with_b",
    "count_no_b",
]


class CsvLogger:
    """
    執行緒安全的 CSV 記錄器（寫入操作在呼叫端執行緒，由 Qt Signal 保證序列化）。

    檔案在首次 log() 呼叫時才建立，避免空白 CSV。
    """

    def __init__(self, output_dir: str) -> None:
        """
        Args:
            output_dir: CSV 輸出目錄（不存在時自動建立）
        """
        self._output_dir = self._resolve_output_dir(output_dir)
        self._filepath: Optional[str] = None
        self._file = None
        self._writer = None
        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_output_dir(output_dir: str) -> str:
        if not output_dir:
            return os.path.join(BASE_DIR, "logs")
        if os.path.isabs(output_dir):
            return output_dir
        return os.path.join(BASE_DIR, output_dir)

    def _ensure_opened(self) -> None:
        """確保 CSV 檔案已開啟；若尚未開啟則建立。"""
        if self._file is not None:
            return

        os.makedirs(self._output_dir, exist_ok=True)
        filename = f"count_{self._session_start}.csv"
        self._filepath = os.path.join(self._output_dir, filename)

        self._file = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    # ── 公開介面 ──────────────────────────────────────────────────────────────

    def log(self, event: CountEvent) -> None:
        """
        寫入單筆計數事件。

        Args:
            event: 來自 CounterEngine.history 的 CountEvent
        """
        self._ensure_opened()

        dt = datetime.fromtimestamp(event.timestamp)
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

        self._writer.writerow({
            "count":        event.count_at,
            "track_id":     event.track_id,
            "datetime":     dt_str,
            "mode":         event.mode,
            "has_b":        event.has_b,
            "count_with_b": event.count_with_b,
            "count_no_b":   event.count_no_b,
        })
        self._file.flush()  # 確保即時寫入磁碟

    def close(self) -> None:
        """關閉 CSV 檔案。"""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def new_session(self) -> None:
        """開始新的記錄 session（換影片來源時呼叫）。"""
        self.close()
        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def filepath(self) -> Optional[str]:
        """當前 CSV 檔案路徑，尚未建立時為 None。"""
        return self._filepath

    def __del__(self) -> None:
        self.close()
