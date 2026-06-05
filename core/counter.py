"""
core/counter.py — 計數邏輯（Trip Wire + ROI Enter 兩種模式）

核心計數引擎，負責：
1. 模式 A（Trip Wire）：判斷物件中心是否由左向右穿越線段
2. 模式 B（ROI Enter）：判斷物件中心是否進入多邊形區域
3. 以 Track ID 集合避免重複計數
4. 維護計數歷史清單供 CSV 輸出與 GUI 顯示
5. 判斷 B 物件中心點是否落在 A 的 BBox 內，分別統計「含 B」與「無 B」
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import cv2

from config import CountMode
from core.inference import DetectionResult


@dataclass
class CountEvent:
    """
    單筆計數事件記錄。

    Attributes:
        track_id     : A 物件的 ByteTrack ID
        timestamp    : Unix 時間戳（秒）
        mode         : CountMode 常數
        count_at     : 發生當下的累計計數（A 總數）
        has_b        : 計數當下 A 的 BBox 內是否含有 B 物件
        count_with_b : 發生當下「含 B 的 A」累計數
        count_no_b   : 發生當下「無 B 的 A」累計數
    """
    track_id: int
    timestamp: float
    mode: str
    count_at: int
    has_b: bool
    count_with_b: int
    count_no_b: int


class CounterEngine:
    """
    計數引擎，無狀態依賴（除了追蹤 ID 集合）。

    使用方式：
        engine = CounterEngine()
        engine.set_roi(normalized_points, frame_w, frame_h)
        engine.set_wire(normalized_wire, frame_w, frame_h)
        engine.set_mode(CountMode.TRIP_WIRE)

        a_dets, b_dets = inference_engine.infer(frame)
        for det in a_dets:
            engine.update(det, b_dets)

        print(engine.count)
        print(engine.count_with_b)
        print(engine.count_no_b)
    """

    def __init__(self) -> None:
        self._count: int = 0            # A 物件總計數
        self._count_with_b: int = 0     # 含 B 的 A 計數
        self._count_no_b: int = 0       # 無 B 的 A 計數

        self._counted_ids: Set[int] = set()             # 已計數的 Track ID
        self._counted_id_has_b: Dict[int, bool] = {}    # 已計數 ID → 是否含 B
        self._history: List[CountEvent] = []

        # 模式 A：Trip Wire（像素座標）
        self._wire_p1: Optional[Tuple[int, int]] = None
        self._wire_p2: Optional[Tuple[int, int]] = None

        # 模式 B：ROI 多邊形（像素座標，numpy array）
        self._roi_polygon: Optional[np.ndarray] = None

        # 每個 track_id 的上一幀中心點（用於判斷穿越方向）
        self._prev_centers: Dict[int, Tuple[int, int]] = {}

        self._mode: str = CountMode.TRIP_WIRE

    # ── 設定介面 ──────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """切換計數模式。"""
        if mode not in (CountMode.TRIP_WIRE, CountMode.ROI_ENTER):
            raise ValueError(f"未知計數模式：{mode}")
        self._mode = mode

    def set_roi(
        self,
        normalized_points: List[List[float]],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        設定 ROI 多邊形。

        Args:
            normalized_points: 正規化座標 [[x1,y1],[x2,y2],...] (0~1)
            frame_w, frame_h : 影像尺寸（像素）
        """
        if len(normalized_points) < 3:
            self._roi_polygon = None
            return
        pts = [
            [int(p[0] * frame_w), int(p[1] * frame_h)]
            for p in normalized_points
        ]
        self._roi_polygon = np.array(pts, dtype=np.int32)

    def set_wire(
        self,
        normalized_wire: List[float],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        設定 Trip Wire 線段。

        Args:
            normalized_wire: [x1, y1, x2, y2] 正規化座標 (0~1)
            frame_w, frame_h: 影像尺寸（像素）
        """
        if len(normalized_wire) != 4:
            self._wire_p1 = None
            self._wire_p2 = None
            return
        x1 = int(normalized_wire[0] * frame_w)
        y1 = int(normalized_wire[1] * frame_h)
        x2 = int(normalized_wire[2] * frame_w)
        y2 = int(normalized_wire[3] * frame_h)
        self._wire_p1 = (x1, y1)
        self._wire_p2 = (x2, y2)

    def reset_count(self) -> None:
        """重置計數（不清除 ID 紀錄，避免已過物件重複計數）。"""
        self._count = 0
        self._count_with_b = 0
        self._count_no_b = 0
        self._history.clear()

    def set_counts(self, count: int, count_with_b: int, count_no_b: int) -> None:
        """還原累計計數數字，用於重開程式後接續上次結果。"""
        self._count = max(0, int(count))
        self._count_with_b = max(0, int(count_with_b))
        self._count_no_b = max(0, int(count_no_b))
        if self._count != self._count_with_b + self._count_no_b:
            self._count = self._count_with_b + self._count_no_b

    def full_reset(self) -> None:
        """完整重置，包含 ID 集合（換影片來源時使用）。"""
        self._count = 0
        self._count_with_b = 0
        self._count_no_b = 0
        self._counted_ids.clear()
        self._counted_id_has_b.clear()
        self._prev_centers.clear()
        self._history.clear()

    # ── 計數更新 ──────────────────────────────────────────────────────────────

    def update(
        self,
        detection: DetectionResult,
        b_detections: List[DetectionResult],
    ) -> bool:
        """
        處理單一 A 偵測結果，若觸發計數則回傳 True。

        B 物件清單用於判斷「是否有 B 的中心點落在此 A 的 BBox 內」。
        每個 A 最多只計入一個 B（找到第一個即停止）。

        Args:
            detection   : A 類別的 DetectionResult
            b_detections: 當前幀所有 B 類別的 DetectionResult 清單

        Returns:
            True 表示本次更新觸發了計數事件
        """
        tid = detection.track_id

        # 已計數過的 ID 直接跳過（仍更新中心點供 Trip Wire 方向判斷）
        if tid in self._counted_ids:
            self._prev_centers[tid] = detection.center
            return False

        triggered = False

        if self._mode == CountMode.TRIP_WIRE:
            triggered = self._check_trip_wire(detection)
        elif self._mode == CountMode.ROI_ENTER:
            triggered = self._check_roi_enter(detection)

        if triggered:
            has_b = self._check_b_inside_a(detection, b_detections)

            self._count += 1
            if has_b:
                self._count_with_b += 1
            else:
                self._count_no_b += 1

            self._counted_ids.add(tid)
            self._counted_id_has_b[tid] = has_b

            event = CountEvent(
                track_id=tid,
                timestamp=time.time(),
                mode=self._mode,
                count_at=self._count,
                has_b=has_b,
                count_with_b=self._count_with_b,
                count_no_b=self._count_no_b,
            )
            self._history.append(event)

        # 更新上一幀中心點
        self._prev_centers[tid] = detection.center
        return triggered

    # ── 私有計數邏輯 ──────────────────────────────────────────────────────────

    def _check_b_inside_a(
        self,
        a_det: DetectionResult,
        b_detections: List[DetectionResult],
    ) -> bool:
        """
        判斷是否有任一 B 物件的中心點落在 A 的 BBox 內。

        條件：B.cx 在 [A.x1, A.x2]，B.cy 在 [A.y1, A.y2]。
        一個 A 最多對應一個 B，找到第一個即回傳 True。

        Args:
            a_det       : A 類別偵測結果
            b_detections: 當前幀所有 B 類別偵測結果

        Returns:
            True 表示 A 的 BBox 內包含至少一個 B 的中心點
        """
        ax1, ay1, ax2, ay2 = a_det.bbox
        for b in b_detections:
            bcx, bcy = b.cx, b.cy
            if ax1 <= bcx <= ax2 and ay1 <= bcy <= ay2:
                return True
        return False

    def _check_trip_wire(self, detection: DetectionResult) -> bool:
        """
        判斷物件是否由左向右穿越絆線。

        原理：
        - 取上一幀中心點與當前中心點，形成移動向量
        - 計算兩端相對於線段的側邊（cross product 符號）
        - 若符號不同 → 穿越了線段
        - 再確認移動方向為向右（dx > 0）
        """
        if self._wire_p1 is None or self._wire_p2 is None:
            return False

        tid = detection.track_id
        curr = detection.center

        if tid not in self._prev_centers:
            return False

        prev = self._prev_centers[tid]

        # 線段向量
        lx = self._wire_p2[0] - self._wire_p1[0]
        ly = self._wire_p2[1] - self._wire_p1[1]

        # 上一幀相對於線段起點的向量 → cross product
        prev_cross = (
            lx * (prev[1] - self._wire_p1[1]) -
            ly * (prev[0] - self._wire_p1[0])
        )
        curr_cross = (
            lx * (curr[1] - self._wire_p1[1]) -
            ly * (curr[0] - self._wire_p1[0])
        )

        # 符號不同 → 穿越線段
        crossed = (prev_cross > 0) != (curr_cross > 0)
        if not crossed:
            return False

        # 向右移動（dx > 0）才計數
        dx = curr[0] - prev[0]
        return dx > 0

    def _check_roi_enter(self, detection: DetectionResult) -> bool:
        """
        判斷物件中心點是否進入 ROI 多邊形。

        使用 OpenCV pointPolygonTest：
        - 回傳值 >= 0 → 點在多邊形內或邊線上
        """
        if self._roi_polygon is None:
            return False

        curr = detection.center
        inside = cv2.pointPolygonTest(
            self._roi_polygon,
            (float(curr[0]), float(curr[1])),
            measureDist=False,
        )
        return inside >= 0

    # ── 屬性 ─────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """A 物件累計總計數。"""
        return self._count

    @property
    def count_with_b(self) -> int:
        """含 B 的 A 累計計數。"""
        return self._count_with_b

    @property
    def count_no_b(self) -> int:
        """無 B 的 A 累計計數。"""
        return self._count_no_b

    @property
    def counted_ids(self) -> Set[int]:
        """已計數的 Track ID 集合（唯讀）。"""
        return frozenset(self._counted_ids)

    @property
    def counted_id_has_b(self) -> Dict[int, bool]:
        """已計數 ID 對應的 has_b 狀態（唯讀副本）。"""
        return dict(self._counted_id_has_b)

    @property
    def history(self) -> List[CountEvent]:
        """計數歷史清單（唯讀副本）。"""
        return list(self._history)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def roi_polygon(self) -> Optional[np.ndarray]:
        return self._roi_polygon

    @property
    def wire_points(self) -> Tuple[
        Optional[Tuple[int, int]], Optional[Tuple[int, int]]
    ]:
        return (self._wire_p1, self._wire_p2)
