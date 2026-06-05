"""
core/inference.py — YOLOv8 + ByteTrack 推論封裝

封裝 ultralytics YOLOv8 推論與 ByteTrack 追蹤，
對外輸出標準化的 DetectionResult 物件清單。

設計要點（修正閃爍 / ID 不穩 / ID 跳號）：
  - 每幀只呼叫一次 self._model.track()，同時偵測 A 和 B 兩個類別。
  - A 和 B 都參與同一次 ByteTrack pipeline，結果按 class_id 分流。
  - A：有 track_id（ByteTrack 持續追蹤）。
  - B：track_id 固定覆寫為 -1（B 不需要 ID，但仍需讓 tracker 看到
        B 的 box 以維持正確的 NMS 與 Kalman 狀態，避免干擾 A）。
  - 絕對不對同一幀呼叫兩次推論（消除閃爍與 ID 跳號根源）。

ID Remapper 說明：
  ByteTrack 的 ID counter 是全域的，A 和 B 共用同一個序列，
  導致 A 的 ID 看起來不連續（1, 3, 5…）。
  InferenceEngine 內部維護一個 _id_remap 字典，
  將 ByteTrack 分配給 A 的原始 ID 映射為連續整數（1, 2, 3…）。
  此映射在 InferenceEngine 生命週期內持續存在；
  呼叫 reset_id_map() 可配合影片換源使用（full_reset 場景）。
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from ultralytics import YOLO
from config import (
    BASE_DIR,
    TARGET_CLASS_NAME, TARGET_CLASS_B_NAME,
    INFERENCE_CONF, INFERENCE_IOU, TRACKER_CONFIG,
)


@dataclass
class DetectionResult:
    """
    單一偵測結果，包含位置、追蹤 ID、信心度、類別名稱。

    Attributes:
        track_id   : 對 A 類別為重映射後的連續 ID；B 類別固定為 -1
        bbox       : (x1, y1, x2, y2) 像素座標
        conf       : 信心度 0.0 ~ 1.0
        class_name : 偵測類別名稱（TARGET_CLASS_NAME 或 TARGET_CLASS_B_NAME）
    """
    track_id: int
    bbox: Tuple[int, int, int, int]
    conf: float
    class_name: str = "A"

    @property
    def cx(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def cy(self) -> int:
        return (self.bbox[1] + self.bbox[3]) // 2

    @property
    def center(self) -> Tuple[int, int]:
        return (self.cx, self.cy)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """底部中心點，適合用於接地物件的計數判斷。"""
        return (self.cx, self.bbox[3])


class InferenceEngine:
    """
    YOLOv8 推論引擎，整合 ByteTrack 追蹤器。

    核心原則：每幀僅呼叫一次 model.track()，同時處理 A、B 兩個類別，
    再依 class_id 分流輸出，根除雙次推論造成的閃爍與 ID 不穩問題。

    ID 連續性：
    透過 _id_remap 將 ByteTrack 原始 ID 映射為 A 專屬連續整數，
    消除因 B 類別佔用 ID 序列造成的跳號現象。
    """

    def __init__(self, model_path: str, conf_threshold: float = INFERENCE_CONF) -> None:
        """
        Args:
            model_path: YOLO 模型權重路徑（.pt 檔案）
        """
        resolved_model_path = (
            model_path if os.path.isabs(model_path)
            else os.path.join(BASE_DIR, model_path)
        )
        self._model = YOLO(resolved_model_path)
        self._model_path = model_path
        self._conf_threshold = self._clamp_confidence(conf_threshold)

        # 解析 A、B 的 class_id
        self._class_a_id: Optional[int] = self._resolve_class_id(TARGET_CLASS_NAME)
        self._class_b_id: Optional[int] = self._resolve_class_id(TARGET_CLASS_B_NAME)

        # 組合要偵測的 class_id 集合（過濾無關類別，加速推論）
        self._target_class_ids: Optional[List[int]] = self._build_class_filter()

        # 快速查詢用 set
        self._a_ids: Set[int] = (
            {self._class_a_id} if self._class_a_id is not None else set()
        )
        self._b_ids: Set[int] = (
            {self._class_b_id} if self._class_b_id is not None else set()
        )

        # ── A 專屬 ID Remapper ────────────────────────────────────────────
        # key  = ByteTrack 原始 tracker_id（可能跳號）
        # value = 重映射後的連續整數（從 1 開始遞增）
        self._id_remap: Dict[int, int] = {}
        self._next_mapped_id: int = 1

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_confidence(conf_threshold: float) -> float:
        return max(0.05, min(0.95, float(conf_threshold)))

    def _resolve_class_id(self, class_name: str) -> Optional[int]:
        """在模型類別名稱中尋找指定 class_name（大小寫不敏感）。"""
        names: dict = self._model.names
        target_lower = class_name.lower()
        for cls_id, name in names.items():
            if name.lower() == target_lower:
                return int(cls_id)
        print(
            f"[警告] 模型中找不到類別「{class_name}」，"
            f"已知類別：{list(names.values())}"
        )
        return None

    def _build_class_filter(self) -> Optional[List[int]]:
        """
        組合 A、B 的 class_id 清單，供 model.track() 的 classes 參數使用。
        若兩者都找不到，回傳 None（偵測全部類別）。
        """
        ids = []
        if self._class_a_id is not None:
            ids.append(self._class_a_id)
        if self._class_b_id is not None:
            ids.append(self._class_b_id)
        return ids if ids else None

    def _remap_id(self, raw_id: int) -> int:
        """
        將 ByteTrack 原始 tracker_id 映射為 A 專屬連續整數。

        同一個 raw_id 永遠對應同一個 mapped_id（跨幀穩定）。
        新出現的 raw_id 依序分配下一個可用整數。

        Args:
            raw_id: ByteTrack 分配的原始整數 ID

        Returns:
            連續整數 ID（從 1 開始）
        """
        if raw_id not in self._id_remap:
            self._id_remap[raw_id] = self._next_mapped_id
            self._next_mapped_id += 1
        return self._id_remap[raw_id]

    # ── 公開介面 ──────────────────────────────────────────────────────────────

    def reset_id_map(self) -> None:
        """
        清除 ID 映射表，重新從 1 開始分配。

        應於換影片來源（full_reset 場景）時呼叫，
        確保新影片的 A 物件 ID 從 1 重新計起。
        """
        self._id_remap.clear()
        self._next_mapped_id = 1

    def set_confidence_threshold(self, conf_threshold: float) -> None:
        """更新偵測信心值門檻。"""
        self._conf_threshold = self._clamp_confidence(conf_threshold)

    def set_next_mapped_id(self, next_id: int) -> None:
        """
        設定下一個顯示用連續 ID。

        用於程式重開後延續上一輪計數，例如上次累計 35，
        新出現的 A 物件可從 ID 36 開始。
        """
        self._id_remap.clear()
        self._next_mapped_id = max(1, int(next_id))

    def infer(
        self,
        frame: np.ndarray,
        ignore_a_right_of_x: Optional[int] = None,
    ) -> Tuple[
        List[DetectionResult], List[DetectionResult]
    ]:
        """
        對單一幀執行「一次」推論，分流回傳 A、B 兩類偵測結果。

        只呼叫一次 model.track()，同時偵測 A 與 B：
          - A：從結果中過濾 class_a_id，ByteTrack track_id 經 remapper 轉為連續 ID。
          - B：從結果中過濾 class_b_id，track_id 強制設為 -1（不需要追蹤）。

        Args:
            frame: BGR 格式的 OpenCV 影像幀
            ignore_a_right_of_x: 若有設定 Trip Wire，中心點位於此 x 座標右側的
                A 偵測不輸出、不建立 remap ID，避免右側新出現物件造成 ID 跳號。

        Returns:
            (a_detections, b_detections)
        """
        results = self._model.track(
            frame,
            persist=True,           # 跨幀保持 ByteTrack 狀態（必須）
            conf=self._conf_threshold,
            iou=INFERENCE_IOU,
            tracker=TRACKER_CONFIG,
            classes=self._target_class_ids,  # 只偵測 A + B，過濾其他類別
            verbose=False,
        )

        a_detections: List[DetectionResult] = []
        b_detections: List[DetectionResult] = []

        if not results or len(results) == 0:
            return a_detections, b_detections

        result = results[0]
        if result.boxes is None:
            return a_detections, b_detections

        boxes = result.boxes

        # boxes.id 在 tracker 尚未指派 ID 時（第一幀）可能為 None
        has_ids = boxes.id is not None
        pending_a_detections = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].item())

            if cls_id in self._a_ids:
                # A 類別：必須有 track_id 才有追蹤意義
                if not has_ids:
                    continue
                raw_track_id = int(boxes.id[i].item())
                cx = (x1 + x2) // 2
                if ignore_a_right_of_x is not None:
                    if cx > ignore_a_right_of_x and raw_track_id not in self._id_remap:
                        continue
                pending_a_detections.append((
                    raw_track_id,
                    cx,
                    (x1, y1, x2, y2),
                    conf,
                ))

            elif cls_id in self._b_ids:
                # B 類別：不需要 track_id，固定 -1
                b_detections.append(DetectionResult(
                    track_id=-1,
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                    class_name=TARGET_CLASS_B_NAME,
                ))

        # New A IDs are assigned from right to left by x position. Existing
        # remapped IDs keep their original number, so tracking remains stable.
        pending_a_detections.sort(
            key=lambda item: (
                item[0] in self._id_remap,
                -item[1],
            )
        )
        for raw_track_id, _cx, bbox, conf in pending_a_detections:
            mapped_id = self._remap_id(raw_track_id)
            a_detections.append(DetectionResult(
                track_id=mapped_id,
                bbox=bbox,
                conf=conf,
                class_name=TARGET_CLASS_NAME,
            ))

        return a_detections, b_detections

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def class_names(self) -> dict:
        return self._model.names
