"""YOLOv8 行人检测 (仅 person 类) on Hailo-8.

Hailo Model Zoo yolov8n HEF 输出 (NMS 已内嵌):
    out[name][batch][class_id] -> ndarray (N, 5)
        每行 = [ymin, xmin, ymax, xmax, score], 全部归一化到 [0, 1],
        相对 640x640 letterbox 输入.

预处理:
    letterbox (保持纵横比, 黑边 pad) + BGR -> RGB.

后处理:
    仅保留 class 0 (person), 反向 letterbox 还原到原图坐标 (x1, y1, x2, y2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import cv2
import numpy as np

from .hailo_runner import HailoMultiRunner

logger = logging.getLogger(__name__)


DEFAULT_SCORE_THRESHOLD = 0.40


@dataclass
class Detection:
    """单个检测结果."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 原图坐标
    score: float
    cls: int = 0  # COCO person


class YoloV8PersonDetector:
    """YOLOv8 行人检测器, 包装 Hailo-8 上的 yolov8n HEF.

    必须配合 HailoMultiRunner 共享 VDevice, 因为属性识别也需要同一个 device.
    """

    PERSON_CLS = 0
    DEFAULT_INPUT_SIZE: Tuple[int, int] = (640, 640)
    DEFAULT_SCORE_THRESH = 0.40
    DEFAULT_SCORE_THRESHOLD = 0.40
    PAD_COLOR: Tuple[int, int, int] = (0, 0, 0)

    MODEL_NAME = "yolov8n"

    def __init__(
        self,
        multi_runner: HailoMultiRunner,
        hef_path: str,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        score_thresh: float = DEFAULT_SCORE_THRESH,
    ) -> None:
        self.multi = multi_runner
        self.hef_path = hef_path
        self.input_w, self.input_h = input_size
        self.score_thresh = float(score_thresh)
        self._slot: Any = None
        self._input_layer: str = ""
        self._output_layer: str = ""

    def setup(self) -> None:
        """调用方在 lifespan 启动时调用, 把 HEF 装入 multi runner."""
        self._slot = self.multi.add(self.MODEL_NAME, self.hef_path)
        self._input_layer = self._slot.input_infos[0].name
        self._output_layer = self._slot.output_infos[0].name
        logger.info(
            "YoloV8PersonDetector ready: in=%s out=%s score_thresh=%.2f",
            self._input_layer, self._output_layer, self.score_thresh,
        )

    @property
    def input_layer(self) -> str:
        return self._input_layer

    @property
    def output_layer(self) -> str:
        return self._output_layer

    @staticmethod
    def _letterbox(
        frame: np.ndarray, new_w: int, new_h: int
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = frame.shape[:2]
        scale = min(new_w / w, new_h / h)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        pad_l = (new_w - rw) // 2
        pad_t = (new_h - rh) // 2
        canvas[pad_t : pad_t + rh, pad_l : pad_l + rw] = resized
        return canvas, scale, (pad_l, pad_t)

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        canvas, scale, (pad_l, pad_t) = self._letterbox(frame_bgr, self.input_w, self.input_h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb, scale, pad_l, pad_t

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        rgb, scale, pad_l, pad_t = self._preprocess(frame_bgr)
        x = rgb[np.newaxis, ...]
        out, _dt = self.multi.run_with_timing(self.MODEL_NAME, {self._input_layer: x})
        return self._parse_output(out, h, w, scale, pad_l, pad_t)

    def detect_with_timing(
        self, frame_bgr: np.ndarray
    ) -> Tuple[List[Detection], float]:
        h, w = frame_bgr.shape[:2]
        rgb, scale, pad_l, pad_t = self._preprocess(frame_bgr)
        x = rgb[np.newaxis, ...]
        out, dt = self.multi.run_with_timing(self.MODEL_NAME, {self._input_layer: x})
        return self._parse_output(out, h, w, scale, pad_l, pad_t), dt

    def _parse_output(
        self,
        out: Mapping[str, Any],
        h: int,
        w: int,
        scale: float,
        pad_l: int,
        pad_t: int,
    ) -> List[Detection]:
        """解析 NMS 输出, 提取 person bbox."""
        raw = out[self._output_layer]
        # batch dim
        if isinstance(raw, list):
            per_class = raw[0] if (len(raw) > 0 and isinstance(raw[0], list)) else raw
        elif isinstance(raw, np.ndarray):
            per_class = raw[0] if raw.ndim >= 3 else raw
        else:
            return []
        if not per_class or len(per_class) <= self.PERSON_CLS:
            return []
        person = per_class[self.PERSON_CLS]
        person = np.asarray(person, dtype=np.float32)
        if person.ndim != 2 or person.shape[1] != 5:
            return []
        scores = person[:, 4]
        mask = scores >= self.score_thresh
        if not np.any(mask):
            return []
        boxes = person[mask, :4]  # (N, 4): [ymin, xmin, ymax, xmax] normalized
        scores_valid = scores[mask]

        # 640x640 像素坐标
        y1_px = boxes[:, 0] * self.input_h - pad_t
        x1_px = boxes[:, 1] * self.input_w - pad_l
        y2_px = boxes[:, 2] * self.input_h - pad_t
        x2_px = boxes[:, 3] * self.input_w - pad_l

        inv = 1.0 / scale if scale > 0 else 1.0
        x1 = np.clip(x1_px * inv, 0.0, w - 1.0)
        y1 = np.clip(y1_px * inv, 0.0, h - 1.0)
        x2 = np.clip(x2_px * inv, 0.0, w - 1.0)
        y2 = np.clip(y2_px * inv, 0.0, h - 1.0)

        dets: List[Detection] = []
        for i in range(len(scores_valid)):
            x1i, y1i, x2i, y2i = float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])
            if x2i - x1i < 4 or y2i - y1i < 4:
                continue
            dets.append(
                Detection(
                    bbox=(x1i, y1i, x2i, y2i),
                    score=float(scores_valid[i]),
                    cls=self.PERSON_CLS,
                )
            )
        return dets

    def set_score_thresh(self, score_thresh: float) -> None:
        self.score_thresh = float(score_thresh)
