"""Person Attribute ResNet v1.18 (PETA 35 类) on Hailo-8.

HEF 输入: 224x224 RGB uint8.
HEF 输出: 35 维向量, 可能是 [0,1] 概率或原始 logits.

PETA 标签 (35 类) — 全部按官方 PETA 顺序:
    0: Age16-30          12: Jacket        24: Shoes
    1: Age31-45          13: Jeans         25: Shorts
    2: Age46-60          14: Leather shoes 26: Short sleeve
    3: AgeAbove61        15: Logo          27: Skirt
    4: Backpack          16: Long hair     28: Sneaker
    5: CarryingOther     17: Male          29: Stripes
    6: Casual lower      18: Messenger bag 30: Sunglasses
    7: Casual upper      19: Muffler       31: Trousers
    8: Formal lower      20: No accesory   32: T-shirt
    9: Formal upper      21: No carrying   33: UpperOther
    10: Hat              22: Plaid         34: V-Neck
    11: Hat              23: Plastic bag

dashboard 白名单 (其它类别在 UI 上隐藏):
    age:     "Age16-30", "Age31-45", "Age46-60", "AgeAbove61"
    gender:  "Male"
    extras:  "Hat", "Sunglasses", "Long hair", "Muffler", "Logo", "Plastic bag"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .hailo_runner import HailoMultiRunner

logger = logging.getLogger(__name__)


# 35 类 PETA 标签 (按 HEF 输出顺序)
PETA_LABELS: Tuple[str, ...] = (
    "Age16-30",
    "Age31-45",
    "Age46-60",
    "AgeAbove61",
    "Backpack",
    "CarryingOther",
    "Casual lower",
    "Casual upper",
    "Formal lower",
    "Formal upper",
    "Hat",
    "Jacket",
    "Jeans",
    "Leather shoes",
    "Logo",
    "Long hair",
    "Male",
    "Messenger bag",
    "Muffler",
    "No accesory",
    "No carrying",
    "Plaid",
    "Plastic bag",
    "Sandals",
    "Shoes",
    "Shorts",
    "Short sleeve",
    "Skirt",
    "Sneaker",
    "Stripes",
    "Sunglasses",
    "Trousers",
    "T-shirt",
    "UpperOther",
    "V-Neck",
)

# dashboard 用白名单 (仅 gender + age, 配饰/外观不再输出)
DASHBOARD_LABELS = {
    "Age16-30",
    "Age31-45",
    "Age46-60",
    "AgeAbove61",
    "Male",
}

DEFAULT_THRESHOLD = 0.50
DEFAULT_INPUT_SIZE: Tuple[int, int] = (224, 224)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def _select_output(outputs: Mapping[str, Any]) -> Optional[Tuple[np.ndarray, str]]:
    if not outputs:
        return None
    for name, value in outputs.items():
        arr = np.asarray(value)
        if arr.size >= len(PETA_LABELS):
            return arr, name
    name = next(iter(outputs))
    return np.asarray(outputs[name]), name


def parse_attributes(
    raw_outputs: Mapping[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """把 HEF 输出解析为属性列表 (按置信度降序).

    返回每项:
        {id, class, raw_class, confidence, is_dashboard}
    """
    selected = _select_output(raw_outputs)
    if selected is None:
        return []
    raw, output_name = selected
    raw_shape = raw.shape
    scores = raw.astype(np.float32).reshape(-1)
    if scores.size < len(PETA_LABELS):
        raise ValueError(
            f"Person attribute output too small: shape={raw_shape}, size={scores.size}"
        )
    scores = scores[: len(PETA_LABELS)]

    finite = scores[np.isfinite(scores)]
    if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
        scores = _sigmoid(scores)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

    attributes: List[Dict[str, Any]] = []
    for idx, (label, score) in enumerate(zip(PETA_LABELS, scores)):
        confidence = float(score)
        is_dashboard = label in DASHBOARD_LABELS

        if label == "Male":
            if confidence >= threshold:
                attributes.append({
                    "id": idx,
                    "class": "Male",
                    "raw_class": label,
                    "confidence": confidence,
                    "is_dashboard": True,
                    "group": "gender",
                })
            else:
                attributes.append({
                    "id": idx,
                    "class": "Female",
                    "raw_class": label,
                    "confidence": float(1.0 - confidence),
                    "is_dashboard": True,
                    "group": "gender",
                })
            continue

        if is_dashboard and confidence >= threshold:
            if label.startswith("Age"):
                group = "age"
            else:
                group = "extra"
            attributes.append({
                "id": idx,
                "class": label,
                "raw_class": label,
                "confidence": confidence,
                "is_dashboard": True,
                "group": group,
            })

    attributes.sort(key=lambda item: item["confidence"], reverse=True)

    if not hasattr(parse_attributes, "_logged_shape"):
        stats = (
            f"min={float(scores.min()):.4f}, max={float(scores.max()):.4f}, "
            f"mean={float(scores.mean()):.4f}"
        )
        logger.info(
            "PersonAttr: output=%s, shape=%s, attrs=%d, %s",
            output_name, raw_shape, len(attributes), stats,
        )
        parse_attributes._logged_shape = True  # type: ignore[attr-defined]

    return attributes


class PersonAttrClassifier:
    """Person Attribute ResNet v1.18 (PETA 35 类) 包装."""

    MODEL_NAME = "person_attr_resnet"

    def __init__(
        self,
        multi_runner: HailoMultiRunner,
        hef_path: str,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.multi = multi_runner
        self.hef_path = hef_path
        self.input_w, self.input_h = input_size
        self.threshold = float(threshold)
        self._slot: Any = None
        self._input_layer: str = ""

    def setup(self) -> None:
        self._slot = self.multi.add(self.MODEL_NAME, self.hef_path)
        self._input_layer = self._slot.input_infos[0].name
        info = self._slot.input_infos[0]
        shape = tuple(info.shape)
        if len(shape) >= 3:
            self.input_h, self.input_w = int(shape[0]), int(shape[1])
        logger.info(
            "PersonAttrClassifier ready: in=%s size=%dx%d threshold=%.2f",
            self._input_layer, self.input_w, self.input_h, self.threshold,
        )

    @property
    def input_layer(self) -> str:
        return self._input_layer

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.shape[1] != self.input_w or crop_bgr.shape[0] != self.input_h:
            crop_bgr = cv2.resize(crop_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def infer_crop(self, crop_bgr: np.ndarray) -> List[Dict[str, Any]]:
        x = self.preprocess(crop_bgr)[np.newaxis, ...]
        out, _dt = self.multi.run_with_timing(self.MODEL_NAME, {self._input_layer: x})
        return parse_attributes(out, self.threshold)

    def infer_crop_with_timing(
        self, crop_bgr: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], float]:
        x = self.preprocess(crop_bgr)[np.newaxis, ...]
        out, dt = self.multi.run_with_timing(self.MODEL_NAME, {self._input_layer: x})
        return parse_attributes(out, self.threshold), dt

    def infer_image(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """整图直接推理 (不裁剪). 用于 REST 单图属性接口."""
        return self.infer_crop(image_bgr)
