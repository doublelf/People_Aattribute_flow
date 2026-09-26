"""OpenCV drawing helpers: 框/属性标签/统计面板.

字号与线宽通过顶部常量集中控制:
  - DRAW_FONT_SCALE = 1.6      文本 (fontScale, line_h, pad, box_w)
  - DRAW_BOX_THICK_SCALE = 1.4 检测框线宽
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .stats import StatsSnapshot


# 颜色 (BGR)
COLOR_BOX = (80, 220, 80)
DRAW_BOX_THICK_SCALE = 1.4
COLOR_BOX_THICK = max(1, round(2 * DRAW_BOX_THICK_SCALE))  # 2 -> 3
COLOR_LABEL_BG = (15, 20, 24)
COLOR_LABEL_FG = (245, 245, 245)
COLOR_PANEL_BG = (15, 20, 24)
COLOR_PANEL_FG = (220, 230, 240)
COLOR_PANEL_TITLE = (80, 255, 80)
COLOR_PANEL_HIGHLIGHT = (0, 255, 255)
COLOR_ERROR = (0, 0, 255)

# 文本字号统一缩放系数
DRAW_FONT_SCALE = 1.6

# Base values (fontScale=0.5, thickness=1, line_h=18, pad=4/8/10 等)
BASE_LINE_H = 18
BASE_PAD_PERSON = 4
BASE_PAD_PANEL_X = 10
BASE_PAD_PANEL_Y = 8
BASE_BOX_W_MIN = 120
BASE_BOTTOM_MARGIN = 10

# 派生常量
_LINE_H = int(BASE_LINE_H * DRAW_FONT_SCALE)              # 28
_PAD_PERSON = int(BASE_PAD_PERSON * DRAW_FONT_SCALE)      # 6
_PAD_PANEL_X = int(BASE_PAD_PANEL_X * DRAW_FONT_SCALE)    # 16
_PAD_PANEL_Y = int(BASE_PAD_PANEL_Y * DRAW_FONT_SCALE)    # 12
_BOX_W_MIN = int(BASE_BOX_W_MIN * DRAW_FONT_SCALE)        # 192
_BOTTOM_MARGIN = int(BASE_BOTTOM_MARGIN * DRAW_FONT_SCALE)  # 16

_FONT_SCALE_PERSON = 0.5 * DRAW_FONT_SCALE       # 0.8
_FONT_SCALE_PANEL = 0.5 * DRAW_FONT_SCALE        # 0.8
_FONT_SCALE_FPS = 0.6 * DRAW_FONT_SCALE          # 0.96
_FONT_SCALE_ERROR = 0.7 * DRAW_FONT_SCALE       # 1.12
_THICKNESS_THIN = max(1, round(1 * DRAW_FONT_SCALE))  # 2
_THICKNESS_BOLD = max(2, round(2 * DRAW_FONT_SCALE))  # 3


def _color_for_class(cls: str) -> Tuple[int, int, int]:
    """给不同 group 上色."""
    if cls in ("Male", "Female"):
        return (180, 220, 255)
    if cls.startswith("Age"):
        return (255, 200, 120)
    return (200, 255, 200)


def _text_width(text: str, font_face: int, font_scale: float, thickness: int) -> int:
    """精确计算 cv2.putText 文本宽度 (像素)."""
    (w, _), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    return int(w)


def _wrap_text(
    text: str,
    max_w: int,
    font_face: int,
    font_scale: float,
    thickness: int,
) -> List[str]:
    """按空格切分保证每行宽度 <= max_w (像素).

    整行能放下则不切. 仅用空格切, 不强制断字符 (避免英文单词支离破碎).
    """
    if _text_width(text, font_face, font_scale, thickness) <= max_w:
        return [text]
    words = text.split(" ")
    if len(words) <= 1:
        return [text]
    lines: List[str] = []
    cur = ""
    for word in words:
        cand = (cur + " " + word).strip() if cur else word
        if _text_width(cand, font_face, font_scale, thickness) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_person(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    score: float,
    attrs: List[Dict[str, Any]],
) -> None:
    """画一个人的检测框 + 头顶属性标签."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, COLOR_BOX_THICK)

    # 头顶属性标签 (gender + age, 取 confidence 最高)
    gender_attr = next((a for a in attrs if a.get("group") == "gender"), None)
    age_attr = next((a for a in attrs if a.get("group") == "age"), None)

    label_lines: List[str] = []
    if gender_attr:
        label_lines.append(f"{gender_attr['class']} {gender_attr['confidence']:.2f}")
    if age_attr:
        label_lines.append(f"{age_attr['class']} {age_attr['confidence']:.2f}")
    label_lines.append(f"person {score:.2f}")

    if not label_lines:
        return

    # 多行标签背景 (用 getTextSize 精确测量宽度)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_widths = [
        _text_width(s, font, _FONT_SCALE_PERSON, _THICKNESS_THIN) for s in label_lines
    ]
    pad = _PAD_PERSON
    box_w = max(_BOX_W_MIN, max(text_widths) + pad * 2)
    box_h = _LINE_H * len(label_lines) + pad * 2
    ty = max(0, y1 - box_h - 2)
    cv2.rectangle(
        frame, (x1, ty), (min(w, x1 + box_w), ty + box_h), COLOR_LABEL_BG, -1
    )
    for i, line in enumerate(label_lines):
        color = (
            _color_for_class(label_lines[0].split(" ")[0])
            if i == 0
            else COLOR_LABEL_FG
        )
        cv2.putText(
            frame,
            line,
            (x1 + pad, ty + pad + (i + 1) * _LINE_H - int(4 * DRAW_FONT_SCALE)),
            font,
            _FONT_SCALE_PERSON,
            color,
            _THICKNESS_THIN,
            cv2.LINE_AA,
        )


def draw_panel(
    frame: np.ndarray,
    stats: StatsSnapshot,
    pos: Tuple[int, int] = (16, 16),
) -> None:
    """左上角实时统计面板."""
    raw_lines: List[Tuple[str, Tuple[int, int, int]]] = []
    raw_lines.append(("perview - Realtime People Counter", COLOR_PANEL_TITLE))
    raw_lines.append((f"Current: {stats.current_count}", COLOR_PANEL_HIGHLIGHT))
    raw_lines.append((f"Cumulative: {stats.cumulative_unique}", COLOR_PANEL_FG))
    raw_lines.append((f"FPS:     {stats.fps:.1f}", COLOR_PANEL_FG))
    raw_lines.append((
        f"Infer:   {stats.inference_ms:.0f} ms (YOLO {stats.yolo_ms:.0f} + Attr {stats.attr_ms:.0f})",
        COLOR_PANEL_FG,
    ))

    male = stats.gender.get("Male", 0)
    female = stats.gender.get("Female", 0)
    raw_lines.append(("---", COLOR_PANEL_FG))
    raw_lines.append((f"[Realtime]  M:{male}  F:{female}", COLOR_PANEL_FG))

    age_labels = ("Age16-30", "Age31-45", "Age46-60", "AgeAbove61")
    age_short = ("<30", "31-45", "46-60", "60+")
    age_str = "  ".join(
        f"{short}:{stats.age.get(lbl, 0)}"
        for lbl, short in zip(age_labels, age_short)
    )
    raw_lines.append((f"[Realtime]  Age {age_str}", COLOR_PANEL_FG))

    if stats.cumulative_total > 0:
        cg = stats.cumulative_gender
        cp = stats.gender_pct
        m_pct = cp.get("Male", 0.0)
        f_pct = cp.get("Female", 0.0)
        raw_lines.append(("---", COLOR_PANEL_FG))
        raw_lines.append((
            f"[Cumulative] M:{cg.get('Male',0)} ({m_pct:.1f}%)  "
            f"F:{cg.get('Female',0)} ({f_pct:.1f}%)",
            COLOR_PANEL_HIGHLIGHT,
        ))
        ca = stats.cumulative_age
        ap = stats.age_pct
        age_pct_str = "  ".join(
            f"{lbl.replace('Age','')}:{ca.get(lbl,0)}({ap.get(lbl,0.0):.1f}%)"
            for lbl in age_labels
        )
        raw_lines.append((f"[Cumulative] Age {age_pct_str}", COLOR_PANEL_HIGHLIGHT))

    # 长行自动换行 (> 帧宽一半)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    max_line_w_px = w // 2
    wrapped: List[Tuple[str, Tuple[int, int, int]]] = []
    for text, color in raw_lines:
        sub_lines = _wrap_text(text, max_line_w_px, font, _FONT_SCALE_PANEL, _THICKNESS_THIN)
        for sub in sub_lines:
            wrapped.append((sub, color))

    # 计算面板大小
    text_widths = [
        _text_width(t, font, _FONT_SCALE_PANEL, _THICKNESS_THIN)
        for t, _ in wrapped
    ]
    pad_x = _PAD_PANEL_X
    pad_y = _PAD_PANEL_Y
    max_text_w = max(text_widths) if text_widths else 0
    box_w = max_text_w + pad_x * 2
    box_h = _LINE_H * len(wrapped) + pad_y * 2

    x, y = pos
    x = max(0, min(w - box_w - 1, x))
    y = max(0, min(h - box_h - 1, y))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    for i, (text, color) in enumerate(wrapped):
        cv2.putText(
            frame,
            text,
            (x + pad_x, y + pad_y + (i + 1) * _LINE_H - int(4 * DRAW_FONT_SCALE)),
            font,
            _FONT_SCALE_PANEL,
            color,
            _THICKNESS_THIN,
            cv2.LINE_AA,
        )


def draw_fps_watermark(
    frame: np.ndarray,
    fps: float,
    pos: Tuple[int, int] = (16, -1),
) -> None:
    """左下角 FPS 水印."""
    text = f"Hailo FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, _FONT_SCALE_FPS, _THICKNESS_BOLD)
    h, w = frame.shape[:2]
    if pos[1] < 0:
        y = h - _BOTTOM_MARGIN
    else:
        y = pos[1]
    x = max(0, min(w - tw - _BOTTOM_MARGIN, pos[0]))
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        _FONT_SCALE_FPS,
        (0, 255, 0),
        _THICKNESS_BOLD,
        cv2.LINE_AA,
    )


def draw_error(frame: np.ndarray, msg: str) -> None:
    cv2.putText(
        frame,
        f"Inference error: {msg}",
        (int(20 * DRAW_FONT_SCALE), int(40 * DRAW_FONT_SCALE)),
        cv2.FONT_HERSHEY_SIMPLEX,
        _FONT_SCALE_ERROR,
        COLOR_ERROR,
        _THICKNESS_BOLD,
    )