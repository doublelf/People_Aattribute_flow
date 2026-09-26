"""离线预渲染任务.

单 worker 线程, 全程处理一个视频:
  1. FrameSource 读帧
  2. YOLO.detect + N×Attr.infer_crop
  3. draw_person / draw_panel 标注
  4. 写入 ffmpeg (libx264 ultrafast) 子进程 stdin
  5. 每帧 StatsAggregator.update → 追加到 JSON 缓冲
  6. 结束后 flush ffmpeg, 写 stats.json

输出文件:
  - workspace/outputs/<name>_annotated.mp4   (H.264 mp4)
  - workspace/outputs/<name>_stats.json     (per-frame 元数据)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.draw import draw_error, draw_fps_watermark, draw_panel, draw_person
from app.stats import StatsAggregator, StatsSnapshot
from app.stream import (
    FrameSource,
    close_writer,
    open_cv2_writer,
    open_ffmpeg_writer,
    write_frame,
)

logger = logging.getLogger(__name__)


class PrerenderJob:
    """单个预渲染任务.

    状态机: idle -> running -> (done | cancelled | error)
    通过 self.progress / self.frames_done / self.fps_proc 供 /api/video/progress 查询.
    """

    def __init__(
        self,
        task_id: str,
        video_path: str,
        output_name: str,            # 不含扩展名的输出名, 如 "test" 或 "<uuid>"
        output_dir: str,
        yolo,                         # YoloV8PersonDetector
        attr,                         # PersonAttrClassifier
        stats: StatsAggregator,
        yolo_thresh: float = 0.5,
        attr_thresh: float = 0.5,
        pad_ratio: float = 0.05,
        jpeg_quality: int = 80,
        loop_video: bool = False,
    ) -> None:
        self.task_id = task_id
        self.video_path = video_path
        self.output_name = output_name
        self.output_dir = output_dir
        self.yolo = yolo
        self.attr = attr
        self.stats = stats
        self.yolo_thresh = float(yolo_thresh)
        self.attr_thresh = float(attr_thresh)
        self.pad_ratio = float(pad_ratio)
        self.jpeg_quality = int(jpeg_quality)
        self.loop_video = bool(loop_video)

        self.mp4_path = os.path.join(output_dir, f"{output_name}_annotated.mp4")
        self.json_path = os.path.join(output_dir, f"{output_name}_stats.json")

        # 状态
        self.status: str = "idle"
        self.progress: float = 0.0
        self.frames_done: int = 0
        self.total_frames: int = 0
        self.fps_proc: float = 0.0
        self.eta_s: float = 0.0
        self.error_msg: str = ""

        self._cancel_event = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "task_id": self.task_id,
                "status": self.status,
                "progress": round(self.progress, 4),
                "frames_done": self.frames_done,
                "total_frames": self.total_frames,
                "fps_proc": round(self.fps_proc, 2),
                "eta_s": round(self.eta_s, 1),
                "video_path": self.video_path,
                "output_name": self.output_name,
                "mp4_ready": os.path.exists(self.mp4_path),
                "json_ready": os.path.exists(self.json_path),
                "error": self.error_msg,
            }

    def _safe_crop(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            return None
        pad_x, pad_y = bw * self.pad_ratio, bh * self.pad_ratio
        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(w, int(round(x2 + pad_x)))
        cy2 = min(h, int(round(y2 + pad_y)))
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()

    def run(self) -> None:
        """Worker 入口. 由 threading.Thread 启动."""
        with self._lock:
            self.status = "running"
            self.error_msg = ""
        os.makedirs(self.output_dir, exist_ok=True)
        # 删除旧输出 (重新渲染)
        for p in (self.mp4_path, self.json_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

        source = FrameSource(self.video_path, loop=self.loop_video)
        try:
            source.open()
        except RuntimeError as e:
            with self._lock:
                self.status = "error"
                self.error_msg = str(e)
            return

        meta = source.metadata()
        width, height = meta["width"], meta["height"]
        fps = meta["fps"]
        self.total_frames = meta["total_frames"]

        # 启动编码器 (优先 ffmpeg, fallback cv2 mp4v)
        writer = open_ffmpeg_writer(self.mp4_path, width, height, fps)
        if writer is None:
            logger.warning("PrerenderJob %s: ffmpeg 不可用, fallback cv2 mp4v", self.task_id)
            writer = open_cv2_writer(self.mp4_path, width, height, fps)
        if writer is None:
            with self._lock:
                self.status = "error"
                self.error_msg = "Failed to open video writer (no ffmpeg, cv2 mp4v failed)"
            source.close()
            return
        writer_kind = writer[1]
        logger.info(
            "PrerenderJob %s: writer=%s, output=%s (%dx%d @ %.1f fps, %d frames)",
            self.task_id, writer_kind, self.mp4_path, width, height, fps, self.total_frames,
        )

        # 应用阈值
        self.yolo.set_score_thresh(self.yolo_thresh)
        self.attr.set_threshold(self.attr_thresh)

        # 重置 stats (累计 + tracker)
        self.stats.reset_cumulative()

        # 帧数据缓冲 (每帧一条, 末尾一次性写 json)
        frames_meta: List[Dict[str, Any]] = []
        video_meta = {
            "video": os.path.basename(self.video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": self.total_frames,
            "duration_s": meta["duration_s"],
            "yolo_score_thresh": self.yolo_thresh,
            "attr_thresh": self.attr_thresh,
            "created_at": time.time(),
            "frames": frames_meta,
        }

        t_start = time.perf_counter()
        last_fps_update = t_start
        last_frames_for_fps = 0

        try:
            for idx in range(self.total_frames):
                if self._cancel_event.is_set():
                    with self._lock:
                        self.status = "cancelled"
                    break

                frame = source.read()
                if frame is None:
                    break

                persons: List[dict] = []
                yolo_ms = 0.0
                attr_ms = 0.0
                err: Optional[str] = None
                try:
                    dets, yolo_ms = self.yolo.detect_with_timing(frame)
                    t0 = time.perf_counter()
                    for det in dets:
                        crop = self._safe_crop(frame, det.bbox)
                        if crop is None:
                            continue
                        try:
                            attrs, _dt = self.attr.infer_crop_with_timing(crop)
                        except Exception as e:
                            logger.warning("attr infer error: %s", e)
                            attrs = []
                        persons.append({
                            "bbox": [int(det.bbox[0]), int(det.bbox[1]),
                                     int(det.bbox[2]), int(det.bbox[3])],
                            "score": float(det.score),
                            "attrs": [
                                {"class": a["class"], "confidence": float(a["confidence"]),
                                 "group": a.get("group", "")}
                                for a in attrs if a.get("group") in ("gender", "age")
                            ],
                        })
                    attr_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    err = str(e)
                    logger.exception("pipeline step error: %s", e)

                snap = self.stats.update(persons, yolo_ms=yolo_ms, attr_ms=attr_ms)

                # 绘制标注
                if err:
                    draw_error(frame, err)
                for p in persons:
                    draw_person(frame, p["bbox"], p["score"], p["attrs"])
                draw_panel(frame, snap)
                draw_fps_watermark(frame, snap.fps)

                # 写入 mp4
                try:
                    write_frame(writer, frame)
                except (BrokenPipeError, OSError) as e:
                    logger.error("writer write failed: %s", e)
                    with self._lock:
                        self.status = "error"
                        self.error_msg = f"writer closed: {e}"
                    break

                # 记录 per-frame 元数据
                frames_meta.append({
                    "i": idx,
                    "t": round(idx / fps, 4),
                    "count": snap.current_count,
                    "gender": dict(snap.gender),
                    "age": dict(snap.age),
                    "cumulative": {
                        "unique": snap.cumulative_unique,
                        "total": snap.cumulative_total,
                        "gender": dict(snap.cumulative_gender),
                        "age": dict(snap.cumulative_age),
                        "gender_pct": dict(snap.gender_pct),
                        "age_pct": dict(snap.age_pct),
                    },
                })

                # 更新进度
                now = time.perf_counter()
                with self._lock:
                    self.frames_done = idx + 1
                    self.progress = (idx + 1) / max(1, self.total_frames)
                    if now - last_fps_update >= 1.0:
                        elapsed = now - last_fps_update
                        self.fps_proc = (self.frames_done - last_frames_for_fps) / elapsed
                        last_fps_update = now
                        last_frames_for_fps = self.frames_done
                        if self.fps_proc > 0:
                            remain = self.total_frames - self.frames_done
                            self.eta_s = remain / self.fps_proc
        finally:
            close_writer(writer)
            source.close()
            elapsed = time.perf_counter() - t_start
            logger.info(
                "PrerenderJob %s: done frames=%d/%d in %.1fs (%.1f fps proc)",
                self.task_id, self.frames_done, self.total_frames, elapsed,
                self.frames_done / max(0.001, elapsed),
            )

        # 如果未取消且未出错, 写 json
        with self._lock:
            if self.status == "running":
                self.status = "done"
        if self.status in ("done",):
            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(video_meta, f, ensure_ascii=False)
            except OSError as e:
                logger.error("write stats.json failed: %s", e)
                with self._lock:
                    self.status = "error"
                    self.error_msg = f"json write failed: {e}"