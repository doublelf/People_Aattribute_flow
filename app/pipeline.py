"""Pipeline orchestrator (v1.2: 离线预渲染 + 回放).

线程模型:
  - 主线程: FastAPI 请求处理, 包括 start_prerender 触发
  - 预渲染 worker (1 个): PrerenderJob.run, 持有 YOLO/属性 inference
  - 输出产物: workspace/outputs/<name>_annotated.mp4 + <name>_stats.json
  - Web 端: <video> + JS 同步 currentTime -> JSON 帧数据

缓存策略: 按 output_name (即视频 basename 或上传 uuid) 缓存. 同名复用.
          threshold 变更不自动重处理 (需调 /api/video/reprocess).
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from detection.hailo_runner import HailoMultiRunner
from detection.person_attr import DEFAULT_THRESHOLD, PersonAttrClassifier
from detection.yolov8_person import DEFAULT_SCORE_THRESHOLD, YoloV8PersonDetector

from .prerender import PrerenderJob
from .stats import StatsAggregator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    yolo_hef: str
    attr_hef: str
    video_path: str
    yolo_score_thresh: float = DEFAULT_SCORE_THRESHOLD
    attr_thresh: float = DEFAULT_THRESHOLD
    pad_ratio: float = 0.05
    jpeg_quality: int = 80
    loop_video: bool = False  # 预渲染默认不循环 (一次性跑完整视频)


class Pipeline:
    """封装 HEF 加载 + HailoMultiRunner + 异步预渲染 + 输出缓存管理."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._default_video_path: str = os.path.abspath(config.video_path)
        self.multi: Optional[HailoMultiRunner] = None
        self.yolo: Optional[YoloV8PersonDetector] = None
        self.attr: Optional[PersonAttrClassifier] = None
        self.stats = StatsAggregator()

        self._job_lock = threading.Lock()
        self._current_job: Optional[PrerenderJob] = None
        self._job_thread: Optional[threading.Thread] = None

        # 上传 + 输出目录
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._upload_dir: str = os.path.join(base, "workspace", "uploads")
        self._output_dir: str = os.path.join(base, "workspace", "outputs")
        os.makedirs(self._upload_dir, exist_ok=True)
        os.makedirs(self._output_dir, exist_ok=True)

        # 当前激活的 output_name (供前端 <video src>)
        self._current_output_name: Optional[str] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动时: 加载 HEF, 异步触发默认视频的预渲染."""
        if not os.path.exists(self.config.yolo_hef):
            raise FileNotFoundError(f"YOLO HEF not found: {self.config.yolo_hef}")
        if not os.path.exists(self.config.attr_hef):
            raise FileNotFoundError(f"Person attribute HEF not found: {self.config.attr_hef}")

        self.multi = HailoMultiRunner().__enter__()
        self.yolo = YoloV8PersonDetector(
            self.multi,
            self.config.yolo_hef,
            score_thresh=self.config.yolo_score_thresh,
        )
        self.yolo.setup()
        self.attr = PersonAttrClassifier(
            self.multi,
            self.config.attr_hef,
            threshold=self.config.attr_thresh,
        )
        self.attr.setup()
        logger.info(
            "Pipeline ready: yolo=%s attr=%s default_video=%s",
            self.config.yolo_hef, self.config.attr_hef, self._default_video_path,
        )

        # 启动时异步预渲染默认视频
        self.start_prerender(self._default_video_path, force=False)

    def stop(self) -> None:
        self._cancel_current()
        if self.multi is not None:
            self.multi.close()
            self.multi = None

    # ------------------------------------------------------------------
    # 预渲染任务管理
    # ------------------------------------------------------------------
    def start_prerender(self, video_path: str, force: bool = False) -> Dict:
        """异步启动预渲染.

        - 若 output_name 已有缓存且 force=False, 直接复用, 不启动 worker.
        - 若 force=True 或无缓存, 取消当前 worker, 启动新 worker.

        Returns: {"output_name": str, "task_id": str, "started": bool, "cached": bool}
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_name = self._output_name_for(video_path)
        cached = self._output_cached(output_name)

        with self._job_lock:
            if cached and not force:
                self._current_output_name = output_name
                logger.info("Pipeline: cache hit for %s, no worker started", output_name)
                return {
                    "output_name": output_name,
                    "task_id": "",
                    "started": False,
                    "cached": True,
                }
            # 取消旧任务
            self._cancel_current()
            # 创建新任务
            task_id = uuid.uuid4().hex[:12]
            job = PrerenderJob(
                task_id=task_id,
                video_path=video_path,
                output_name=output_name,
                output_dir=self._output_dir,
                yolo=self.yolo,
                attr=self.attr,
                stats=self.stats,
                yolo_thresh=self.config.yolo_score_thresh,
                attr_thresh=self.config.attr_thresh,
                pad_ratio=self.config.pad_ratio,
                jpeg_quality=self.config.jpeg_quality,
                loop_video=self.config.loop_video,
            )
            self._current_job = job
            self._job_thread = threading.Thread(
                target=job.run, name=f"prerender-{task_id}", daemon=True,
            )
            self._job_thread.start()
            self._current_output_name = output_name
            logger.info("Pipeline: started prerender task %s -> %s", task_id, output_name)
            return {
                "output_name": output_name,
                "task_id": task_id,
                "started": True,
                "cached": False,
            }

    def reprocess(self, video_path: Optional[str] = None) -> Dict:
        """强制重新预渲染 (忽略缓存)."""
        if video_path is None:
            video_path = self._current_video_path() or self._default_video_path
        return self.start_prerender(video_path, force=True)

    def _cancel_current(self) -> None:
        if self._current_job is not None:
            try:
                self._current_job.cancel()
            except Exception:
                logger.exception("cancel job failed")
        if self._job_thread is not None:
            self._job_thread.join(timeout=3.0)
            self._job_thread = None
        self._current_job = None

    def current_progress(self) -> Dict:
        if self._current_job is None:
            return {
                "task_id": "",
                "status": "idle",
                "progress": 0.0,
                "frames_done": 0,
                "total_frames": 0,
                "fps_proc": 0.0,
                "eta_s": 0.0,
                "output_name": self._current_output_name or "",
                "video_path": "",
                "mp4_ready": self._current_output_name is not None
                              and self._mp4_exists(self._current_output_name),
                "json_ready": self._current_output_name is not None
                              and self._json_exists(self._current_output_name),
                "error": "",
            }
        snap = self._current_job.snapshot()
        return snap

    def _current_video_path(self) -> Optional[str]:
        if self._current_job is not None:
            return self._current_job.video_path
        return None

    # ------------------------------------------------------------------
    # 阈值调整
    # ------------------------------------------------------------------
    def set_yolo_thresh(self, value: float) -> None:
        self.config.yolo_score_thresh = float(value)
        if self.yolo is not None:
            self.yolo.set_score_thresh(float(value))

    def set_attr_thresh(self, value: float) -> None:
        self.config.attr_thresh = float(value)
        if self.attr is not None:
            self.attr.set_threshold(float(value))

    def thresholds_need_reprocess(self) -> bool:
        """若缓存的输出使用了不同阈值, 提示前端需要 reprocess."""
        name = self._current_output_name
        if not name or not self._json_exists(name):
            return False
        try:
            with open(self._json_path(name), "r", encoding="utf-8") as f:
                meta = json_lib_load(f)
        except Exception:
            return False
        yolo_meta = meta.get("yolo_score_thresh", -1)
        attr_meta = meta.get("attr_thresh", -1)
        return (
            abs(yolo_meta - self.config.yolo_score_thresh) > 1e-6
            or abs(attr_meta - self.config.attr_thresh) > 1e-6
        )

    # ------------------------------------------------------------------
    # 输出文件管理
    # ------------------------------------------------------------------
    def _output_name_for(self, video_path: str) -> str:
        return os.path.splitext(os.path.basename(video_path))[0]

    def _mp4_path(self, name: str) -> str:
        return os.path.join(self._output_dir, f"{name}_annotated.mp4")

    def _json_path(self, name: str) -> str:
        return os.path.join(self._output_dir, f"{name}_stats.json")

    def _mp4_exists(self, name: str) -> bool:
        return os.path.exists(self._mp4_path(name))

    def _json_exists(self, name: str) -> bool:
        return os.path.exists(self._json_path(name))

    def _output_cached(self, name: str) -> bool:
        return self._mp4_exists(name) and self._json_exists(name)

    def mp4_path_for(self, name: str) -> Optional[str]:
        p = self._mp4_path(name)
        return p if os.path.exists(p) else None

    def json_path_for(self, name: str) -> Optional[str]:
        p = self._json_path(name)
        return p if os.path.exists(p) else None

    def list_outputs(self) -> List[Dict]:
        items: List[Dict] = []
        try:
            files = sorted(os.listdir(self._output_dir))
        except FileNotFoundError:
            files = []
        for f in files:
            if not f.endswith("_annotated.mp4"):
                continue
            name = f[: -len("_annotated.mp4")]
            mp4 = self._mp4_path(name)
            js = self._json_path(name)
            try:
                mp4_size = os.path.getsize(mp4) if os.path.exists(mp4) else 0
            except OSError:
                mp4_size = 0
            # 读 json meta
            duration = 0.0
            width = height = fps = total = 0
            try:
                if os.path.exists(js):
                    with open(js, "r", encoding="utf-8") as fh:
                        meta = json_lib_load(fh)
                    duration = meta.get("duration_s", 0.0)
                    width = meta.get("width", 0)
                    height = meta.get("height", 0)
                    fps = meta.get("fps", 0.0)
                    total = meta.get("total_frames", 0)
            except Exception:
                pass
            items.append({
                "name": name,
                "mp4_size_bytes": mp4_size,
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total,
                "duration_s": round(duration, 2),
                "is_current": name == self._current_output_name,
            })
        return items

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def upload_dir(self) -> str:
        return self._upload_dir

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def default_video_path(self) -> str:
        return self._default_video_path

    @property
    def current_output_name(self) -> Optional[str]:
        return self._current_output_name


def json_lib_load(f):
    """薄封装, 方便测试时替换."""
    import json
    return json.load(f)