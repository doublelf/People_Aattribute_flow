"""视频源 (FrameSource) + JPEG 编码工具.

本模块已被简化: v1.2 起不再做实时推理推流, 仅保留 PrerenderJob
需要的工具 (FrameSource 读视频, make_jpeg 给 dashboard 用).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource:
    """视频文件源 (循环/不循环读取)."""

    def __init__(self, video_path: str, loop: bool = True) -> None:
        self.video_path = video_path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "FrameSource opened: %s (%dx%d @ %.1f fps, %d frames)",
            self.video_path, self.width, self.height, self.fps, self.frame_count,
        )

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    return None
            else:
                return None
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reset(self) -> None:
        """回到首帧 (供 prerender 复用)."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def metadata(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.frame_count,
            "duration_s": (self.frame_count / self.fps) if self.fps > 0 else 0.0,
        }


def make_jpeg(frame: np.ndarray, quality: int = 85) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


def open_ffmpeg_writer(path: str, width: int, height: int, fps: float) -> Optional[tuple]:
    """启动 ffmpeg libx264 ultrafast 子进程; 返回 (proc, kind="ffmpeg") 或 None.

    失败时返回 None (调用方 fallback cv2 mp4v).
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        path,
    ]
    import subprocess
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return proc, "ffmpeg"
    except FileNotFoundError:
        return None


def open_cv2_writer(path: str, width: int, height: int, fps: float) -> Optional[tuple]:
    """fallback: cv2 mp4v. 返回 (writer, kind='mp4v') 或 None."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if w.isOpened():
        return w, "mp4v"
    w.release()
    return None


def close_writer(writer_tuple) -> None:
    """统一关闭 ffmpeg/cv2 writer."""
    if writer_tuple is None:
        return
    obj, kind = writer_tuple
    if kind == "ffmpeg":
        try:
            obj.stdin.close()
        except Exception:
            pass
        try:
            obj.wait(timeout=30)
        except Exception:
            try:
                obj.kill()
            except Exception:
                pass
    else:
        try:
            obj.release()
        except Exception:
            pass


def write_frame(writer_tuple, frame: np.ndarray) -> None:
    obj, kind = writer_tuple
    if kind == "ffmpeg":
        obj.stdin.write(frame.tobytes())
    else:
        obj.write(frame)