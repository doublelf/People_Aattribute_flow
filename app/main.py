"""FastAPI entry point (v1.2: 预渲染 + 回放).

Routes:
    GET  /                            -> dashboard HTML
    GET  /static/{name}               -> CSS/JS

    GET  /api/output/list             -> 已完成预渲染列表
    GET  /api/output/{name}.info      -> 元信息
    GET  /api/output/{name}.mp4       -> 录制视频 (HTTP Range)
    GET  /api/output/{name}.json      -> per-frame 数据 (gzip)
    GET  /api/output/{name}.frame/{idx}.jpg -> 单帧 JPEG (可选)

    GET  /api/config                  -> 当前阈值 + 是否需要重处理提示
    POST /api/config                  -> 修改阈值 (不影响已缓存)

    POST /api/video/upload            -> 上传视频到 workspace/uploads/
    POST /api/video/start             -> 异步触发预渲染, 返回 task_id
    POST /api/video/reprocess         -> 强制重新预渲染 (忽略缓存)
    GET  /api/video/progress          -> 预渲染进度
    POST /api/video/stop              -> 取消当前预渲染任务
    GET  /api/video/list              -> 已上传列表

    POST /api/models/yolov8/predict            -> 单图检测
    POST /api/models/person_attr_resnet/predict -> 单图属性
    POST /api/pipeline/predict                 -> 单图端到端
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.draw import draw_panel, draw_person
from app.pipeline import Pipeline, PipelineConfig
from app.stream import make_jpeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("perview")


# ---------------------------------------------------------------------------
# 全局
# ---------------------------------------------------------------------------
PIPELINE: Optional[Pipeline] = None
PIPELINE_LOCK = threading.Lock()
UPLOAD_MAX_BYTES = 500 * 1024 * 1024  # 500MB
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
NAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")


def get_pipeline() -> Pipeline:
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return PIPELINE


def _safe_name(name: str) -> bool:
    if not name or ".." in name or "/" in name or "\\" in name:
        return False
    return bool(NAME_RE.match(name))


# ---------------------------------------------------------------------------
# 路由辅助
# ---------------------------------------------------------------------------
def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def _safe_crop(frame: np.ndarray, bbox, pad_ratio: float = 0.05):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(w, int(round(x2 + pad_x)))
    cy2 = min(h, int(round(y2 + pad_y)))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE
    cfg = app.state.pipeline_config
    with PIPELINE_LOCK:
        PIPELINE = Pipeline(cfg)
        PIPELINE.start()
    logger.info("FastAPI startup complete")
    try:
        yield
    finally:
        with PIPELINE_LOCK:
            if PIPELINE is not None:
                PIPELINE.stop()
                PIPELINE = None
        logger.info("FastAPI shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="perview · Person Counting (reComputer R + Hailo-8)",
    version="1.2.0",
    lifespan=lifespan,
)

WEB_DIR = Path(__file__).parent / "web"


# ---------------------------------------------------------------------------
# 页面与静态资源
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = WEB_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>perview</h1><p>UI not built.</p>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/static/{name}")
async def static_asset(name: str):
    if not _safe_name(name):
        raise HTTPException(status_code=400, detail="Invalid name")
    path = WEB_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    media = "text/css" if name.endswith(".css") else "application/javascript"
    return FileResponse(path, media_type=media)


# ---------------------------------------------------------------------------
# 输出文件 (预渲染产物)
# ---------------------------------------------------------------------------
@app.get("/api/output/list")
async def output_list():
    pipe = get_pipeline()
    return {
        "outputs": pipe.list_outputs(),
        "current": pipe.current_output_name or "",
    }


@app.get("/api/output/{name}.info")
async def output_info(name: str):
    if not _safe_name(name):
        raise HTTPException(status_code=400, detail="Invalid name")
    pipe = get_pipeline()
    js_path = pipe.json_path_for(name)
    if js_path is None:
        raise HTTPException(status_code=404, detail=f"Output not found: {name}")
    import json
    try:
        with open(js_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except OSError:
        raise HTTPException(status_code=500, detail="Read json failed")
    return {
        "name": name,
        "video": meta.get("video", ""),
        "width": meta.get("width", 0),
        "height": meta.get("height", 0),
        "fps": meta.get("fps", 0.0),
        "total_frames": meta.get("total_frames", 0),
        "duration_s": meta.get("duration_s", 0.0),
        "yolo_score_thresh": meta.get("yolo_score_thresh", 0.5),
        "attr_thresh": meta.get("attr_thresh", 0.5),
        "created_at": meta.get("created_at", 0.0),
    }


@app.get("/api/output/{name}.mp4")
async def output_mp4(name: str):
    if not _safe_name(name):
        raise HTTPException(status_code=400, detail="Invalid name")
    pipe = get_pipeline()
    path = pipe.mp4_path_for(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"MP4 not found: {name}")
    # FileResponse 自动支持 HTTP Range (关键: 让 <video> 能 seek)
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/output/{name}.json")
async def output_json(name: str):
    if not _safe_name(name):
        raise HTTPException(status_code=400, detail="Invalid name")
    pipe = get_pipeline()
    path = pipe.json_path_for(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"JSON not found: {name}")
    return FileResponse(
        path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def api_get_config():
    pipe = get_pipeline()
    return {
        "yolo_score_thresh": pipe.config.yolo_score_thresh,
        "attr_thresh": pipe.config.attr_thresh,
        "current_output": pipe.current_output_name or "",
        "thresholds_need_reprocess": pipe.thresholds_need_reprocess(),
    }


@app.post("/api/config")
async def api_set_config(payload: dict):
    pipe = get_pipeline()
    changed = False
    if "yolo_score_thresh" in payload:
        try:
            v = float(payload["yolo_score_thresh"])
            if not (0.0 < v <= 1.0):
                raise ValueError
            pipe.set_yolo_thresh(v)
            changed = True
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="yolo_score_thresh must be in (0, 1]")
    if "attr_thresh" in payload:
        try:
            v = float(payload["attr_thresh"])
            if not (0.0 < v <= 1.0):
                raise ValueError
            pipe.set_attr_thresh(v)
            changed = True
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="attr_thresh must be in (0, 1]")
    if "reset_cumulative" in payload and bool(payload["reset_cumulative"]):
        # 重渲染会重置 stats; 此处不立即做
        pass
    return {
        "status": "success",
        "yolo_score_thresh": pipe.config.yolo_score_thresh,
        "attr_thresh": pipe.config.attr_thresh,
        "thresholds_need_reprocess": pipe.thresholds_need_reprocess(),
        "note": "thresholds updated; cached output unchanged. Click Reprocess to regenerate."
        if changed else "no changes",
    }


# ---------------------------------------------------------------------------
# 视频上传 + 预渲染触发
# ---------------------------------------------------------------------------
@app.post("/api/video/upload")
async def video_upload(file: UploadFile = File(...)):
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {ext}. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}",
        )
    pipe = PIPELINE
    os.makedirs(pipe.upload_dir, exist_ok=True)
    dest_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(pipe.upload_dir, dest_name)
    total = 0
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > UPLOAD_MAX_BYTES:
                    f.close()
                    try:
                        os.remove(dest_path)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (> {UPLOAD_MAX_BYTES // (1024*1024)} MB)",
                    )
                f.write(chunk)
    finally:
        await file.close()
    cap = cv2.VideoCapture(dest_path)
    if not cap.isOpened():
        try:
            os.remove(dest_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Invalid or unreadable video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = (total_frames / fps) if fps > 0 else 0.0
    cap.release()
    return {
        "filename": dest_name,
        "original_name": file.filename,
        "size_bytes": total,
        "size_mb": round(total / (1024 * 1024), 2),
        "video": {
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": total_frames,
            "duration_s": round(duration, 2),
        },
    }


@app.post("/api/video/start")
async def video_start(payload: dict):
    """异步触发预渲染任务."""
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")
    pipe = PIPELINE
    if filename in ("default", "__default__"):
        video_path = pipe.default_video_path
    else:
        if not _safe_name(filename):
            raise HTTPException(status_code=400, detail="Invalid filename")
        video_path = os.path.join(pipe.upload_dir, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    try:
        info = pipe.start_prerender(video_path, force=False)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "status": "queued" if info["started"] else "cached",
        "output_name": info["output_name"],
        "task_id": info["task_id"],
        "video_path": video_path,
        "is_default": filename in ("default", "__default__"),
    }


@app.post("/api/video/reprocess")
async def video_reprocess(payload: Optional[dict] = None):
    """强制重新预渲染 (忽略缓存)."""
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipe = PIPELINE
    payload = payload or {}
    filename = payload.get("filename")
    if filename in (None, "default", "__default__"):
        # 重新预渲染当前 default
        video_path = pipe.default_video_path
    elif not _safe_name(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    else:
        video_path = os.path.join(pipe.upload_dir, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    info = pipe.start_prerender(video_path, force=True)
    return {
        "status": "queued",
        "output_name": info["output_name"],
        "task_id": info["task_id"],
        "video_path": video_path,
    }


@app.get("/api/video/progress")
async def video_progress():
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return PIPELINE.current_progress()


@app.post("/api/video/stop")
async def video_stop():
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipe = PIPELINE
    # 通过 reprocess(None, force=False) 不会停, 需直接 cancel. 简化: cancel 当前 job
    if pipe._current_job is not None:  # noqa: SLF001 (内部使用)
        pipe._current_job.cancel()      # noqa: SLF001
    return {"status": "cancellation_requested"}


@app.get("/api/video/list")
async def video_list():
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipe = PIPELINE
    try:
        files = sorted(os.listdir(pipe.upload_dir))
    except FileNotFoundError:
        files = []
    items = []
    for f in files:
        full = os.path.join(pipe.upload_dir, f)
        try:
            size = os.path.getsize(full)
        except OSError:
            size = 0
        items.append({"filename": f, "size_bytes": size})
    return {
        "uploads": items,
        "default": os.path.basename(pipe.default_video_path),
    }


# ---------------------------------------------------------------------------
# 单图推理 (兼容 seeed 规范, 用于调试)
# ---------------------------------------------------------------------------
@app.post("/api/models/yolov8/predict")
async def predict_yolov8(file: UploadFile = File(...)):
    pipe = get_pipeline()
    data = await file.read()
    img = _decode_image(data)
    dets, yolo_ms = pipe.yolo.detect_with_timing(img)
    predictions = []
    for d in dets:
        predictions.append({
            "class": "person",
            "cls": d.cls,
            "confidence": float(d.score),
            "box": {
                "x1": int(d.bbox[0]),
                "y1": int(d.bbox[1]),
                "x2": int(d.bbox[2]),
                "y2": int(d.bbox[3]),
            },
        })
    return {
        "model": "yolov8n",
        "image": {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "yolo_ms": yolo_ms,
        "predictions": predictions,
    }


@app.post("/api/models/person_attr_resnet/predict")
async def predict_attr(file: UploadFile = File(...), threshold: Optional[float] = None):
    pipe = get_pipeline()
    data = await file.read()
    img = _decode_image(data)
    old_th = pipe.config.attr_thresh
    if threshold is not None:
        try:
            pipe.set_attr_thresh(float(threshold))
        except ValueError:
            raise HTTPException(status_code=400, detail="threshold must be float")
    try:
        attrs, dt = pipe.attr.infer_crop_with_timing(img)
    finally:
        if threshold is not None:
            pipe.set_attr_thresh(old_th)
    canvas = img.copy()
    for a in attrs:
        if a.get("group") in ("gender", "age"):
            cv2.putText(
                canvas,
                f"{a['class']}: {a['confidence']:.2f}",
                (20, 30 + 30 * attrs.index(a)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
    jpeg = make_jpeg(canvas, quality=85)
    image_b64 = base64.b64encode(jpeg).decode("utf-8") if jpeg else None
    return {
        "model": "person_attr_resnet",
        "predictions": attrs,
        "attributes": attrs,
        "threshold": pipe.config.attr_thresh,
        "inference_ms": dt,
        "image": image_b64,
    }


@app.post("/api/pipeline/predict")
async def predict_pipeline(file: UploadFile = File(...)):
    pipe = get_pipeline()
    data = await file.read()
    img = _decode_image(data)
    dets, yolo_ms = pipe.yolo.detect_with_timing(img)
    persons = []
    t0 = __import__("time").perf_counter()
    for d in dets:
        crop = _safe_crop(img, d.bbox, pad_ratio=pipe.config.pad_ratio)
        if crop is None:
            continue
        try:
            attrs, _dt = pipe.attr.infer_crop_with_timing(crop)
        except Exception as e:
            logger.warning("attr infer error: %s", e)
            attrs = []
        persons.append({
            "class": "person",
            "confidence": float(d.score),
            "box": {
                "x1": int(d.bbox[0]),
                "y1": int(d.bbox[1]),
                "x2": int(d.bbox[2]),
                "y2": int(d.bbox[3]),
            },
            "attributes": attrs,
        })
    attr_ms = (__import__("time").perf_counter() - t0) * 1000.0

    canvas = img.copy()
    for p in persons:
        bbox = (p["box"]["x1"], p["box"]["y1"], p["box"]["x2"], p["box"]["y2"])
        draw_person(canvas, bbox, p["confidence"], p["attributes"])
    snap = pipe.stats.update(persons, yolo_ms=yolo_ms, attr_ms=attr_ms)
    draw_panel(canvas, snap)
    jpeg = make_jpeg(canvas, quality=85)

    return {
        "model": "pipeline",
        "image": {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "yolo_ms": yolo_ms,
        "attr_ms": attr_ms,
        "predictions": persons,
        "stats": snap.to_dict(),
        "image_jpeg": base64.b64encode(jpeg).decode("utf-8") if jpeg else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> tuple:
    parser = argparse.ArgumentParser(description="perview · Person counting on reComputer R + Hailo-8")
    parser.add_argument("--yolo_hef", default="model/yolov8n.hef")
    parser.add_argument("--attr_hef", default="model/person_attr_resnet_v1_18.hef")
    parser.add_argument("--video_path", default="video/test.mp4")
    parser.add_argument("--yolo_score_thresh", type=float, default=0.5)
    parser.add_argument("--attr_thresh", type=float, default=0.5)
    parser.add_argument("--pad_ratio", type=float, default=0.05)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--loop_video", action="store_true",
                        help="预渲染时循环读取视频 (默认: 一次性跑完整视频)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(), parser


def main() -> None:
    args, parser = parse_args()
    cfg = PipelineConfig(
        yolo_hef=args.yolo_hef,
        attr_hef=args.attr_hef,
        video_path=args.video_path,
        yolo_score_thresh=args.yolo_score_thresh,
        attr_thresh=args.attr_thresh,
        pad_ratio=args.pad_ratio,
        jpeg_quality=args.jpeg_quality,
        loop_video=args.loop_video,
    )
    if not os.path.exists(cfg.yolo_hef):
        parser.error(f"YOLO HEF not found: {cfg.yolo_hef}")
    if not os.path.exists(cfg.attr_hef):
        parser.error(f"Person Attribute HEF not found: {cfg.attr_hef}")
    if not os.path.exists(cfg.video_path):
        parser.error(f"Video file not found: {cfg.video_path}")

    app.state.pipeline_config = cfg
    logger.info(
        "Starting perview: host=%s port=%d yolo=%s attr=%s video=%s",
        args.host, args.port, cfg.yolo_hef, cfg.attr_hef, cfg.video_path,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()