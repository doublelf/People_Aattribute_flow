#!/usr/bin/env bash
# perview 启动脚本 (主机侧)
set -euo pipefail

YOLO_HEF="${YOLO_HEF:-/home/seeed/perview/model/yolov8n.hef}"
ATTR_HEF="${ATTR_HEF:-/home/seeed/perview/model/person_attr_resnet_v1_18.hef}"
VIDEO_PATH="${VIDEO_PATH:-/home/seeed/Desktop/view_person/data/input/e1.mp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -f "$YOLO_HEF" ]]; then
  echo "ERROR: YOLO HEF not found: $YOLO_HEF" >&2
  exit 1
fi
if [[ ! -f "$ATTR_HEF" ]]; then
  echo "ERROR: Person Attribute HEF not found: $ATTR_HEF" >&2
  exit 1
fi
if [[ ! -e "$VIDEO_PATH" ]]; then
  echo "ERROR: Video source not found: $VIDEO_PATH" >&2
  exit 1
fi

if [[ ! -e /dev/hailo0 ]]; then
  echo "ERROR: /dev/hailo0 missing — Hailo-8 not detected" >&2
  exit 1
fi

VIDEO_DIR="$(realpath "$(dirname "$VIDEO_PATH")")"
VIDEO_NAME="$(basename "$VIDEO_PATH")"
CONTAINER_VIDEO_PATH="/app/video/${VIDEO_NAME}"

if command -v docker >/dev/null 2>&1; then
  echo "[perview] Docker detected, building image..."
  cd "$(dirname "$0")"
  docker build -t perview:latest .
  echo "[perview] Starting container on ${HOST}:${PORT} with video ${CONTAINER_VIDEO_PATH}..."
  sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    -e VIDEO_PATH="${CONTAINER_VIDEO_PATH}" \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    -v "${VIDEO_DIR}:/app/video:ro" \
    perview:latest
else
  echo "[perview] No Docker; running directly..."
  cd "$(dirname "$0")"
  python -m app.main \
    --yolo_hef "$YOLO_HEF" \
    --attr_hef "$ATTR_HEF" \
    --video_path "$VIDEO_PATH" \
    --host "$HOST" --port "$PORT"
fi
