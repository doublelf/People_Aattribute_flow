FROM ghcr.io/seeed-projects/recomputer-r20-cv/person_attr_resnet:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY detection/ ./detection/
COPY model/ ./model/

RUN mkdir -p /app/workspace /app/video

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV YOLO_HEF=model/yolov8n.hef \
    ATTR_HEF=model/person_attr_resnet_v1_18.hef \
    VIDEO_PATH=video/test.mp4 \
    HOST=0.0.0.0 \
    PORT=8000

ENTRYPOINT ["sh", "-c", "exec python -m app.main \
            --yolo_hef \"$YOLO_HEF\" \
            --attr_hef \"$ATTR_HEF\" \
            --video_path \"$VIDEO_PATH\" \
            --host \"$HOST\" \
            --port \"$PORT\""]
