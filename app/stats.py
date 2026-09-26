"""Thread-safe statistics aggregator.

每帧推理完成后调用 update(persons). /api/stats 读取最近一次 snapshot.

数据结构:
  - 实时 (0.5s 滑动窗): current_count, gender, age
  - 累计 (启动以来): cumulative_unique, cumulative_gender, cumulative_age,
                      cumulative_total, gender_pct, age_pct
  - 全程时间线: timeline (每秒一桶, 含累计快照, 用于前端折线图)

累计去重策略: 简易 IoU tracker. 每个 track 第一次出现时, 把它
的 gender/age 计入 cumulative_*; 离开再回来 (track 消失后重新匹配)
不重复计入, 但 cumulative_unique += 1 (因为是"另一个新行人的首次出现" —
通过 5s 后未被匹配触发新 track 实现).
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


AGE_LABELS: Tuple[str, ...] = ("Age16-30", "Age31-45", "Age46-60", "AgeAbove61")
GENDER_LABELS: Tuple[str, ...] = ("Male", "Female")


@dataclass
class StatsSnapshot:
    # 实时 (0.5s 滑动窗)
    current_count: int = 0
    gender: Dict[str, int] = field(default_factory=dict)
    age: Dict[str, int] = field(default_factory=dict)

    # 累计 (启动以来)
    cumulative_unique: int = 0
    cumulative_gender: Dict[str, int] = field(default_factory=dict)
    cumulative_age: Dict[str, int] = field(default_factory=dict)
    cumulative_total: int = 0
    gender_pct: Dict[str, float] = field(default_factory=dict)
    age_pct: Dict[str, float] = field(default_factory=dict)

    # 时间线 (全程历史, 每秒一桶)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    # 性能
    fps: float = 0.0
    inference_ms: float = 0.0
    yolo_ms: float = 0.0
    attr_ms: float = 0.0

    # 元信息
    timestamp: float = 0.0
    frame_id: int = 0
    current_video: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _PersonTrack:
    """无外观特征 tracker, 用 IoU + 中心距离做关联."""

    __slots__ = ("key", "cx", "cy", "w", "h", "last_seen", "bboxes")

    def __init__(self, cx: float, cy: float, w: float, h: float, now: float) -> None:
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.last_seen = now
        self.bboxes = [(cx, cy, w, h)]
        self.key = id(self)

    def update(self, cx: float, cy: float, w: float, h: float, now: float) -> None:
        a = 0.4
        self.cx = self.cx * (1 - a) + cx * a
        self.cy = self.cy * (1 - a) + cy * a
        self.w = self.w * (1 - a) + w * a
        self.h = self.h * (1 - a) + h * a
        self.last_seen = now
        self.bboxes.append((cx, cy, w, h))
        if len(self.bboxes) > 8:
            self.bboxes = self.bboxes[-8:]

    def distance(self, cx: float, cy: float, w: float, h: float) -> float:
        denom = max(1.0, (w + h) * 0.5)
        return ((cx - self.cx) ** 2 + (cy - self.cy) ** 2) ** 0.5 / denom


class StatsAggregator:
    """线程安全的统计聚合器.

    累计去重: 简易 tracker (IoU + 中心距离), 每个 track 在
    离开画面 (N 秒未出现) 后才计为"新出现" +1.
    累计 gender/age: 每个 track 第一次出现时把 gender/age 计入累计.
    """

    COOLDOWN_SEC = 2.0
    TRACK_FORGET_SEC = 5.0
    SMOOTH_WINDOW = 0.5
    MAX_TRACKS = 256
    TIMELINE_BUCKET_SEC = 1.0   # 每秒一桶
    TIMELINE_MAX_BUCKETS = 86400  # 24h 上限

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = StatsSnapshot()
        self._tracks: List[_PersonTrack] = []
        self._track_attrs: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
        self._track_counted: Dict[int, bool] = {}  # 是否已计入 cumulative
        self._samples: Deque[Tuple[float, int, Counter, Counter]] = deque(maxlen=64)
        self._fps_window_start: float = time.time()
        self._fps_window_count: int = 0
        self._frame_id: int = 0
        # 时间线累积
        self._cum_gender: Counter = Counter()
        self._cum_age: Counter = Counter()
        self._cum_unique: int = 0
        self._timeline: List[Dict[str, Any]] = []
        self._last_bucket_ts: float = 0.0

    def set_current_video(self, name: str) -> None:
        with self._lock:
            self._snap.current_video = name

    def _associate(
        self, centers: List[Tuple[float, float, float, float]], now: float
    ) -> Tuple[List[Tuple[int, _PersonTrack]], List[int]]:
        """Greedy associate current frame detections with existing tracks."""
        self._tracks = [
            t for t in self._tracks if (now - t.last_seen) <= self.TRACK_FORGET_SEC
        ]
        # 清理已过期 track 的 attrs
        alive_keys = {id(t) for t in self._tracks}
        for k in list(self._track_attrs.keys()):
            if k not in alive_keys:
                self._track_attrs.pop(k, None)
                self._track_counted.pop(k, None)

        used: set = set()
        pairs: List[Tuple[int, _PersonTrack]] = []
        for i, (cx, cy, w, h) in enumerate(centers):
            best_t: Optional[_PersonTrack] = None
            best_d = float("inf")
            for t in self._tracks:
                if id(t) in used:
                    continue
                d = t.distance(cx, cy, w, h)
                if d < best_d and d < 0.6:
                    best_d = d
                    best_t = t
            if best_t is not None:
                used.add(id(best_t))
                best_t.update(cx, cy, w, h, now)
                pairs.append((i, best_t))
        matched = {i for i, _ in pairs}
        unmatched = [i for i in range(len(centers)) if i not in matched]
        return pairs, unmatched

    def _extract_attrs(
        self, attrs: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str]]:
        gender: Optional[str] = None
        age: Optional[str] = None
        for a in attrs:
            grp = a.get("group", "")
            cls = a.get("class", "")
            if grp == "gender" and gender is None:
                gender = cls
            elif grp == "age" and age is None:
                age = cls
        return gender, age

    def _bucket_gender(self, g: Optional[str]) -> Optional[str]:
        if g in GENDER_LABELS:
            return g
        return None

    def _bucket_age(self, a: Optional[str]) -> Optional[str]:
        if a in AGE_LABELS:
            return a
        return None

    def update(
        self,
        persons: List[Dict[str, Any]],
        yolo_ms: float = 0.0,
        attr_ms: float = 0.0,
    ) -> StatsSnapshot:
        now = time.time()
        gender_c: Counter = Counter()
        age_c: Counter = Counter()

        centers: List[Tuple[float, float, float, float]] = []
        for p in persons:
            bbox = p.get("bbox", (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            centers.append((cx, cy, w, h))

        pairs, unmatched = self._associate(centers, now)

        for i, t in pairs:
            attrs = persons[i].get("attrs", [])
            g, a = self._extract_attrs(attrs)
            if g:
                gender_c[g] += 1
            if a:
                age_c[a] += 1
            # 累计: 仅当 track 首次出现时, 把它的 gender/age 计入
            if not self._track_counted.get(t.key, False):
                gb = self._bucket_gender(g)
                ab = self._bucket_age(a)
                if gb:
                    self._cum_gender[gb] += 1
                if ab:
                    self._cum_age[ab] += 1
                self._track_counted[t.key] = True
                self._track_attrs[t.key] = (gb, ab)
                self._cum_unique += 1

        for i in unmatched:
            cx, cy, w, h = centers[i]
            if len(self._tracks) >= self.MAX_TRACKS:
                self._tracks.sort(key=lambda t: t.last_seen)
                old = self._tracks.pop(0)
                self._track_attrs.pop(old.key, None)
                self._track_counted.pop(old.key, None)
            new_t = _PersonTrack(cx, cy, w, h, now)
            self._tracks.append(new_t)
            attrs = persons[i].get("attrs", [])
            g, a = self._extract_attrs(attrs)
            if g:
                gender_c[g] += 1
            if a:
                age_c[a] += 1
            gb = self._bucket_gender(g)
            ab = self._bucket_age(a)
            if gb:
                self._cum_gender[gb] += 1
            if ab:
                self._cum_age[ab] += 1
            self._track_counted[new_t.key] = True
            self._track_attrs[new_t.key] = (gb, ab)
            self._cum_unique += 1

        # 时间线: 每秒一桶
        if now - self._last_bucket_ts >= self.TIMELINE_BUCKET_SEC:
            cum_g = dict(self._cum_gender)
            cum_a = dict(self._cum_age)
            cum_total = sum(cum_g.values()) or 0
            self._timeline.append({
                "t": now,
                "cum_total": cum_total,
                "cum_gender": cum_g,
                "cum_age": cum_a,
                "cum_unique": self._cum_unique,
            })
            if len(self._timeline) > self.TIMELINE_MAX_BUCKETS:
                # 截断头部
                self._timeline = self._timeline[-self.TIMELINE_MAX_BUCKETS:]
            self._last_bucket_ts = now

        # 计算百分比
        cum_total = sum(self._cum_gender.values()) or 0
        gender_pct = (
            {k: round(v / cum_total * 100, 1) for k, v in self._cum_gender.items()}
            if cum_total > 0 else {}
        )
        age_pct = (
            {k: round(v / cum_total * 100, 1) for k, v in self._cum_age.items()}
            if cum_total > 0 else {}
        )

        with self._lock:
            self._frame_id += 1
            self._samples.append((now, len(persons), gender_c, age_c))
            while self._samples and (now - self._samples[0][0]) > self.SMOOTH_WINDOW * 4:
                self._samples.popleft()

            self._fps_window_count += 1
            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                fps = self._fps_window_count / elapsed
                self._fps_window_start = now
                self._fps_window_count = 0
            else:
                fps = self._snap.fps

            recent = [s for s in self._samples if (now - s[0]) <= self.SMOOTH_WINDOW]
            if recent:
                avg_count = sum(s[1] for s in recent) / len(recent)
                merged_gender: Counter = Counter()
                merged_age: Counter = Counter()
                for _, _, g, a in recent:
                    merged_gender.update(g)
                    merged_age.update(a)
                current_count = int(round(avg_count))
            else:
                current_count = len(persons)
                merged_gender = gender_c
                merged_age = age_c

            snap = StatsSnapshot(
                current_count=current_count,
                gender=dict(merged_gender),
                age=dict(merged_age),
                cumulative_unique=self._cum_unique,
                cumulative_gender=dict(self._cum_gender),
                cumulative_age=dict(self._cum_age),
                cumulative_total=cum_total,
                gender_pct=gender_pct,
                age_pct=age_pct,
                timeline=list(self._timeline),
                fps=fps,
                yolo_ms=yolo_ms,
                attr_ms=attr_ms,
                inference_ms=yolo_ms + attr_ms,
                timestamp=now,
                frame_id=self._frame_id,
                current_video=self._snap.current_video,
            )
            self._snap = snap
            return snap

    def reset_cumulative(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._track_attrs.clear()
            self._track_counted.clear()
            self._cum_gender.clear()
            self._cum_age.clear()
            self._cum_unique = 0
            self._timeline.clear()
            self._last_bucket_ts = 0.0
            self._snap.cumulative_unique = 0
            self._snap.cumulative_gender = {}
            self._snap.cumulative_age = {}
            self._snap.cumulative_total = 0
            self._snap.gender_pct = {}
            self._snap.age_pct = {}
            self._snap.timeline = []

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            return self._snap
