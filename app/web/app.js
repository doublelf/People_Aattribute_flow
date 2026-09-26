// perview dashboard frontend v1.3
// Pre-render + <video> playback + currentTime sync + i18n (EN) + font scale toggle
//
// Flow:
//   1. Load /api/output/list, pick current output_name
//   2. GET /api/output/{name}.info + {name}.json  (one-time)
//   3. <video src="/api/output/{name}.mp4">
//   4. setInterval(80ms) syncWithVideo() -> lookup frames[idx] -> update dashboard
//   5. video.timeupdate + seeked -> syncWithVideo()
//   6. Upload + reprocess -> poll /api/video/progress -> loadOutput(newName)

const AGE_LABELS = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"];
const GENDER_LABELS = ["Male", "Female"];

const COLORS = {
  male: "#4a90e2",
  female: "#f5a623",
  age: ["#50e3c2", "#4a90e2", "#9013fe", "#f5a623"],
  timeline: {
    male: "#4a90e2",
    female: "#f5a623",
  },
};

const $ = (id) => document.getElementById(id);

const UPLOAD_MAX_MB = 500;

let genderRtChart, ageRtChart, genderCumChart, ageCumChart, timelineChart;

let statsJson = null;            // full per-frame data
let totalFrames = 0;
let fps = 30.0;
let duration = 0;
let currentOutput = null;
let lastFrameIdx = -1;           // dedup: skip if frame unchanged
let pollTimer = null;

// ----------------------------------------------------------------
// Font scale
// ----------------------------------------------------------------
function getFontScale() {
  return document.documentElement.classList.contains("size-18x") ? 1.8 : 1.5;
}
function chartFontSize(base = 12) {
  return Math.round(base * getFontScale() / 1.5);
}

// ----------------------------------------------------------------
// Chart helpers
// ----------------------------------------------------------------
const commonOpts = () => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  animation: { duration: 150 },
});

function makeBarChart(ctxId, labels, colors, indexAxis = "x") {
  const fs = chartFontSize();
  return new Chart(ctxId, {
    type: "bar",
    data: { labels, datasets: [{ data: labels.map(() => 0), backgroundColor: colors }] },
    options: {
      ...commonOpts(),
      indexAxis,
      plugins: {
        legend: { labels: { color: "#b3bcc7", font: { size: fs } } },
      },
      scales: {
        x: {
          ticks: { color: "#b3bcc7", font: { size: fs } },
          grid: { display: indexAxis === "y" },
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#7d8590", precision: 0, font: { size: fs } },
          grid: { color: "#2a2f37" },
        },
      },
    },
  });
}

function initCharts() {
  genderRtChart = makeBarChart(
    $("gender-rt-chart").getContext("2d"), GENDER_LABELS, [COLORS.male, COLORS.female]
  );
  ageRtChart = makeBarChart(
    $("age-rt-chart").getContext("2d"), AGE_LABELS, COLORS.age
  );
  genderCumChart = makeBarChart(
    $("gender-cum-chart").getContext("2d"), GENDER_LABELS, [COLORS.male, COLORS.female]
  );
  ageCumChart = makeBarChart(
    $("age-cum-chart").getContext("2d"), AGE_LABELS, COLORS.age
  );

  const fs = chartFontSize();
  const tlCtx = $("timeline-chart").getContext("2d");
  timelineChart = new Chart(tlCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Cum Male %",   data: [], borderColor: COLORS.timeline.male,
          backgroundColor: COLORS.timeline.male + "33",
          fill: false, tension: 0.25, pointRadius: 0 },
        { label: "Cum Female %", data: [], borderColor: COLORS.timeline.female,
          backgroundColor: COLORS.timeline.female + "33",
          fill: false, tension: 0.25, pointRadius: 0 },
      ],
    },
    options: {
      ...commonOpts(),
      plugins: {
        legend: { display: true, position: "top",
                  labels: { color: "#b3bcc7", font: { size: fs } } },
      },
      scales: {
        x: { display: false },
        y: { beginAtZero: true, max: 100,
             ticks: { color: "#7d8590", callback: (v) => v + "%", font: { size: fs } },
             grid: { color: "#2a2f37" } },
      },
    },
  });
}

// Apply current font scale to all charts (for toggle)
function applyChartFontScale() {
  const fs = chartFontSize();
  const charts = [genderRtChart, ageRtChart, genderCumChart, ageCumChart, timelineChart];
  for (const ch of charts) {
    if (!ch) continue;
    if (ch.options.plugins?.legend?.labels?.font) {
      ch.options.plugins.legend.labels.font.size = fs;
    }
    if (ch.options.scales?.x?.ticks?.font) ch.options.scales.x.ticks.font.size = fs;
    if (ch.options.scales?.y?.ticks?.font) ch.options.scales.y.ticks.font.size = fs;
    ch.update();
  }
}

function fmtTime(s) {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function setProcessStatus(state, text) {
  const dot = $("process-status");
  const txt = $("process-text");
  if (state === "ok") {
    dot.className = "dot online";
    txt.textContent = text || "Ready";
  } else if (state === "busy") {
    dot.className = "dot busy";
    txt.textContent = text || "Processing...";
  } else if (state === "error") {
    dot.className = "dot offline";
    txt.textContent = text || "Error";
  } else {
    dot.className = "dot offline";
    txt.textContent = text || "Not loaded";
  }
}

// ----------------------------------------------------------------
// Load output (video + meta + per-frame JSON)
// ----------------------------------------------------------------
async function loadOutput(name) {
  if (!name) {
    setProcessStatus("offline", "No output, please upload a video");
    return;
  }
  setProcessStatus("busy", `Loading ${name} ...`);
  try {
    // 1) info
    const infoResp = await fetch(`/api/output/${encodeURIComponent(name)}.info`);
    if (!infoResp.ok) {
      throw new Error(`info HTTP ${infoResp.status}`);
    }
    const info = await infoResp.json();
    duration = info.duration_s;
    totalFrames = info.total_frames;
    fps = info.fps;
    $("duration").textContent = fmtTime(duration);
    $("resolution").textContent = `${info.width}x${info.height}`;
    $("fps-info").textContent = `${fps.toFixed(1)} fps`;

    // 2) json
    const jsonResp = await fetch(`/api/output/${encodeURIComponent(name)}.json`);
    if (!jsonResp.ok) {
      throw new Error(`json HTTP ${jsonResp.status}`);
    }
    statsJson = await jsonResp.json();
    if (!statsJson.frames || statsJson.frames.length === 0) {
      throw new Error("Empty stats frames");
    }

    // 3) Set video src (no cache-buster needed: server sends correct Cache-Control)
    const player = $("player");
    player.src = `/api/output/${encodeURIComponent(name)}.mp4`;
    player.load();
    $("output-name").textContent = `· ${name}`;
    currentOutput = name;
    setProcessStatus("ok", `Loaded ${name} (${fmtTime(duration)})`);
    $("last-update").textContent = `Loaded · ${new Date().toLocaleTimeString()}`;
    startSync();
  } catch (e) {
    setProcessStatus("error", `Load failed: ${e.message}`);
    console.error("loadOutput failed:", e);
  }
}

// ----------------------------------------------------------------
// Sync with current video time
// ----------------------------------------------------------------
function currentFrameIdx() {
  const player = $("player");
  if (!player || !statsJson || totalFrames === 0) return 0;
  const t = player.currentTime || 0;
  let idx = Math.floor(t * fps);
  if (idx < 0) idx = 0;
  if (idx >= totalFrames) idx = totalFrames - 1;
  return idx;
}

function syncWithVideo() {
  if (!statsJson) return;
  const idx = currentFrameIdx();
  if (idx === lastFrameIdx) return;
  lastFrameIdx = idx;
  const frame = statsJson.frames[idx];
  if (!frame) return;

  // Realtime panel (current frame)
  $("current-count").textContent = frame.count || 0;
  const gender = frame.gender || {};
  genderRtChart.data.datasets[0].data = [gender.Male || 0, gender.Female || 0];
  genderRtChart.update();
  const ageRt = frame.age || {};
  ageRtChart.data.datasets[0].data = AGE_LABELS.map((k) => ageRt[k] || 0);
  ageRtChart.update();

  // Cumulative panel (up to current frame)
  const cum = frame.cumulative || {};
  $("cumulative-count").textContent = cum.unique || 0;
  const cg = cum.gender || {};
  const cp = cum.gender_pct || {};
  $("male-pct").textContent = `${(cp.Male || 0).toFixed(1)}%`;
  $("female-pct").textContent = `${(cp.Female || 0).toFixed(1)}%`;
  $("male-count").textContent = `M: ${cg.Male || 0}`;
  $("female-count").textContent = `F: ${cg.Female || 0}`;
  genderCumChart.data.datasets[0].data = [cg.Male || 0, cg.Female || 0];
  genderCumChart.update();
  const ca = cum.age || {};
  ageCumChart.data.datasets[0].data = AGE_LABELS.map((k) => ca[k] || 0);
  ageCumChart.update();

  // Timeline: redraw from start to current frame
  updateTimelineUpTo(idx);

  // Time display
  const player = $("player");
  $("time-display").textContent = `${fmtTime(player.currentTime)} / ${fmtTime(duration)}`;
  $("frame-idx").textContent = `frame ${idx + 1}/${totalFrames}`;
  $("last-update").textContent =
    `frame #${idx} · ${new Date().toLocaleTimeString()}`;
}

function updateTimelineUpTo(idx) {
  if (!statsJson || !statsJson.frames) return;
  const frames = statsJson.frames;
  const labels = [];
  const malePct = [];
  const femalePct = [];
  let lastT = -1;
  for (let i = 0; i <= idx; i++) {
    const f = frames[i];
    if (!f) continue;
    const tSec = Math.floor(f.t);
    if (tSec !== lastT) {
      labels.push(tSec + 1);
      const cum = f.cumulative || {};
      const total = cum.total || 0;
      malePct.push(total ? ((cum.gender?.Male || 0) / total) * 100 : 0);
      femalePct.push(total ? ((cum.gender?.Female || 0) / total) * 100 : 0);
      lastT = tSec;
    }
  }
  timelineChart.data.labels = labels;
  timelineChart.data.datasets[0].data = malePct;
  timelineChart.data.datasets[1].data = femalePct;
  timelineChart.update();
}

function startSync() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(syncWithVideo, 80);
  const player = $("player");
  player.addEventListener("timeupdate", syncWithVideo);
  player.addEventListener("seeked", syncWithVideo);
  player.addEventListener("loadeddata", syncWithVideo);
  player.addEventListener("play", syncWithVideo);
  player.addEventListener("pause", syncWithVideo);
}

// ----------------------------------------------------------------
// Threshold sliders (set only; do not trigger re-render)
// ----------------------------------------------------------------
function bindControls() {
  const yoloSlider = $("yolo-thresh");
  const yoloVal = $("yolo-thresh-val");
  const attrSlider = $("attr-thresh");
  const attrVal = $("attr-thresh-val");
  const reprocessBtn = $("reprocess-btn");

  let yoloDebounce = null;
  let attrDebounce = null;

  yoloSlider.addEventListener("input", () => {
    const v = Number(yoloSlider.value);
    yoloVal.textContent = v.toFixed(2);
    if (yoloDebounce) clearTimeout(yoloDebounce);
    yoloDebounce = setTimeout(async () => {
      const resp = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ yolo_score_thresh: v }),
      });
      const data = await resp.json();
      if (data.thresholds_need_reprocess) {
        $("reprocess-hint").hidden = false;
      }
    }, 250);
  });

  attrSlider.addEventListener("input", () => {
    const v = Number(attrSlider.value);
    attrVal.textContent = v.toFixed(2);
    if (attrDebounce) clearTimeout(attrDebounce);
    attrDebounce = setTimeout(async () => {
      const resp = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ attr_thresh: v }),
      });
      const data = await resp.json();
      if (data.thresholds_need_reprocess) {
        $("reprocess-hint").hidden = false;
      }
    }, 250);
  });

  reprocessBtn.addEventListener("click", async () => {
    reprocessBtn.disabled = true;
    reprocessBtn.textContent = "⏳ Processing...";
    try {
      const resp = await fetch("/api/video/reprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert(`Reprocess failed: ${data.detail || resp.statusText}`);
      } else {
        await pollProgress(data.output_name);
      }
    } catch (e) {
      alert(`Request failed: ${e.message}`);
    } finally {
      reprocessBtn.disabled = false;
      reprocessBtn.textContent = "🔄 Reprocess";
    }
  });
}

// ----------------------------------------------------------------
// Font scale toggle
// ----------------------------------------------------------------
function bindSizeToggle() {
  // Restore saved scale
  let saved = null;
  try {
    saved = localStorage.getItem("perview-font-scale");
  } catch (_) { /* ignore */ }
  if (saved === "1.8") {
    document.documentElement.classList.add("size-18x");
    updateSizeToggleUI("1.8");
  } else {
    document.documentElement.classList.add("size-15x");
    updateSizeToggleUI("1.5");
  }

  document.querySelectorAll(".size-toggle button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const scale = btn.dataset.scale;
      if (scale === "1.8") {
        document.documentElement.classList.add("size-18x");
        document.documentElement.classList.remove("size-15x");
      } else {
        document.documentElement.classList.add("size-15x");
        document.documentElement.classList.remove("size-18x");
      }
      try { localStorage.setItem("perview-font-scale", scale); } catch (_) {}
      updateSizeToggleUI(scale);
      applyChartFontScale();
    });
  });
}

function updateSizeToggleUI(activeScale) {
  document.querySelectorAll(".size-toggle button").forEach((btn) => {
    if (btn.dataset.scale === activeScale) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
}

// ----------------------------------------------------------------
// Video upload + start prerender
// ----------------------------------------------------------------
function bindUpload() {
  const btn = $("upload-btn");
  const input = $("upload-input");
  const wrap = $("upload-progress");
  const fill = $("upload-fill");
  const msg = $("upload-msg");

  btn.addEventListener("click", () => input.click());

  input.addEventListener("change", async () => {
    const f = input.files?.[0];
    if (!f) return;
    input.value = "";

    const sizeMb = f.size / (1024 * 1024);
    if (sizeMb > UPLOAD_MAX_MB) {
      alert(`File too large (${sizeMb.toFixed(1)} MB), exceeds ${UPLOAD_MAX_MB} MB limit`);
      return;
    }
    const allowedExts = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
    const ext = "." + (f.name.split(".").pop() || "").toLowerCase();
    if (!allowedExts.includes(ext)) {
      alert(`Unsupported video format: ${ext}`);
      return;
    }

    wrap.hidden = false;
    fill.style.width = "0%";
    fill.style.background = "linear-gradient(90deg, #50e3c2, #4a90e2)";
    msg.textContent = `Uploading 0% (${sizeMb.toFixed(1)} MB)`;

    try {
      const upRes = await uploadWithProgress(f, (pct) => {
        const half = pct * 50;
        fill.style.width = half + "%";
        msg.textContent = `Uploading ${pct.toFixed(0)}% (${sizeMb.toFixed(1)} MB)`;
      });
      if (!upRes.ok) {
        const err = await upRes.json().catch(() => ({ detail: upRes.statusText }));
        throw new Error(err.detail || `HTTP ${upRes.status}`);
      }
      const upData = await upRes.json();
      fill.style.width = "60%";
      msg.textContent = `Uploaded (${upData.video?.width}x${upData.video?.height}), pre-rendering...`;

      const startRes = await fetch("/api/video/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: upData.filename, loop: false }),
      });
      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({ detail: startRes.statusText }));
        throw new Error(err.detail || `HTTP ${startRes.status}`);
      }
      const startData = await startRes.json();
      if (startData.status === "cached") {
        await loadOutput(startData.output_name);
        fill.style.width = "100%";
        msg.textContent = `Using cache ${startData.output_name}`;
        setTimeout(() => { wrap.hidden = true; }, 1500);
        return;
      }
      await pollProgress(startData.output_name);
      fill.style.width = "100%";
      msg.textContent = `Done ${startData.output_name}`;
      setTimeout(() => { wrap.hidden = true; }, 3000);
    } catch (e) {
      fill.style.background = "#ff5252";
      msg.textContent = `Failed: ${e.message}`;
      setTimeout(() => {
        wrap.hidden = true;
        fill.style.background = "";
        fill.style.width = "0%";
      }, 5000);
    }
  });
}

function uploadWithProgress(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/video/upload");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) onProgress((e.loaded / e.total) * 100);
    });
    xhr.onload = () => {
      resolve({
        ok: xhr.status >= 200 && xhr.status < 300,
        status: xhr.status,
        json: () => Promise.resolve(JSON.parse(xhr.responseText || "{}")),
      });
    };
    xhr.onerror = () => reject(new Error("Network error"));
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

async function pollProgress(outputName) {
  setProcessStatus("busy", `Pre-rendering ${outputName} ...`);
  return new Promise((resolve) => {
    const start = Date.now();
    const tick = async () => {
      try {
        const resp = await fetch("/api/video/progress");
        const p = await resp.json();
        const pct = Math.round((p.progress || 0) * 100);
        const msg = $("upload-msg");
        const fill = $("upload-fill");
        if (!$("upload-progress").hidden) {
          msg.textContent =
            `Pre-render ${pct}% (${p.frames_done || 0}/${p.total_frames || 0} frames, ` +
            `${(p.fps_proc || 0).toFixed(1)} fps, ETA ${(p.eta_s || 0).toFixed(0)}s)`;
          fill.style.width = (60 + (p.progress || 0) * 40) + "%";
        }
        if (p.status === "done") {
          await loadOutput(outputName);
          resolve();
          return;
        } else if (p.status === "error") {
          msg.textContent = `Error: ${p.error || "unknown"}`;
          resolve();
          return;
        } else if (p.status === "cancelled") {
          msg.textContent = "Cancelled";
          resolve();
          return;
        }
        if (Date.now() - start > 30 * 60 * 1000) {
          msg.textContent = "Timeout";
          resolve();
          return;
        }
        setTimeout(tick, 500);
      } catch (e) {
        const msg = $("upload-msg");
        msg.textContent = `Poll failed: ${e.message}`;
        resolve();
      }
    };
    tick();
  });
}

// ----------------------------------------------------------------
// Bootstrap
// ----------------------------------------------------------------
async function loadInitialConfig() {
  try {
    const cfgResp = await fetch("/api/config");
    if (cfgResp.ok) {
      const cfg = await cfgResp.json();
      if (typeof cfg.yolo_score_thresh === "number") {
        $("yolo-thresh").value = cfg.yolo_score_thresh.toFixed(2);
        $("yolo-thresh-val").textContent = cfg.yolo_score_thresh.toFixed(2);
      }
      if (typeof cfg.attr_thresh === "number") {
        $("attr-thresh").value = cfg.attr_thresh.toFixed(2);
        $("attr-thresh-val").textContent = cfg.attr_thresh.toFixed(2);
      }
      if (cfg.thresholds_need_reprocess) {
        $("reprocess-hint").hidden = false;
      }
    }
    const listResp = await fetch("/api/output/list");
    if (listResp.ok) {
      const list = await listResp.json();
      const name = list.current || (list.outputs?.[0]?.name);
      if (name) {
        await loadOutput(name);
        return;
      }
    }
  } catch (e) {
    console.warn("loadInitialConfig failed:", e);
  }
  setProcessStatus("offline", "No output, please upload a video");
}

document.addEventListener("DOMContentLoaded", async () => {
  bindSizeToggle();   // must run before initCharts so font scale is known
  initCharts();
  bindControls();
  bindUpload();
  await loadInitialConfig();
});