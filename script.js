const LOOP_MS = 8400;

const stages = [
  { at: 0, index: "01", label: "读取长视频", state: "正在建立内容索引" },
  { at: 950, index: "02", label: "扫描多模态时间轴", state: "融合语义、场景与情绪信号" },
  { at: 2800, index: "03", label: "标记高价值切点", state: "优化语句完整性与片段边界" },
  { at: 4700, index: "04", label: "生成 15-60 秒短片", state: "多条高光切片已准备就绪" },
  { at: 7350, index: "01", label: "读取长视频", state: "正在重启演示循环" },
];

const stageLabel = document.querySelector("#stageLabel");
const stageIndex = document.querySelector("#stageIndex");
const processState = document.querySelector("#processState");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let cycleTimer = null;
let stageTimers = [];

function renderStage(stage) {
  stageLabel.textContent = stage.label;
  stageIndex.textContent = stage.index;
  processState.textContent = stage.state;
}

function clearStageTimers() {
  stageTimers.forEach(window.clearTimeout);
  stageTimers = [];
  window.clearTimeout(cycleTimer);
  cycleTimer = null;
}

function runStageCycle() {
  clearStageTimers();
  stages.forEach((stage) => {
    stageTimers.push(window.setTimeout(() => renderStage(stage), stage.at));
  });
  cycleTimer = window.setTimeout(runStageCycle, LOOP_MS);
}

function syncMotionPreference() {
  clearStageTimers();
  if (reduceMotion.matches) {
    renderStage({
      index: "04",
      label: "高光短片已生成",
      state: "静态预览模式",
    });
    return;
  }
  runStageCycle();
}

reduceMotion.addEventListener("change", syncMotionPreference);
syncMotionPreference();

const revealItems = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window && !reduceMotion.matches) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.14 }
  );
  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}

const trialButton = document.querySelector("#trialButton");
const trialDialog = document.querySelector("#trialDialog");
const videoInput = document.querySelector("#videoInput");
const dropZone = document.querySelector("#dropZone");
const fileLabel = document.querySelector("#fileLabel");
const fileHelp = document.querySelector("#fileHelp");
const formMessage = document.querySelector("#formMessage");
const analyzeButton = document.querySelector("#analyzeButton");
const cancelJobButton = document.querySelector("#cancelJobButton");
const targetDuration = document.querySelector("#targetDuration");
const outputFormat = document.querySelector("#outputFormat");
const captionsEnabled = document.querySelector("#captionsEnabled");
const jobProgress = document.querySelector("#jobProgress");
const progressPhase = document.querySelector("#progressPhase");
const progressValue = document.querySelector("#progressValue");
const progressTrack = document.querySelector("#progressTrack");
const progressBar = document.querySelector("#progressBar");
const jobResult = document.querySelector("#jobResult");
const resultSummary = document.querySelector("#resultSummary");
const resultGrid = document.querySelector("#resultGrid");

let selectedFile = null;
let activeJobId = null;

trialButton.addEventListener("click", () => {
  trialDialog.showModal();
});

trialDialog.addEventListener("click", (event) => {
  if (event.target === trialDialog) trialDialog.close();
});

function setSelectedFile(file) {
  if (activeJobId) return;
  const supportedExtensions = [".mp4", ".mov", ".mkv", ".webm", ".m4v"];
  formMessage.classList.remove("is-error");
  jobProgress.hidden = true;
  jobResult.hidden = true;
  resultGrid.replaceChildren();
  activeJobId = null;

  if (!file) {
    selectedFile = null;
    fileLabel.textContent = "选择视频，或拖放到这里";
    fileHelp.textContent = "单个文件，建议时长 10 分钟以上";
    formMessage.textContent = "";
    analyzeButton.disabled = true;
    return;
  }

  const lowercaseName = file.name.toLowerCase();
  const isSupported = supportedExtensions.some((extension) => lowercaseName.endsWith(extension));
  if (!isSupported) {
    selectedFile = null;
    fileLabel.textContent = "这个格式暂不支持";
    fileHelp.textContent = "请选择 MP4、MOV、MKV、WebM 或 M4V 文件";
    formMessage.textContent = "文件未被添加，请重新选择。";
    formMessage.classList.add("is-error");
    analyzeButton.disabled = true;
    return;
  }

  selectedFile = file;
  const sizeInMb = Math.max(0.1, file.size / 1024 / 1024).toFixed(1);
  fileLabel.textContent = file.name;
  fileHelp.textContent = `${sizeInMb} MB，原文件会在本机分析结束后删除`;
  formMessage.textContent = "视频已就绪，可以开始分析。";
  analyzeButton.disabled = false;
}

function setTaskControlsLocked(locked) {
  videoInput.disabled = locked;
  targetDuration.disabled = locked;
  outputFormat.disabled = locked;
  captionsEnabled.disabled = locked;
  dropZone.classList.toggle("is-disabled", locked);
}

videoInput.addEventListener("change", () => {
  setSelectedFile(videoInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  setSelectedFile(file);
});

function updateProgress(progress, phase) {
  const safeProgress = Math.min(100, Math.max(0, Number(progress) || 0));
  jobProgress.hidden = false;
  progressPhase.textContent = phase;
  progressValue.textContent = `${safeProgress}%`;
  progressTrack.setAttribute("aria-valuenow", String(safeProgress));
  progressBar.style.transform = `scaleX(${safeProgress / 100})`;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function renderResult(result, jobId, savedFeedback = []) {
  const { metadata, signals, clips } = result;
  const exportProfile = result.export_profile ?? {
    key: "source",
    label: "保持原画幅",
    width: metadata.width,
    height: metadata.height,
    captions_rendered: 0,
  };
  const previewFormat =
    exportProfile.key !== "source"
      ? exportProfile.key
      : exportProfile.width / exportProfile.height < 0.8
        ? "portrait"
        : Math.abs(exportProfile.width / exportProfile.height - 1) < 0.08
          ? "square"
          : "landscape";
  const rankingProfile = result.scoring_profile;
  const rankingLabel =
    rankingProfile?.status === "active" && rankingProfile.version > 0
      ? `反馈排序 v${rankingProfile.version} · ${rankingProfile.sample_count} 条样本`
      : "默认排序权重";
  const duplicateCount = signals.selection?.visual_filtered ?? 0;
  const visualLabel = duplicateCount
    ? `已过滤 ${duplicateCount} 个重复画面候选`
    : `${signals.visual_samples ?? 0} 个视觉指纹`;
  const exportLabel = exportProfile.captions_rendered
    ? `${exportProfile.label} · ${exportProfile.captions_rendered} 条字幕`
    : exportProfile.label;
  const speakerLabel = signals.speaker_turns
    ? ` · ${signals.speaker_count} 位说话人 / ${signals.speaker_turns} 个轮次`
    : "";
  const performance = result.performance ?? {};
  const totalSeconds = Number(performance.total_ms) / 1000;
  const throughput = Number(
    performance.source_seconds_per_processing_second
  );
  const performanceLabel =
    performance.total_ms != null && Number.isFinite(totalSeconds)
    ? ` · 处理 ${totalSeconds.toFixed(1)}s${
        performance.source_seconds_per_processing_second != null &&
        Number.isFinite(throughput)
          ? ` / ${throughput.toFixed(1)}× 实时`
          : ""
      }`
    : "";
  const cacheLabel = result.cache?.hit
    ? " · 相同内容已安全复用"
    : "";
  const feedbackByClip = new Map(
    savedFeedback.map((item) => [item.clip_filename, item.rating])
  );
  resultSummary.textContent =
    `${metadata.width}×${metadata.height} · ${formatTime(metadata.duration)}` +
    `${performanceLabel}${cacheLabel} · ` +
    `${signals.scene_changes} 个场景变化 · ${signals.black_ranges ?? 0} 个黑帧区间 · ` +
    `${signals.semantic_spans ?? 0} 条文本证据${speakerLabel} · ` +
    `${visualLabel} · ${exportLabel} · ${rankingLabel}`;
  resultGrid.replaceChildren();

  clips.forEach((clip, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.classList.add(`format-${previewFormat}`);

    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = clip.url;
    video.setAttribute("playsinline", "");

    const copy = document.createElement("div");
    copy.className = "result-card-copy";
    const row = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `候选片段 ${String(index + 1).padStart(2, "0")}`;
    const time = document.createElement("time");
    time.textContent = `${formatTime(clip.start)}–${formatTime(clip.end)}`;
    row.append(title, time);

    const evidence = document.createElement("p");
    const warnings = clip.quality?.warnings ?? [];
    const renderEvidence = [];
    if (clip.render?.caption_count) {
      renderEvidence.push(`自动字幕 ${clip.render.caption_count} 条`);
    }
    if (clip.render?.audio?.normalized) {
      renderEvidence.push(`${clip.render.audio.target_lufs} LUFS`);
    }
    if (clip.render?.delivery_check?.av_sync_delta_seconds != null) {
      renderEvidence.push(
        `音画差 ${Math.round(
          clip.render.delivery_check.av_sync_delta_seconds * 1000
        )}ms`
      );
    }
    evidence.textContent = [
      ...clip.evidence,
      ...renderEvidence,
      ...warnings,
    ].join(" · ");
    if (warnings.length) card.classList.add("has-warning");
    const semanticPreview = document.createElement("p");
    semanticPreview.className = "semantic-preview";
    semanticPreview.textContent = clip.semantic?.preview
      ? `“${clip.semantic.preview}”`
      : "";
    const download = document.createElement("a");
    download.href = clip.url;
    download.download = clip.filename;
    download.textContent = "下载 MP4 ↗";

    const actions = document.createElement("div");
    actions.className = "result-actions";
    const feedbackActions = document.createElement("div");
    feedbackActions.className = "feedback-actions";
    const feedbackButtons = ["keep", "reject"].map((rating) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "feedback-action";
      button.textContent = rating === "keep" ? "保留" : "不合适";
      button.setAttribute("aria-pressed", "false");
      if (feedbackByClip.get(clip.filename) === rating) {
        button.classList.add("is-selected");
        button.setAttribute("aria-pressed", "true");
      }
      button.addEventListener("click", () => {
        submitFeedback(jobId, clip.filename, rating, feedbackButtons);
      });
      return button;
    });
    feedbackActions.append(...feedbackButtons);
    actions.append(download, feedbackActions);
    copy.append(row, evidence);
    if (semanticPreview.textContent) copy.append(semanticPreview);
    copy.append(actions);
    card.append(video, copy);
    resultGrid.append(card);
  });

  jobResult.hidden = false;
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.detail || "服务返回了未知错误";
  } catch {
    return "无法连接本地分析服务";
  }
}

async function submitFeedback(jobId, clipFilename, rating, buttons) {
  buttons.forEach((button) => {
    button.disabled = true;
  });
  try {
    const response = await fetch(`/api/jobs/${jobId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clip_filename: clipFilename,
        rating,
      }),
    });
    if (!response.ok) throw new Error(await readError(response));
    const payload = await response.json();
    buttons.forEach((button) => {
      const selected =
        (button.textContent === "保留" && rating === "keep") ||
        (button.textContent === "不合适" && rating === "reject");
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    formMessage.classList.remove("is-error");
    if (payload.calibration?.status === "active") {
      formMessage.textContent =
        `反馈已记录，排序模型已更新为 v${payload.calibration.version}。`;
    } else if (payload.calibration?.status === "insufficient_data") {
      const remaining = Math.max(
        0,
        (payload.calibration.minimums?.samples ?? 12) -
          payload.calibration.sample_count
      );
      formMessage.textContent = remaining
        ? `反馈已记录，再积累 ${remaining} 条可训练反馈后自动校准。`
        : "反馈已记录，正在等待更均衡的保留与淘汰样本。";
    } else {
      formMessage.textContent = "反馈已记录；新权重未优于当前基线，继续使用安全配置。";
    }
  } catch (error) {
    formMessage.textContent = error.message;
    formMessage.classList.add("is-error");
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

async function waitForJob(jobId) {
  while (activeJobId === jobId) {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) throw new Error(await readError(response));
    const job = await response.json();
    updateProgress(job.progress, job.phase);
    if (job.status === "completed") {
      renderResult(job.result, job.id, job.feedback ?? []);
      return job;
    }
    if (job.status === "cancelled") return job;
    if (job.status === "failed") throw new Error(job.error || "视频分析失败");
    await new Promise((resolve) => window.setTimeout(resolve, 900));
  }
}

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "正在上传";
  formMessage.classList.remove("is-error");
  formMessage.textContent = "";
  jobResult.hidden = true;
  updateProgress(2, "上传视频到本机服务");
  setTaskControlsLocked(true);

  const formData = new FormData();
  formData.append("video", selectedFile);
  formData.append("target_duration", targetDuration.value);
  formData.append("output_format", outputFormat.value);
  formData.append("captions_enabled", String(captionsEnabled.checked));

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error(await readError(response));
    const job = await response.json();
    activeJobId = job.id;
    cancelJobButton.hidden = false;
    analyzeButton.textContent = "正在分析";
    const finishedJob = await waitForJob(job.id);
    if (finishedJob.status === "cancelled") {
      jobProgress.hidden = true;
      analyzeButton.textContent = "开始分析";
      formMessage.textContent = "任务已取消，原视频已从本机任务目录删除。";
    } else {
      analyzeButton.textContent = "重新分析";
      formMessage.textContent = "真实媒体分析已完成，候选短片保存在本机。";
    }
  } catch (error) {
    jobProgress.hidden = true;
    formMessage.textContent =
      error instanceof TypeError
        ? "未连接本地服务，请先运行 python server.py。"
        : error.message;
    formMessage.classList.add("is-error");
    analyzeButton.textContent = "开始分析";
  } finally {
    activeJobId = null;
    cancelJobButton.hidden = true;
    cancelJobButton.disabled = false;
    setTaskControlsLocked(false);
    analyzeButton.disabled = false;
  }
});

cancelJobButton.addEventListener("click", async () => {
  if (!activeJobId) return;
  cancelJobButton.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${activeJobId}/cancel`, {
      method: "POST",
    });
    if (!response.ok) throw new Error(await readError(response));
    const job = await response.json();
    updateProgress(job.progress, job.phase);
    formMessage.textContent = "取消请求已提交，当前媒体步骤结束后停止。";
  } catch (error) {
    formMessage.textContent = error.message;
    formMessage.classList.add("is-error");
    cancelJobButton.disabled = false;
  }
});

window.addEventListener(
  "pagehide",
  () => {
    clearStageTimers();
  },
  { once: true }
);
