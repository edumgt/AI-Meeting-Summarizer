import { marked } from "https://cdn.jsdelivr.net/npm/marked@12.0.2/lib/marked.esm.js";

const API_BASE = window.API_BASE || "http://localhost:8000";
const $ = (id) => document.getElementById(id);

const state = {
  mediaRecorder: null,
  mediaStream: null,
  recordedChunks: [],
  recordTimer: null,
  recordStartedAt: null,
  currentReport: null,
  selectedPdf: null,
  health: null,
};

$("apiBaseTop").textContent = API_BASE;

function getAiProvider() {
  return document.querySelector('input[name="aiProvider"]:checked')?.value || "local";
}

function getReportSettings() {
  return {
    meeting_title: $("meetingTitle").value.trim() || "회의록",
    meeting_date_hint: $("dateHint").value || "",
    include_summary: $("includeSummary").checked,
    report_format: $("reportFormat").value,
    ai_provider: getAiProvider(),
  };
}

function setStatus(message, tone = "normal") {
  const text = message || "대기";
  $("status").textContent = text;
  $("statusTop").textContent = text;
  $("status").classList.toggle("danger", tone === "error");
}

function formatDuration(ms) {
  const s = Math.floor(ms / 1000);
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function buildRecordedFile() {
  if (!state.recordedChunks.length) return null;
  const blob = new Blob(state.recordedChunks, { type: "audio/webm" });
  return new File([blob], `meeting-${Date.now()}.webm`, { type: blob.type });
}

function stopStream() {
  if (!state.mediaStream) return;
  state.mediaStream.getTracks().forEach((track) => track.stop());
  state.mediaStream = null;
}

function updateRecorderUi() {
  const recording = !!state.mediaRecorder && state.mediaRecorder.state === "recording";
  $("btnStartRecording").disabled = recording;
  $("btnStopRecording").disabled = !recording;
  $("btnUploadAudio").disabled = recording;
  $("btnTranscribe").disabled = recording;
  if (!recording) $("recordTime").textContent = "00:00";
}

function updateModeUi() {
  const provider = getAiProvider();
  const format = $("reportFormat").value;
  $("summaryEngine").textContent = provider === "openai" ? "OpenAI" : "Local AI";
  $("currentFormat").textContent = format;
  $("aiModeHint").textContent =
    provider === "openai"
      ? "OpenAI 모드에서는 OpenAI 요약 모델을 사용합니다. 오디오 전사는 기존처럼 OpenAI STT를 사용합니다."
      : "Local AI 모드에서는 텍스트 정리와 PDF 정리를 로컬 요약 모델로 처리합니다.";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderAttachmentList() {
  const holder = $("attachmentList");
  holder.innerHTML = "";
  if (!state.selectedPdf) return;

  const item = document.createElement("div");
  item.className = "attachment-item";
  item.innerHTML = `
    <div>
      <strong>${escapeHtml(state.selectedPdf.name)}</strong>
      <div class="muted">${Math.round(state.selectedPdf.size / 1024)}KB PDF</div>
    </div>
    <button class="btn ghost" id="btnRemovePdf">제거</button>
  `;
  holder.appendChild(item);
  $("btnRemovePdf")?.addEventListener("click", () => {
    state.selectedPdf = null;
    $("pdfFile").value = "";
    $("pdfInfo").textContent = "PDF 업로드 대기 중입니다.";
    renderAttachmentList();
  });
}

function renderAudioDownload(url, fileName) {
  $("audioDownload").innerHTML = "";
  if (!url || !fileName) return;

  const wrapper = document.createElement("div");
  wrapper.className = "attachment-item";
  wrapper.innerHTML = `
    <div>
      <strong>서버 저장 음성</strong>
      <div class="muted">${escapeHtml(fileName)}</div>
    </div>
  `;
  const link = document.createElement("a");
  link.href = `${API_BASE}${url}`;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.className = "btn";
  link.textContent = "mp3 다운로드";
  wrapper.appendChild(link);
  $("audioDownload").appendChild(wrapper);
}

function createReportMarkdownDownload() {
  if (!state.currentReport?.markdown) {
    alert("저장할 회의록이 없습니다.");
    return;
  }
  const blob = new Blob([state.currentReport.markdown], { type: "text/markdown;charset=utf-8" });
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = href;
  link.download = `${(state.currentReport.title || "meeting-report").replace(/\s+/g, "-")}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

function openPrintWindow() {
  if (!state.currentReport?.markdown) {
    alert("PDF로 저장할 회의록이 없습니다.");
    return;
  }

  const html = marked.parse(state.currentReport.markdown);
  const win = window.open("", "_blank", "width=960,height=720");
  if (!win) {
    alert("팝업이 차단되어 PDF 저장 창을 열 수 없습니다.");
    return;
  }

  win.document.write(`
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <title>${escapeHtml(state.currentReport.title || "회의록")}</title>
      <style>
        body { font-family: "Noto Sans KR", sans-serif; margin: 36px; color: #111827; line-height: 1.7; }
        h1, h2, h3 { margin-top: 1.4em; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }
        pre, code { background: #f3f4f6; border-radius: 8px; }
      </style>
    </head>
    <body>${html}</body>
    </html>
  `);
  win.document.close();
  win.focus();
  setTimeout(() => win.print(), 250);
}

function openMailDraft() {
  if (!state.currentReport?.markdown) {
    alert("메일로 보낼 회의록이 없습니다.");
    return;
  }
  const to = $("emailTo").value.trim();
  const subject = $("emailSubject").value.trim() || `[회의록] ${state.currentReport.title || "회의록"}`;
  const body = encodeURIComponent(state.currentReport.markdown);
  window.location.href = `mailto:${encodeURIComponent(to)}?subject=${encodeURIComponent(subject)}&body=${body}`;
}

function renderReport(payload, extra = {}) {
  const title = extra.title || $("meetingTitle").value.trim() || "회의록";
  const transcript = extra.transcript || payload.transcript || payload.text || "(원문 없음)";
  const summary = payload.summary ? `요약 포함 · ${payload.summary.slice(0, 120)}${payload.summary.length > 120 ? "..." : ""}` : "요약 없음";
  const sourceLabel = extra.sourceLabel || "텍스트 입력";
  const formatLabel = getReportSettings().report_format;
  const provider = getAiProvider();

  state.currentReport = {
    title,
    markdown: payload.markdown,
    extracted: payload.extracted,
    transcript,
    summary: payload.summary || "",
    sourceLabel,
    formatLabel,
    provider,
  };

  $("reportMd").innerHTML = marked.parse(payload.markdown || "");
  $("extractJson").textContent = JSON.stringify(payload.extracted || {}, null, 2);
  $("transcriptText").textContent = transcript;
  $("reportMeta").textContent = `${sourceLabel} · ${provider === "openai" ? "OpenAI" : "Local AI"} · ${formatLabel} · ${summary}`;
  $("emailSubject").value = `[회의록] ${title}`;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

async function postAudioAndTranscribe(audioFile) {
  const fd = new FormData();
  const settings = getReportSettings();
  fd.append("audio", audioFile, audioFile.name || "meeting.webm");
  Object.entries(settings).forEach(([key, value]) => fd.append(key, String(value)));

  const res = await fetch(`${API_BASE}/transcribe-and-report`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

async function postPdfReport(pdfFile) {
  const fd = new FormData();
  const settings = getReportSettings();
  fd.append("pdf", pdfFile, pdfFile.name);
  Object.entries(settings).forEach(([key, value]) => fd.append(key, String(value)));

  const res = await fetch(`${API_BASE}/pdf-report`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.health = await res.json();
    $("pdfSupportState").textContent = state.health.pdf_support ? "지원됨" : "설치 필요";
  } catch (error) {
    console.error(error);
    $("pdfSupportState").textContent = "확인 실패";
  }
}

function clearAll() {
  $("inputText").value = "";
  $("audioFile").value = "";
  $("pdfFile").value = "";
  $("reportMd").textContent = "(아직 없음)";
  $("extractJson").textContent = "{}";
  $("transcriptText").textContent = "(아직 없음)";
  $("reportMeta").textContent = "아직 생성된 회의록이 없습니다.";
  $("audioDownload").innerHTML = "";
  $("recordInfo").textContent = "녹음 또는 파일 업로드 후 회의록 생성을 실행하세요.";
  $("pdfInfo").textContent = "PDF 업로드 대기 중입니다.";
  $("emailTo").value = "";
  state.recordedChunks = [];
  state.selectedPdf = null;
  state.currentReport = null;
  renderAttachmentList();
  setStatus("대기");
  updateRecorderUi();
}

function initOffcanvas() {
  const offcanvas = $("offcanvas");
  const backdrop = $("ocBackdrop");

  const open = () => {
    offcanvas.classList.add("open");
    backdrop.classList.add("open");
    offcanvas.setAttribute("aria-hidden", "false");
    backdrop.setAttribute("aria-hidden", "false");
  };

  const close = () => {
    offcanvas.classList.remove("open");
    backdrop.classList.remove("open");
    offcanvas.setAttribute("aria-hidden", "true");
    backdrop.setAttribute("aria-hidden", "true");
  };

  $("btnMenu").addEventListener("click", open);
  $("btnMenuClose").addEventListener("click", close);
  backdrop.addEventListener("click", close);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
  });

  document.querySelectorAll("[data-scroll-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.scrollTarget);
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
      close();
    });
  });
}

$("btnClear").addEventListener("click", clearAll);

$("btnReport").addEventListener("click", async () => {
  const text = $("inputText").value.trim();
  if (!text) {
    alert("회의록 텍스트를 입력하세요.");
    return;
  }

  $("btnReport").disabled = true;
  setStatus("텍스트 회의록 생성 중...");

  try {
    const payload = await postJson(`${API_BASE}/report`, {
      text,
      ...getReportSettings(),
    });
    renderReport(payload, { sourceLabel: "텍스트 입력", transcript: "(텍스트 입력 모드)" });
    renderAudioDownload();
    setStatus("텍스트 회의록 완료");
  } catch (error) {
    console.error(error);
    alert(String(error));
    setStatus("텍스트 회의록 실패", "error");
  } finally {
    $("btnReport").disabled = false;
  }
});

$("btnStartRecording").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("현재 브라우저에서 녹음을 지원하지 않습니다.");
    return;
  }

  try {
    state.recordedChunks = [];
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaRecorder = new MediaRecorder(state.mediaStream);

    state.mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) state.recordedChunks.push(event.data);
    });

    state.mediaRecorder.addEventListener("stop", () => {
      stopStream();
      if (state.recordTimer) clearInterval(state.recordTimer);
      updateRecorderUi();
      const file = buildRecordedFile();
      if (file) {
        $("recordInfo").textContent = `녹음 파일 준비됨: ${file.name} (${Math.round(file.size / 1024)}KB)`;
      }
    });

    state.mediaRecorder.start(1000);
    state.recordStartedAt = Date.now();
    $("recordInfo").textContent = "녹음 중입니다. 종료 후 회의록 생성을 실행하세요.";
    state.recordTimer = setInterval(() => {
      $("recordTime").textContent = formatDuration(Date.now() - state.recordStartedAt);
    }, 500);
    updateRecorderUi();
  } catch (error) {
    console.error(error);
    stopStream();
    if (state.recordTimer) clearInterval(state.recordTimer);
    updateRecorderUi();
    alert(`녹음 시작 실패: ${String(error)}`);
  }
});

$("btnStopRecording").addEventListener("click", () => {
  if (!state.mediaRecorder || state.mediaRecorder.state !== "recording") return;
  state.mediaRecorder.stop();
});

$("btnUploadAudio").addEventListener("click", () => $("audioFile").click());

$("audioFile").addEventListener("change", () => {
  const file = $("audioFile").files?.[0];
  if (!file) return;
  $("recordInfo").textContent = `업로드 파일 선택됨: ${file.name} (${Math.round(file.size / 1024)}KB)`;
});

$("btnTranscribe").addEventListener("click", async () => {
  const uploaded = $("audioFile").files?.[0] || null;
  const recorded = buildRecordedFile();
  const sourceFile = uploaded || recorded;

  if (!sourceFile) {
    alert("먼저 오디오 파일을 선택하거나 브라우저에서 녹음을 진행하세요.");
    return;
  }

  $("btnTranscribe").disabled = true;
  setStatus("오디오 전사 및 회의록 생성 중...");

  try {
    const payload = await postAudioAndTranscribe(sourceFile);
    $("inputText").value = payload.transcript || "";
    renderReport(payload, { sourceLabel: "오디오 입력", transcript: payload.transcript || "(전사 없음)" });
    renderAudioDownload(payload.mp3_download_url, payload.mp3_file_name);
    setStatus("오디오 회의록 완료");
  } catch (error) {
    console.error(error);
    alert(String(error));
    setStatus("오디오 회의록 실패", "error");
  } finally {
    $("btnTranscribe").disabled = false;
  }
});

$("btnSelectPdf").addEventListener("click", () => $("pdfFile").click());

$("pdfFile").addEventListener("change", () => {
  const file = $("pdfFile").files?.[0];
  if (!file) return;
  state.selectedPdf = file;
  $("pdfInfo").textContent = `PDF 선택됨: ${file.name} (${Math.round(file.size / 1024)}KB)`;
  renderAttachmentList();
});

$("btnPdfReport").addEventListener("click", async () => {
  if (!state.selectedPdf) {
    alert("먼저 PDF 파일을 선택하세요.");
    return;
  }

  $("btnPdfReport").disabled = true;
  setStatus("PDF 회의록 생성 중...");

  try {
    const payload = await postPdfReport(state.selectedPdf);
    $("inputText").value = payload.text || "";
    renderReport(payload, { sourceLabel: "PDF 업로드", transcript: payload.text || "(PDF 텍스트 없음)" });
    $("pdfInfo").textContent = `PDF 정리 완료: ${state.selectedPdf.name}`;
    setStatus("PDF 회의록 완료");
  } catch (error) {
    console.error(error);
    alert(String(error));
    setStatus("PDF 회의록 실패", "error");
  } finally {
    $("btnPdfReport").disabled = false;
  }
});

$("btnDownloadMd").addEventListener("click", createReportMarkdownDownload);
$("btnSavePdf").addEventListener("click", openPrintWindow);
$("btnSendMail").addEventListener("click", openMailDraft);

document.querySelectorAll('input[name="aiProvider"]').forEach((input) => {
  input.addEventListener("change", updateModeUi);
});
$("reportFormat").addEventListener("change", updateModeUi);

initOffcanvas();
updateRecorderUi();
updateModeUi();
fetchHealth();
clearAll();
