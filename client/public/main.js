import { marked } from "https://cdn.jsdelivr.net/npm/marked@12.0.2/lib/marked.esm.js";

const API_BASE = window.API_BASE ?? "http://localhost:8000";
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
  selectedAnalystPdf: null,
  selectedCallAudio: null,
};

$("apiBaseTop").textContent = API_BASE;

function getAiProvider() {
  return document.querySelector('input[name="aiProvider"]:checked')?.value || "local";
}

function getOllamaModel() {
  return $("ollamaModel").value || "";
}

function getReportSettings() {
  const provider = getAiProvider();
  const settings = {
    meeting_title: $("meetingTitle").value.trim() || "회의록",
    meeting_date_hint: $("dateHint").value || "",
    include_summary: $("includeSummary").checked,
    report_format: $("reportFormat").value,
    ai_provider: provider,
  };
  if (provider === "ollama") {
    settings.ollama_model = getOllamaModel();
  }
  return settings;
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

  const engineLabels = { openai: "OpenAI", ollama: "Ollama", local: "Local AI" };
  $("summaryEngine").textContent = engineLabels[provider] || "Local AI";
  $("currentFormat").textContent = format;

  const hints = {
    openai: "OpenAI 모드에서는 OpenAI 요약 모델을 사용합니다. 오디오 전사는 기존처럼 OpenAI STT를 사용합니다.",
    ollama: "Ollama 모드에서는 로컬 Ollama 서버의 LLM 모델로 회의록을 요약합니다. 오디오 전사는 OpenAI STT를 사용합니다.",
    local: "Local AI 모드에서는 텍스트 정리와 PDF 정리를 로컬 요약 모델로 처리합니다.",
  };
  $("aiModeHint").textContent = hints[provider] || hints.local;
  $("ollamaModelField").style.display = provider === "ollama" ? "" : "none";
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
  const engineLabel = provider === "openai" ? "OpenAI" : provider === "ollama" ? `Ollama (${getOllamaModel() || "default"})` : "Local AI";
  $("reportMeta").textContent = `${sourceLabel} · ${engineLabel} · ${formatLabel} · ${summary}`;
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

async function postAnalystReportPdf(pdfFile, analystFirm, reportDateHint) {
  const fd = new FormData();
  fd.append("pdf", pdfFile, pdfFile.name);
  fd.append("analyst_firm", analystFirm || "");
  fd.append("report_date_hint", reportDateHint || "");

  const res = await fetch(`${API_BASE}/analyst-report/pdf`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

async function postCallSummary(audioFile, settings) {
  const fd = new FormData();
  fd.append("audio", audioFile, audioFile.name || "call.webm");
  Object.entries(settings).forEach(([key, value]) => fd.append(key, String(value)));

  const res = await fetch(`${API_BASE}/call-summary`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

function renderAnalystResult(payload) {
  $("analystResult").textContent = JSON.stringify(payload.recommendations || [], null, 2);
}

const WORLD_TIMEZONES = [
  { country: "한국", tz: "Asia/Seoul" },
  { country: "미국(뉴욕)", tz: "America/New_York" },
  { country: "일본", tz: "Asia/Tokyo" },
  { country: "중국", tz: "Asia/Shanghai" },
  { country: "유럽(독일)", tz: "Europe/Berlin" },
  { country: "영국", tz: "Europe/London" },
  { country: "홍콩", tz: "Asia/Hong_Kong" },
];

function renderWorldClock() {
  const now = new Date();
  $("worldClockBody").innerHTML = WORLD_TIMEZONES.map(({ country, tz }) => {
    let timeText = "-";
    try {
      timeText = new Intl.DateTimeFormat("ko-KR", {
        timeZone: tz,
        hour12: false,
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }).format(now);
    } catch (error) {
      console.error(`시간대 변환 실패: ${tz}`, error);
    }
    return `<tr><td style="padding:6px 8px;">${escapeHtml(country)}</td><td style="padding:6px 8px;">${escapeHtml(timeText)}</td></tr>`;
  }).join("");
}

function startWorldClock() {
  renderWorldClock();
  setInterval(renderWorldClock, 1000);
}

function renderFxTable(fx, fxError) {
  if (fxError) {
    $("fxBody").innerHTML = `<tr><td colspan="3" class="danger" style="padding:6px 8px;">환율 조회 실패: ${escapeHtml(fxError)}</td></tr>`;
    return;
  }
  if (!fx || !fx.length) {
    $("fxBody").innerHTML = `<tr><td colspan="3" class="muted" style="padding:6px 8px;">데이터 없음</td></tr>`;
    return;
  }
  $("fxBody").innerHTML = fx.map((row) => {
    if (row.error) {
      return `<tr><td style="padding:6px 8px;">${escapeHtml(row.country)}</td><td style="padding:6px 8px;">${escapeHtml(row.currency)}</td><td class="danger" style="text-align:right; padding:6px 8px;">조회 실패</td></tr>`;
    }
    return `<tr><td style="padding:6px 8px;">${escapeHtml(row.country)}</td><td style="padding:6px 8px;">${escapeHtml(row.unit_label || row.currency)}</td><td style="text-align:right; padding:6px 8px;">${Number(row.krw_per_unit).toLocaleString("ko-KR")}원</td></tr>`;
  }).join("");
}

function renderIndexTable(indices) {
  if (!indices || !indices.length) {
    $("indexBody").innerHTML = `<tr><td colspan="4" class="muted" style="padding:6px 8px;">데이터 없음</td></tr>`;
    return;
  }
  $("indexBody").innerHTML = indices.map((row) => {
    if (row.error) {
      return `<tr><td style="padding:6px 8px;">${escapeHtml(row.country)}</td><td style="padding:6px 8px;">${escapeHtml(row.index_label)}</td><td colspan="2" class="danger" style="text-align:right; padding:6px 8px;">조회 실패</td></tr>`;
    }
    const changeClass = row.change_pct > 0 ? "" : row.change_pct < 0 ? "danger" : "";
    const changeSign = row.change_pct > 0 ? "+" : "";
    return `<tr>
      <td style="padding:6px 8px;">${escapeHtml(row.country)}</td>
      <td style="padding:6px 8px;">${escapeHtml(row.index_label)}</td>
      <td style="text-align:right; padding:6px 8px;">${Number(row.price).toLocaleString("ko-KR")}</td>
      <td class="${changeClass}" style="text-align:right; padding:6px 8px;">${changeSign}${row.change_pct}%</td>
    </tr>`;
  }).join("");
}

async function fetchMarketOverview(forceRefresh = false) {
  $("marketMeta").textContent = "글로벌 마켓 데이터를 불러오는 중...";
  try {
    const url = `${API_BASE}/market-overview${forceRefresh ? "?refresh=true" : ""}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const payload = await res.json();
    renderFxTable(payload.fx, payload.fx_error);
    renderIndexTable(payload.indices);
    $("marketMeta").textContent = `업데이트 시각(UTC): ${payload.updated_at}`;
  } catch (error) {
    console.error(error);
    $("marketMeta").textContent = `글로벌 마켓 데이터 로딩 실패: ${String(error)}`;
  }
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

async function fetchOllamaModels() {
  try {
    const res = await fetch(`${API_BASE}/ollama/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const select = $("ollamaModel");
    select.innerHTML = "";
    data.models.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      if (name === data.default) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (error) {
    console.error("Ollama 모델 목록 로드 실패:", error);
    $("ollamaModel").innerHTML = '<option value="">Ollama 연결 실패</option>';
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

$("btnAnalystText").addEventListener("click", async () => {
  const text = $("analystText").value.trim();
  if (!text) {
    alert("리포트 텍스트를 입력하세요.");
    return;
  }
  $("btnAnalystText").disabled = true;
  setStatus("애널리스트 리포트 추출 중...");
  try {
    const payload = await postJson(`${API_BASE}/analyst-report`, {
      text,
      analyst_firm: $("analystFirm").value.trim() || null,
      report_date_hint: $("analystReportDate").value || null,
    });
    renderAnalystResult(payload);
    setStatus("애널리스트 리포트 추출 완료");
  } catch (error) {
    console.error(error);
    alert(String(error));
    setStatus("애널리스트 리포트 추출 실패", "error");
  } finally {
    $("btnAnalystText").disabled = false;
  }
});

$("btnSelectAnalystPdf").addEventListener("click", () => $("analystPdfFile").click());

$("analystPdfFile").addEventListener("change", () => {
  const file = $("analystPdfFile").files?.[0];
  if (!file) return;
  state.selectedAnalystPdf = file;
  $("analystPdfInfo").textContent = `PDF 선택됨: ${file.name} (${Math.round(file.size / 1024)}KB)`;
});

$("btnAnalystPdf").addEventListener("click", async () => {
  if (!state.selectedAnalystPdf) {
    alert("먼저 PDF 파일을 선택하세요.");
    return;
  }
  $("btnAnalystPdf").disabled = true;
  setStatus("애널리스트 PDF 추출 중...");
  try {
    const payload = await postAnalystReportPdf(
      state.selectedAnalystPdf,
      $("analystFirm").value.trim(),
      $("analystReportDate").value
    );
    renderAnalystResult(payload);
    $("analystPdfInfo").textContent = `PDF 추출 완료: ${state.selectedAnalystPdf.name}`;
    setStatus("애널리스트 PDF 추출 완료");
  } catch (error) {
    console.error(error);
    alert(String(error));
    setStatus("애널리스트 PDF 추출 실패", "error");
  } finally {
    $("btnAnalystPdf").disabled = false;
  }
});

$("btnConsensus").addEventListener("click", async () => {
  const ticker = $("consensusTicker").value.trim();
  if (!ticker) {
    alert("종목코드를 입력하세요. 예: 005930");
    return;
  }
  $("btnConsensus").disabled = true;
  setStatus("컨센서스 조회 중...");
  try {
    const res = await fetch(`${API_BASE}/analyst-consensus/${encodeURIComponent(ticker)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const payload = await res.json();
    $("consensusResult").textContent = JSON.stringify(payload, null, 2);
    setStatus("컨센서스 조회 완료");
  } catch (error) {
    console.error(error);
    $("consensusResult").textContent = String(error);
    setStatus("컨센서스 조회 실패", "error");
  } finally {
    $("btnConsensus").disabled = false;
  }
});

$("btnSelectCallAudio").addEventListener("click", () => $("callAudioFile").click());

$("callAudioFile").addEventListener("change", () => {
  const file = $("callAudioFile").files?.[0];
  if (!file) return;
  state.selectedCallAudio = file;
  $("callAudioInfo").textContent = `녹음 파일 선택됨: ${file.name} (${Math.round(file.size / 1024)}KB)`;
});

$("btnCallSummary").addEventListener("click", async () => {
  if (!state.selectedCallAudio) {
    alert("먼저 상담 녹음 파일을 선택하세요.");
    return;
  }
  $("btnCallSummary").disabled = true;
  setStatus("상담 통화 QA 생성 중...");
  try {
    const settings = {
      agent_name: $("callAgentName").value.trim(),
      call_date_hint: $("callDateHint").value || "",
      include_summary: $("callIncludeSummary").checked,
      ai_provider: $("callAiProvider").value,
    };
    const payload = await postCallSummary(state.selectedCallAudio, settings);
    $("callSummaryMd").innerHTML = marked.parse(payload.markdown || "");
    $("callExtractJson").textContent = JSON.stringify(payload.extracted || {}, null, 2);
    $("callAudioInfo").textContent = `QA 생성 완료: ${state.selectedCallAudio.name}`;
    setStatus("상담 통화 QA 완료");
  } catch (error) {
    console.error(error);
    alert(String(error));
    setStatus("상담 통화 QA 실패", "error");
  } finally {
    $("btnCallSummary").disabled = false;
  }
});

$("btnRefreshMarket").addEventListener("click", () => fetchMarketOverview(true));

initOffcanvas();
updateRecorderUi();
updateModeUi();
fetchHealth();
fetchOllamaModels();
clearAll();
startWorldClock();
fetchMarketOverview();
