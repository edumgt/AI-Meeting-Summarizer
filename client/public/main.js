import { marked } from "https://cdn.jsdelivr.net/npm/marked@12.0.2/lib/marked.esm.js";

const API_BASE = window.API_BASE || "http://localhost:8000";

const $ = (id) => document.getElementById(id);
$("apiBase").textContent = API_BASE;

let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordTimer = null;
let recordStartedAt = null;

function setStatus(msg) {
  $("status").textContent = msg ? `· ${msg}` : "";
}

function formatDuration(ms) {
  const s = Math.floor(ms / 1000);
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function updateRecorderUi() {
  const recording = !!mediaRecorder && mediaRecorder.state === "recording";
  $("btnStartRecording").disabled = recording;
  $("btnStopRecording").disabled = !recording;
  $("btnUploadAudio").disabled = recording;
  $("btnTranscribe").disabled = recording;

  if (!recording) {
    $("recordTime").textContent = "00:00";
  }
}

function stopStream() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function buildRecordedFile() {
  if (!recordedChunks.length) return null;
  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  return new File([blob], `meeting-${Date.now()}.webm`, { type: blob.type });
}

function renderReport(payload) {
  $("reportMd").innerHTML = marked.parse(payload.markdown);
  $("extractJson").textContent = JSON.stringify(payload.extracted, null, 2);
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

async function postAudioAndTranscribe(audioFile) {
  const fd = new FormData();
  fd.append("audio", audioFile, audioFile.name || "meeting.webm");
  fd.append("meeting_title", $("meetingTitle").value || "회의록");
  fd.append("meeting_date_hint", $("dateHint").value || "");
  fd.append("include_summary", String($("includeSummary").checked));

  const res = await fetch(`${API_BASE}/transcribe-and-report`, {
    method: "POST",
    body: fd
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

$("btnClear").addEventListener("click", () => {
  $("inputText").value = "";
  $("audioFile").value = "";
  $("reportMd").innerHTML = "(아직 없음)";
  $("extractJson").textContent = "{}";
  $("transcriptText").textContent = "(아직 없음)";
  $("audioDownload").innerHTML = "";
  recordedChunks = [];
  setStatus("");
  updateRecorderUi();
});

$("btnReport").addEventListener("click", async () => {
  const text = $("inputText").value.trim();
  if (!text) return alert("작성 회의록을 입력하세요.");

  $("btnReport").disabled = true;
  setStatus("텍스트 보고서 생성 중...");

  try {
    const payload = await postJson(`${API_BASE}/report`, {
      text,
      meeting_title: $("meetingTitle").value || "회의록",
      meeting_date_hint: $("dateHint").value || null,
      include_summary: $("includeSummary").checked
    });

    renderReport(payload);
    $("transcriptText").textContent = "(텍스트 입력 모드 - 전사 없음)";
    $("audioDownload").innerHTML = "";
    setStatus("완료");
  } catch (e) {
    console.error(e);
    alert(String(e));
    setStatus("실패");
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
    recordedChunks = [];
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(mediaStream);

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) recordedChunks.push(event.data);
    });

    mediaRecorder.addEventListener("stop", () => {
      stopStream();
      if (recordTimer) clearInterval(recordTimer);
      updateRecorderUi();
      const file = buildRecordedFile();
      if (file) {
        $("recordInfo").textContent = `녹음 파일 준비됨: ${file.name} (${Math.round(file.size / 1024)}KB)`;
      }
    });

    mediaRecorder.start(1000);
    recordStartedAt = Date.now();
    $("recordInfo").textContent = "녹음 중...";
    recordTimer = setInterval(() => {
      $("recordTime").textContent = formatDuration(Date.now() - recordStartedAt);
    }, 500);
    updateRecorderUi();
  } catch (e) {
    console.error(e);
    alert(`녹음 시작 실패: ${String(e)}`);
    stopStream();
    if (recordTimer) clearInterval(recordTimer);
    updateRecorderUi();
  }
});

$("btnStopRecording").addEventListener("click", () => {
  if (!mediaRecorder || mediaRecorder.state !== "recording") return;
  mediaRecorder.stop();
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
  setStatus("음성 전사 + 보고서 생성 중...");

  try {
    const payload = await postAudioAndTranscribe(sourceFile);
    $("inputText").value = payload.transcript;
    $("transcriptText").textContent = payload.transcript || "(전사 결과 없음)";
    renderReport(payload);

    const link = document.createElement("a");
    link.href = `${API_BASE}${payload.mp3_download_url}`;
    link.textContent = `서버 저장 mp3 다운로드 (${payload.mp3_file_name})`;
    link.target = "_blank";
    link.rel = "noreferrer";
    $("audioDownload").innerHTML = "";
    $("audioDownload").appendChild(link);

    setStatus("완료");
  } catch (e) {
    console.error(e);
    alert(String(e));
    setStatus("실패");
  } finally {
    $("btnTranscribe").disabled = false;
  }
});

$("btnUploadAudio").addEventListener("click", () => {
  $("audioFile").click();
});

$("audioFile").addEventListener("change", () => {
  const f = $("audioFile").files?.[0];
  if (!f) return;
  $("recordInfo").textContent = `업로드 파일 선택됨: ${f.name} (${Math.round(f.size / 1024)}KB)`;
});

updateRecorderUi();
