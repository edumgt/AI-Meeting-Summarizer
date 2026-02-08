import { marked } from "https://cdn.jsdelivr.net/npm/marked@12.0.2/lib/marked.esm.js";

const API_BASE = window.API_BASE || "http://localhost:8000";

const $ = (id) => document.getElementById(id);
$("apiBase").textContent = API_BASE;

function setStatus(msg) {
  $("status").textContent = msg ? `· ${msg}` : "";
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
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

$("btnClear").addEventListener("click", () => {
  $("inputText").value = "";
  $("reportMd").innerHTML = "(아직 없음)";
  $("extractJson").textContent = "{}";
  setStatus("");
});

$("btnReport").addEventListener("click", async () => {
  const text = $("inputText").value.trim();
  if (!text) return alert("작성 회의록을 입력하세요.");

  $("btnReport").disabled = true;
  setStatus("처리 중... (CPU에서는 시간이 걸릴 수 있어요)");

  try {
    const payload = await postJson(`${API_BASE}/report`, {
      text,
      meeting_title: $("meetingTitle").value || "회의록",
      meeting_date_hint: $("dateHint").value || null,
      include_summary: $("includeSummary").checked
    });

    // markdown 출력
    $("reportMd").innerHTML = marked.parse(payload.markdown);

    // extracted JSON
    $("extractJson").textContent = JSON.stringify(payload.extracted, null, 2);

    setStatus("완료");
  } catch (e) {
    console.error(e);
    alert(String(e));
    setStatus("실패");
  } finally {
    $("btnReport").disabled = false;
  }
});
