const fs = require("fs");
const path = require("path");
const axios = require("axios");

const API = "http://127.0.0.1:8000";

(async () => {
  // 이 파일(client/scripts/report_to_md.js) 기준으로 경로 고정
  const filePath = path.resolve(__dirname, "../../samples/meeting_minutes_ko.txt");
  const text = fs.readFileSync(filePath, "utf-8");

  const res = await axios.post(`${API}/report`, {
    text,
    meeting_title: "회의록 리포트",
    meeting_date_hint: "2026-02-08",
    include_summary: true,
  });

  const outPath = path.resolve(__dirname, "meeting_report.md");
  fs.writeFileSync(outPath, res.data.markdown, "utf-8");
  console.log("Saved:", outPath);
})();
