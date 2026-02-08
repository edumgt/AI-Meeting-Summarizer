const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// 정적 파일
app.use(express.static(path.join(__dirname, "public")));

// API_BASE를 브라우저에서 읽을 수 있게 주입
app.get("/config.js", (req, res) => {
  const api = process.env.API_BASE || "http://localhost:8000";
  res.type("application/javascript").send(`window.API_BASE=${JSON.stringify(api)};`);
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Web UI running on http://0.0.0.0:${PORT}`);
});
