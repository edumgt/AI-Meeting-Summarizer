const fs = require("fs");
const path = require("path");

const BASE_URL = process.env.BASE_URL || "http://127.0.0.1:8000";

async function postJson(urlPath, body) {
  const res = await fetch(`${BASE_URL}${urlPath}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = text; }

  if (!res.ok) {
    const err = new Error(`HTTP ${res.status} ${res.statusText}`);
    err.response = data;
    throw err;
  }

  return data;
}

function readSample(name) {
  const p = path.join(__dirname, "..", "..", "samples", name);
  return fs.readFileSync(p, "utf-8");
}

module.exports = { postJson, readSample, BASE_URL };
