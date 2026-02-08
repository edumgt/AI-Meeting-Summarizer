const { postJson, BASE_URL } = require("./_http");

(async () => {
  const text = process.env.TEXT || "오늘 회의는 생산적이었고, 다음 일정이 명확해져서 좋았습니다.";
  console.log(`[POST] ${BASE_URL}/classify`);
  const out = await postJson("/classify", { text });
  console.dir(out, { depth: null });
})().catch((e) => {
  console.error("classify failed:", e.message);
  if (e.response) console.error("response:", e.response);
  process.exit(1);
});
