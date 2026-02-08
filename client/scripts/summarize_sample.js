const { postJson, readSample, BASE_URL } = require("./_http");

(async () => {
  const text = readSample("meeting_minutes_ko.txt");

  const payload = {
    text,
    // 요약 길이(회의록은 너무 짧으면 정보 손실)
    max_length: Number(process.env.MAX_LENGTH || 180),
    min_length: Number(process.env.MIN_LENGTH || 50),
    do_sample: false,
    // 긴 문서 처리(토큰 기준) - CPU에서 속도/품질 트레이드오프
    chunk_max_tokens: Number(process.env.CHUNK_MAX_TOKENS || 900),
    reduce_max_tokens: Number(process.env.REDUCE_MAX_TOKENS || 1100),
  };

  console.log(`[POST] ${BASE_URL}/summarize`);
  const out = await postJson("/summarize", payload);

  console.log("\n=== Final Summary ===\n");
  console.log(out.final_summary);

  console.log("\n=== Chunk Summaries (debug) ===\n");
  out.chunk_summaries.forEach((s, i) => {
    console.log(`#${i + 1}: ${s}`);
  });

  console.log("\nmeta:", out.meta);
})().catch((e) => {
  console.error("summarize failed:", e.message);
  if (e.response) console.error("response:", e.response);
  process.exit(1);
});
