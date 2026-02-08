const { postJson, BASE_URL } = require("./_http");

(async () => {
  const texts = [
    "다음 스프린트 목표를 정리했습니다.",
    "액션 아이템과 담당자를 확정했습니다.",
  ];

  console.log(`[POST] ${BASE_URL}/embed`);
  const out = await postJson("/embed", { texts, normalize: true });
  console.log("dim:", out.dim);
  console.log("first vector head:", out.vectors[0].slice(0, 8));
})().catch((e) => {
  console.error("embed failed:", e.message);
  if (e.response) console.error("response:", e.response);
  process.exit(1);
});
