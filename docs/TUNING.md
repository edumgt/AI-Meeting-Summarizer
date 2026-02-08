# Tuning (CPU-only 긴 문서 요약)

## 속도/품질 트레이드오프

- `chunk_max_tokens`를 줄이면:
  - ✅ 각 청크 요약이 빨라짐
  - ❌ 청크 개수가 늘어 전체 시간이 늘 수도 있음

- `max_length/min_length`를 줄이면:
  - ✅ 요약 생성 시간이 줄어듦
  - ❌ 핵심 정보가 빠질 수 있음

## 추천 시작값

- 회의록(1~3페이지):
  - `chunk_max_tokens=900`
  - `reduce_max_tokens=1100`
  - `max_length=180`, `min_length=50`

- 회의록(매우 김/수십 페이지):
  - `chunk_max_tokens=700`
  - `reduce_max_tokens=900`
  - `max_length=200`, `min_length=60`
  - 그리고 **사전 전처리(의제/결정/액션 중심으로 문단 필터링)** 를 강하게 권장

## 회의록 구조화(다음 단계)

단순 요약을 넘어 아래 형태로 구조화하려면:

- [결정사항] / [이슈] / [Action Items] 같은 섹션을 정규식으로 먼저 추출
- 섹션별로 요약 API를 호출해 섹션 요약을 만들고
- 최종 응답 JSON을 구조화

이 레포는 우선 기본 요약 파이프라인부터 제공합니다.
