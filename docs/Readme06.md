# Readme06 - API 설계 및 인터페이스 구성요소

## 1. API 우선 설계 원칙
본 프로젝트는 기능 확장보다 먼저 API 계약을 고정하는 방식을 따릅니다.

- 엔드포인트 목적이 명확
- JSON 스키마 기반 자동 문서화
- 클라이언트/서버 분리 개발 가능

## 2. 주요 인터페이스 상세

### 2-1. `/summarize`
- 입력: 회의록 원문
- 출력: `chunk_summaries`, `final_summary`
- 특징: 긴 문서 처리 파이프라인 포함

### 2-2. `/classify`
- 입력: 문장 또는 문서
- 출력: label, score
- 특징: 회의 분위기/주제 분류 시 사용

### 2-3. `/embed`
- 입력: 텍스트
- 출력: embedding(vector)
- 특징: 후속 유사도 검색 파이프라인 기반

### 2-4. `/report`
- 입력: 텍스트 + 메타데이터(회의 제목, 날짜 힌트 등)
- 출력: markdown 형태 리포트
- 특징: 실무 전달용 산출물을 빠르게 생성

### 2-5. `/analyst-report`, `/analyst-report/pdf`
- 입력: 애널리스트 리포트 텍스트 또는 PDF + 증권사/발행일(선택)
- 출력: 종목별 `AnalystRecommendation`(투자의견/목표주가/근거) 리스트
- 특징: 정규식/키워드 기반 추출(회의록 결정사항 추출과 동일한 방식), 결과는 Postgres에 누적 저장

### 2-6. `/analyst-consensus/{ticker}`
- 입력: 종목코드(예: `005930`)
- 출력: 리포트 건수, 의견 분포, 목표주가 평균/최소/최대, 최근 리포트 목록
- 특징: `/analyst-report*`로 누적된 데이터를 집계

### 2-7. `/call-summary`, `/call-summary/{call_id}`
- 입력: 상담 녹음 파일 + 상담원/상담일(선택)
- 출력: 전사문, 문의유형, 주문요청 여부, 컴플라이언스 플래그, 리스크 등급, markdown QA 리포트
- 특징: 컴플라이언스 체크는 키워드 매칭 기반 참고용 휴리스틱(법적 요건 보증 아님). 실제 텔레포니/IVR 연동은 범위 밖

### 2-8. `/market-overview`
- 입력: 없음(쿼리 `refresh=true`로 캐시 무시 가능)
- 출력: 원화 기준 주요국 환율(`fx`), 주요국 주가지수(`indices`), `updated_at`
- 특징: 환율은 open.er-api.com(공식 무료 API), 지수는 Yahoo Finance 비공식 차트 엔드포인트 사용(키 불필요, 5분 TTL 캐시). 개별 항목 실패는 `error` 필드로만 표시되고 전체 응답은 유지됨. 세계 시간대는 서버가 아닌 프론트엔드(`Intl.DateTimeFormat`)에서 계산

## 3. API 문서화
- FastAPI Swagger(`/docs`) 자동 제공
- 개발/QA/기획자가 동일 문서를 참조 가능
- 변경 시 문서가 자동 동기화되어 협업 비용 감소

## 4. 권장 API 거버넌스
- 버전 경로 도입(`/v1/...`)
- 응답 표준화(성공/실패 envelope)
- 에러 코드 체계 정립(Validation, ModelError, Timeout)
