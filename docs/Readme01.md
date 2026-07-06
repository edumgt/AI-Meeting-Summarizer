# Readme01 - 프로젝트 아키텍처 개요

## 1. 프로젝트 목적
AI-Meeting-Summarizer는 한국어 회의록 문서를 로컬 환경(CPU 중심)에서 처리하기 위해 설계된 경량 MLOps형 애플리케이션입니다. 핵심 목표는 다음과 같습니다.

- 긴 한국어 회의록을 안정적으로 요약하고, 오디오 녹음/PDF 자료까지 회의록으로 변환
- 회의 발화/문장을 감정 또는 의도 관점으로 분류
- 후속 검색/유사도 분석을 위한 임베딩 벡터 생성
- (신규) 증권사 애널리스트 리포트에서 투자의견/목표주가를 추출해 종목별 컨센서스로 집계
- (신규) 상담 통화 녹음을 문의유형/주문요청/컴플라이언스 관점에서 점검하는 콜 QA 보조
- (신규) 원화 기준 주요국 환율 · 세계 시간대 · 주요국 주가지수를 한 화면에서 보여주는 글로벌 마켓 대시보드

## 2. 상위 아키텍처
시스템은 **단일 FastAPI 서비스 + 정적 프론트엔드** 구조로 구성됩니다(과거 별도 Node.js 클라이언트는 제거되었고, `client/public`은 FastAPI가 직접 정적 파일로 서빙합니다).

1. **Python FastAPI 서버 (AI 추론 + API 계층)**
   - Hugging Face 기반 Transformer 모델 로딩(요약/분류/임베딩)
   - OpenAI STT/요약, Ollama 요약 연동
   - `/summarize`, `/classify`, `/embed`, `/report`, `/transcribe-and-report`, `/pdf-report` API 제공(회의록 도메인)
   - `/analyst-report`, `/analyst-report/pdf`, `/analyst-consensus/{ticker}` API 제공(애널리스트 리포트 도메인, `server/analyst.py`)
   - `/call-summary`, `/call-summary/{call_id}` API 제공(콜센터 QA 도메인, `server/callcenter.py`)
   - `/market-overview` API 제공(글로벌 마켓 도메인, `server/market.py` + `server/market_data.py`) — 무료 공개 API(환율: open.er-api.com, 지수: Yahoo Finance 비공식 엔드포인트)를 5분 TTL로 캐시해 조회
   - 긴 문서 처리 시 Map-Reduce 요약 파이프라인 적용
   - 공용 STT/요약/PDF 헬퍼는 `server/pipeline.py`로 분리되어 세 도메인이 함께 재사용

2. **정적 프론트엔드 (`client/public`)**
   - 빌드 도구 없는 순수 HTML/CSS/JS 단일 페이지
   - FastAPI가 같은 프로세스에서 정적 파일로 직접 서빙(별도 Node 서버 없음)
   - 오프캔버스 메뉴로 회의록/애널리스트 리포트/콜 QA 기능 전환

3. **Postgres (공유 인스턴스, `server/db.py`)**
   - 애널리스트 추천 이력(`analyst_recommendations`)과 콜 QA 이력(`call_summaries`)을 저장
   - 여러 랩 프로젝트가 공유하는 Postgres 컨테이너 위에 전용 DB(`meeting_agent`)만 새로 생성해 사용
   - 회의록 핵심 기능(요약/분류/임베딩)은 DB 없이도 동작 — Postgres 연결 실패 시 경고만 남기고 서버는 계속 기동

## 3. 데이터 흐름 요약

1. 사용자 입력(텍스트/오디오/PDF)
2. 브라우저 UI(또는 직접 API 호출)를 통해 FastAPI 호출
3. FastAPI가 요청 유형에 맞는 모델 추론/추출 로직 수행
4. (애널리스트/콜 QA는) 결과를 Postgres에 누적 저장
5. JSON 응답 및 markdown 리포트 반환

## 4. 설계 의도
- GPU 없는 환경에서도 동작하도록 모델/파이프라인을 보수적으로 구성
- 긴 텍스트 입력에서 생기는 토큰 길이 제한 문제를 서버에서 해결
- API 중심 구조로 향후 웹/모바일/사내 시스템 연동을 쉽게 확장
- 신규 도메인(애널리스트/콜센터)은 별도 모듈(`analyst.py`, `callcenter.py`)로 분리해 기존 회의록 로직과 결합도를 낮춤

## 5. 확장 포인트
- 실시간 시세/종목마스터 API 연동(현재 `server/data/tickers_kr.json`은 정적 시드)
- 콜센터 텔레포니/IVR/CTI 연동(현재는 사후 업로드 방식만 지원, 실시간 착신/상담원 배분은 범위 밖)
- 인증/인가(JWT, OAuth2) 추가
- 모델 교체(대형 한국어 LLM 또는 사내 파인튜닝 모델)
- 비동기 처리 큐(Celery, Redis) 도입
