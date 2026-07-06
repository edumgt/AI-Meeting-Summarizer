# Readme02 - 백엔드 기술 스택(FastAPI + Python)

## 1. FastAPI 선택 이유
FastAPI는 Python 기반 고성능 웹 프레임워크로, AI 추론 API 서버에 적합합니다.

- 타입 힌트 기반 자동 검증(Pydantic)
- OpenAPI/Swagger 문서 자동 생성
- 비동기 엔드포인트 지원
- 추론 API 프로토타이핑 속도가 빠름

## 2. 서버 핵심 구성요소

### 2-1. 엔드포인트

**회의록**
- `GET /health`: 서버 상태 확인
- `POST /summarize`: 긴 회의록 요약(Map-Reduce)
- `POST /classify`: 문장/문서 분류
- `POST /embed`: 텍스트 임베딩 벡터 생성
- `POST /extract` / `POST /report`: 결정사항/액션아이템 추출 및 종합 리포트 생성
- `POST /transcribe-and-report`: 오디오 업로드 → STT 전사 → 회의록 리포트
- `POST /pdf-report`: PDF 업로드 → 텍스트 추출 → 회의록 리포트
- `GET /audio/{file_name}`: 저장된 mp3 다운로드

**애널리스트 리포트** (`server/analyst.py`)
- `POST /analyst-report`: 텍스트에서 투자의견/목표주가/근거 추출
- `POST /analyst-report/pdf`: PDF 리포트 업로드 버전
- `GET /analyst-consensus/{ticker}`: 종목별 컨센서스(의견 분포, 목표주가 평균/최소/최대) 조회

**콜센터 QA** (`server/callcenter.py`)
- `POST /call-summary`: 상담 녹음 업로드 → 전사 → 문의유형/주문요청/컴플라이언스 체크
- `GET /call-summary/{call_id}`: 저장된 콜 QA 리포트 단건 조회

**글로벌 마켓** (`server/market.py`, `server/market_data.py`)
- `GET /market-overview`: 원화 기준 주요국 환율 + 주요국 주가지수(무료 공개 API, 5분 TTL 캐시, 키 불필요)

### 2-2. 요청/응답 스키마
- Pydantic 모델로 필수/선택 필드 검증
- 입력 포맷 오류를 422로 명확히 반환
- 응답 JSON 구조가 고정되어 클라이언트 자동화에 유리

## 3. Python 런타임 및 패키지
- Python 3.10+ 권장
- 주요 패키지: `fastapi`, `uvicorn`, `transformers`, `torch`, `sentence-transformers`, `openai`, `pypdf`, `psycopg2-binary`
- 설치는 `server/requirements.txt` 중심으로 관리
- 모듈 구성: `app.py`(라우팅/회의록 도메인), `pipeline.py`(STT/요약/PDF 공용 헬퍼), `db.py`(Postgres), `analyst.py`/`callcenter.py`(신규 도메인 라우터), `tickers.py`(종목명 매핑)

## 4. 운영 관점 체크포인트
- 서버 시작: `uvicorn app:app --host 127.0.0.1 --port 8000`
- 로컬 테스트: `/docs`에서 Swagger 기반 API 호출
- 버전 고정: 모델/라이브러리 호환성 이슈를 피하기 위해 특정 버전 고정(예: transformers)

## 5. 권장 개선사항
- 프로덕션용 ASGI 서버 튜닝(worker 수, timeout)
- 요청량 증가 시 model warm-up 및 캐시 전략 도입
- 로깅 구조화(JSON logging) 및 추적 ID 부여
