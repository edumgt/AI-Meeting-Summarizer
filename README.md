# AI Meeting Summarizer (KR)
 
브라우저에서 **회의를 녹음**하고, 서버에서 **mp3로 저장 + STT 전사 + 회의록 리포트 생성**까지 한 번에 처리하는 실무형 예제입니다. 회의록 외에도 **증권사 애널리스트 리포트 추천 추출/컨센서스 집계**, **AICC 콜센터 통화 QA 어시스턴트**, **글로벌 마켓(환율/세계 시간대/주요국 주가지수) 대시보드** 기능을 함께 제공합니다.

- **FE (정적 Static Web UI, FastAPI가 직접 서빙)**
  - 브라우저 `MediaRecorder` 기반 녹음
  - 오디오/PDF 업로드 및 전송
  - 전사 결과 + 마크다운 리포트 + 구조화 JSON 확인
  - 애널리스트 리포트 추출, 종목 컨센서스 조회, 상담 통화 QA 화면 포함
- **BE (FastAPI)**
  - 오디오 → mp3 변환(`ffmpeg`)
  - OpenAI 음성 전사(`gpt-4o-mini-transcribe` 기본)
  - 기존 AI 파이프라인(요약/추출/리포트) 연동
  - 애널리스트 리포트/콜 QA 이력을 Postgres(공유 인스턴스)에 저장

---
```
Error: HTTP 500: {"detail":"ffmpeg is not installed on server."}
```
```
sudo apt update
sudo apt install -y ffmpeg
```

## 1. 아키텍처

1) FE에서 녹음(webm) 또는 오디오 파일 업로드  
2) `POST /transcribe-and-report`로 전송  
3) BE에서 mp3 변환 후 서버 디렉터리에 저장  
4) OpenAI STT로 텍스트 전사  
5) 기존 `/report` 파이프라인으로 요약 + 의사결정/액션아이템 추출  
6) FE에서 결과 표시 및 mp3 다운로드 링크 제공

---

## 2. 요구사항

- Python 3.10+
- `ffmpeg` (서버 오디오 변환용)
- `OPENAI_API_KEY` 환경 변수
- (애널리스트/콜 QA 기능 사용 시) 접근 가능한 Postgres 인스턴스 — 없어도 회의록 핵심 기능은 정상 동작

---

## 3. 빠른 시작

### 3.1 서버 실행 (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

export OPENAI_API_KEY=""
export TRANSCRIBE_MODEL="gpt-4o-mini-transcribe"
export TRANSCRIBE_LANGUAGE="ko"
export MP3_OUTPUT_DIR="/tmp/ai_meeting_audio"
# 애널리스트/콜 QA 기능을 쓰려면 Postgres 접속 정보도 설정(없으면 해당 기능만 비활성화)
export POSTGRES_HOST="localhost"
export POSTGRES_PASSWORD="password"

uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir server
```

> `analyst.py`/`callcenter.py`/`pipeline.py`/`db.py`/`tickers.py`가 `server/` 안에서 서로를 flat 모듈(`import pipeline`, `import db` 등)로 참조하므로, 반드시 `--app-dir server`로 `server/`를 임포트 루트에 두고 `app:app`(점 표기 아님)으로 실행해야 합니다. Docker 이미지도 동일하게 `WORKDIR /app`에 `server/`의 내용을 그대로 복사한 뒤 `app:app`으로 실행합니다.

웹 UI는 별도 서버 없이 `http://localhost:8000` 에서 FastAPI가 직접 서빙합니다(`client/public`을 `/static`으로 복사해 마운트).

- Web: http://localhost:8000
- API docs: http://localhost:8000/docs

### 3.2 docker-compose 실행

```bash
docker compose up --build
```

`docker-compose.yml`은 `shared-net`이라는 외부 Docker 네트워크에 연결해, 그 네트워크에 이미 떠 있는 `postgres` 컨테이너를 재사용합니다. 네트워크가 없다면 먼저 `docker network create shared-net` 후 Postgres 컨테이너를 붙이거나, Postgres 없이 회의록 기능만 사용할 계획이면 `docker-compose.yml`에서 `shared-net` 관련 설정을 제거해도 됩니다.

---

## 4. 핵심 API

### 4.1 `POST /transcribe-and-report` (신규)

`multipart/form-data`로 오디오를 업로드해 전사 + 리포트를 한 번에 받습니다.

필드:
- `audio` (필수): 업로드 파일
- `meeting_title` (선택)
- `meeting_date_hint` (선택, `YYYY-MM-DD`)
- `include_summary` (선택, `true/false`)
- `language` (선택, 기본 `ko`)

응답:
- `transcript`: 전사 텍스트
- `markdown`: 정리 회의록
- `extracted`: 구조화 결과(JSON)
- `mp3_download_url`, `mp3_file_name`: 서버 저장 mp3 접근 정보

예시:

```bash
curl -X POST "http://localhost:8000/transcribe-and-report" \
  -F "audio=@./sample.webm" \
  -F "meeting_title=주간 개발 회의" \
  -F "meeting_date_hint=2026-02-08" \
  -F "include_summary=true"
```

### 4.2 `GET /audio/{file_name}` (신규)

서버에 저장된 mp3를 다운로드합니다.

### 4.3 애널리스트 리포트 (신규)

- `POST /analyst-report`: JSON `{ text, analyst_firm, report_date_hint }` → 투자의견/목표주가/근거 추출
- `POST /analyst-report/pdf`: PDF 업로드로 동일 추출 수행
- `GET /analyst-consensus/{ticker}`: 누적된 리포트로 종목별 컨센서스(의견 분포, 목표주가 평균/최소/최대) 조회

```bash
curl -X POST "http://localhost:8000/analyst-report" \
  -H "Content-Type: application/json" \
  -d '{"text": "삼성전자 매수 의견 유지, 목표주가 90,000원으로 상향. 투자포인트: 메모리 업황 개선", "analyst_firm": "한국증권"}'

curl "http://localhost:8000/analyst-consensus/005930"
```

### 4.4 상담 통화 QA / AICC (신규)

- `POST /call-summary`: 상담 녹음(multipart) 업로드 → 전사 → 문의유형/주문요청/컴플라이언스 체크/리스크 등급 산출
- `GET /call-summary/{call_id}`: 저장된 콜 QA 리포트 단건 조회

실제 착신(IVR)/CTI/실시간 스트리밍/상담원 자동 배분 기능은 포함하지 않습니다(녹음 사후 업로드 방식). 컴플라이언스 체크는 키워드 매칭 기반의 참고용 휴리스틱이며 법적 요건을 담보하지 않습니다.

```bash
curl -X POST "http://localhost:8000/call-summary" \
  -F "audio=@./call.webm" \
  -F "agent_name=홍길동" \
  -F "ai_provider=openai"
```

### 4.5 글로벌 마켓 (신규)

- `GET /market-overview` (`?refresh=true`로 캐시 무시하고 즉시 재조회): 원화(KRW) 기준 주요국 환율 + 주요국 주가지수(KOSPI/KOSDAQ 포함 8개)
- 환율은 [open.er-api.com](https://www.exchangerate-api.com/docs/free)(공식 무료 API), 주가지수는 Yahoo Finance 비공식 차트 엔드포인트를 사용하며 API 키가 필요 없습니다.
- 서버에서 5분 TTL로 캐시하며, 개별 통화/지수 조회 실패는 해당 항목만 `error` 필드로 표시되고 나머지 응답에는 영향을 주지 않습니다.
- 세계 시간대는 서버 호출 없이 브라우저 `Intl.DateTimeFormat`으로 계산됩니다(외부 API 상태와 무관하게 항상 정확).
- **주의**: 주가지수 데이터 소스(Yahoo 비공식 엔드포인트)는 문서화되지 않은 API로, 사전 통보 없이 응답 형식이 바뀌거나 접근이 막힐 수 있습니다. 실거래/투자 판단용이 아닌 참고용입니다.

```bash
curl "http://localhost:8000/market-overview"
curl "http://localhost:8000/market-overview?refresh=true"
```

---

## 5. 환경 변수

### 기존 NLP
- `SUM_MODEL_ID` (기본 `gogamza/kobart-summarization`)
- `CLS_MODEL_ID` (기본 `Seonghaa/korean-emotion-classifier-roberta`)
- `EMB_MODEL_ID` (기본 `upskyy/bge-m3-korean`)
- `CHUNK_MAX_TOKENS`, `REDUCE_MAX_TOKENS`

### 신규 음성 파이프라인
- `OPENAI_API_KEY` (필수)
- `TRANSCRIBE_MODEL` (기본 `gpt-4o-mini-transcribe`)
- `TRANSCRIBE_LANGUAGE` (기본 `ko`)
- `MP3_OUTPUT_DIR` (기본 `/tmp/ai_meeting_audio`)

### 애널리스트 리포트 / 콜 QA 저장 (Postgres, 신규)
- `POSTGRES_HOST` (기본 `postgres`) — docker-compose 사용 시 `shared-net`에 떠 있는 공유 Postgres 컨테이너 이름
- `POSTGRES_PORT` (기본 `5432`)
- `POSTGRES_USER` (기본 `postgres`)
- `POSTGRES_PASSWORD` (기본 `password`, `.env`로 오버라이드 권장)
- `POSTGRES_MAINTENANCE_DB` (기본 `postgres`) — 최초 기동 시 전용 DB 생성 여부 확인용
- `POSTGRES_DB` (기본 `meeting_agent`) — 이 앱 전용 DB. 여러 랩 프로젝트가 같은 Postgres 인스턴스를 공유해도 테이블이 섞이지 않도록 분리
- Postgres 연결에 실패해도 회의록 핵심 기능(요약/분류/임베딩)은 정상 동작하며, `/analyst-report*`/`/call-summary*`만 영향을 받습니다.

---

## 6. 운영 팁 (실무)

- mp3 저장소(`MP3_OUTPUT_DIR`)를 영속 볼륨으로 마운트하세요.
- 파일 보관 정책(예: 7일 후 삭제) 배치 작업을 추가하세요.
- CORS 허용 도메인을 운영 도메인으로 제한하세요.
- STT 비용 관리를 위해 파일 길이 제한 및 요청 인증(JWT/API Key)을 붙이세요.
- 콜 QA의 컴플라이언스 키워드 목록(`server/callcenter.py`의 `BANNED_PHRASES`/`REQUIRED_DISCLOSURES`)은 참고용 시드입니다. 운영 적용 전 컴플라이언스 팀 검토를 받으세요.
- 종목 매핑(`server/data/tickers_kr.json`)은 정적 시드입니다. 실제 서비스에서는 KRX 등 공식 종목마스터로 교체하세요.

---

## 7. 레거시 기능

기존 텍스트 기반 엔드포인트도 그대로 사용 가능합니다.

- `POST /summarize`
- `POST /extract`
- `POST /report`
- `POST /classify`
- `POST /embed`

