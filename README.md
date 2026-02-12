# AI Meeting Summarizer (KR)

브라우저에서 **회의를 녹음**하고, 서버에서 **mp3로 저장 + STT 전사 + 회의록 리포트 생성**까지 한 번에 처리하는 실무형 예제입니다.

- **FE (Node + Static Web UI)**
  - 브라우저 `MediaRecorder` 기반 녹음
  - 오디오 업로드/녹음 데이터 전송
  - 전사 결과 + 마크다운 리포트 + 구조화 JSON 확인
- **BE (FastAPI)**
  - 오디오 → mp3 변환(`ffmpeg`)
  - OpenAI 음성 전사(`gpt-4o-mini-transcribe` 기본)
  - 기존 AI 파이프라인(요약/추출/리포트) 연동

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
- Node.js 18+
- `ffmpeg` (서버 오디오 변환용)
- `OPENAI_API_KEY` 환경 변수

---

## 3. 빠른 시작

### 3.1 서버 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

export OPENAI_API_KEY=""
export TRANSCRIBE_MODEL="gpt-4o-mini-transcribe"
export TRANSCRIBE_LANGUAGE="ko"
export MP3_OUTPUT_DIR="/tmp/ai_meeting_audio"

uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3.2 웹 UI 실행

```bash
cd client
npm install
API_BASE=http://localhost:8000 npm start
```

- Web: http://localhost:3000
- API docs: http://localhost:8000/docs

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

---

## 6. 운영 팁 (실무)

- mp3 저장소(`MP3_OUTPUT_DIR`)를 영속 볼륨으로 마운트하세요.
- 파일 보관 정책(예: 7일 후 삭제) 배치 작업을 추가하세요.
- CORS 허용 도메인을 운영 도메인으로 제한하세요.
- STT 비용 관리를 위해 파일 길이 제한 및 요청 인증(JWT/API Key)을 붙이세요.

---

## 7. 레거시 기능

기존 텍스트 기반 엔드포인트도 그대로 사용 가능합니다.

- `POST /summarize`
- `POST /extract`
- `POST /report`
- `POST /classify`
- `POST /embed`

