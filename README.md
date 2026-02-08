# korean-meeting-minutes-summarizer

로컬 **CPU-only** 환경에서 한국어 **회의록 문서 요약(긴 문서 포함)** 을 수행하는 예제 레포입니다.

- **Python(FastAPI) 추론 서버**: Hugging Face 모델로 요약/분류/임베딩 API 제공
- **Node.js(CommonJS) 클라이언트**: 서버 호출 예제(요약/분류/임베딩)

> 권장 사용: 회의록(긴 문서) 요약은 CPU에서 느릴 수 있으므로, 본 레포는 **청크(Map) → 요약(Reduce)** 방식으로 긴 문서를 처리합니다.

---

## 1) 요구 사항

- Python 3.10+ (권장)
- Node.js 18+ (권장: 내장 `fetch` 사용)

---

## 2) 빠른 시작

### 2-1. Python 서버 실행

```bash
sudo apt update
sudo apt install -y python3-venv
python3 -m venv .venv
```

```bash
cd server
source .venv/bin/activate
pip install -r requirements.txt

# (선택) 모델 교체
# export SUM_MODEL_ID=gogamza/kobart-summarization

uvicorn app:app --host 127.0.0.1 --port 8000
```

#### whl은 파이썬 패키지 배포 파일 형식인 “Wheel(휠)” 확장자예요. 쉽게 말해:
#### something-1.2.3-py3-none-any.whl 같은 파일은 미리 빌드(패키징)된 설치 파일

---

### Version Down - 4.49.0 에 맞춤
```
cd /home/AI-Meeting-Summarizer/server
source .venv/bin/activate

pip uninstall -y transformers
pip install "transformers==4.49.0" --upgrade
```
---
```
python3 -c "import transformers; print(transformers.__version__)"
```
---




서버가 뜨면:
- 헬스체크: http://127.0.0.1:8000/health
- API 문서(Swagger): http://127.0.0.1:8000/docs

### 2-2. Node.js 클라이언트 실행

```bash
cd client
npm i

# 샘플 회의록을 요약
npm run summarize:sample

# 분류(옵션)
npm run classify:sample

# 임베딩(옵션)
npm run embed:sample
```

```
curl -s -X POST "http://127.0.0.1:8000/report" \
  -H "Content-Type: application/json" \
  -d '{"text":"김도영: 그럼 성능 개선 제가 하겠습니다. 차주 수요일까지요.\n홍길동: 이슈는 CPU에서 요약이 느린 점입니다.\n이순신: 다음주 금요일에 임베딩 검색 논의하죠.","meeting_title":"주간 개발 회의","meeting_date_hint":"2026-02-08","include_summary":true}' | jq -r .markdown
```
---
```
cd /home/AI-Meeting-Summarizer/client
npm install axios
node scripts/report_to_md.js
```
---
```
root@DESKTOP-D6A344Q:/home/AI-Meeting-Summarizer# npm run report:sample

> korean-meeting-minutes-summarizer@1.0.0 report:sample
> cd client && npm run report:sample


> meeting-minutes-client@1.0.0 report:sample
> node scripts/report_to_md.js

Saved: /home/AI-Meeting-Summarizer/client/scripts/meeting_report.md
```

### Docker 기반으로
```
# 1) 필요 패키지
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 2) Docker 공식 GPG 키 등록
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 3) Docker 공식 저장소 추가 (Ubuntu codename 자동)
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4) compose plugin 설치
sudo apt-get update
sudo apt-get install -y docker-compose-plugin

```
---
```
docker compose version
```
---
```
# 멈춘 컨테이너/네트워크 정리
docker compose down

# 혹시 “남아있는 실패 컨테이너”까지 강제로 청소
docker rm -f ai-meeting-summarizer-api-1 ai-meeting-summarizer-web-1 2>/dev/null || true

# 다시 실행
docker compose up --build
```

---
```
sudo ss -ltnp | egrep ':8000|:3000'
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 웹 확인
- FE: http://localhost:3000
- API: http://localhost:8000/health

---

## 3) 긴 문서 요약 방식(중요)

회의록처럼 긴 문서는 모델 입력 길이를 초과하기 때문에:

1. 문서를 문단/줄 단위로 쪼갠 뒤
2. 토큰 길이 기준으로 청크를 구성(Map)
3. 각 청크를 요약하고
4. 청크 요약들을 다시 한번 요약(Reduce)

을 수행합니다.

`POST /summarize` 는 아래 응답을 제공합니다:
- `chunk_summaries`: 청크별 요약
- `final_summary`: 최종 요약(1~2차 요약 결과)

---

## 4) 환경 변수

서버(`server/.env.example` 참고):
- `SUM_MODEL_ID` (기본: `gogamza/kobart-summarization`)
- `CLS_MODEL_ID` (기본: `Seonghaa/korean-emotion-classifier-roberta`)
- `EMB_MODEL_ID` (기본: `upskyy/bge-m3-korean`)

추가 튜닝:
- `CHUNK_MAX_TOKENS` (기본 900)
- `REDUCE_MAX_TOKENS` (기본 1100)

---

## 5) 샘플

`samples/meeting_minutes_ko.txt` 를 수정해서 본인 회의록을 넣고 테스트하세요.

---

## 6) 라이선스

MIT (원하면 변경하세요)
