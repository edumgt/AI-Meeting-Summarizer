# Readme02 - 백엔드 기술 스택(FastAPI + Python)

## 1. FastAPI 선택 이유
FastAPI는 Python 기반 고성능 웹 프레임워크로, AI 추론 API 서버에 적합합니다.

- 타입 힌트 기반 자동 검증(Pydantic)
- OpenAPI/Swagger 문서 자동 생성
- 비동기 엔드포인트 지원
- 추론 API 프로토타이핑 속도가 빠름

## 2. 서버 핵심 구성요소

### 2-1. 엔드포인트
- `GET /health`: 서버 상태 확인
- `POST /summarize`: 긴 회의록 요약(Map-Reduce)
- `POST /classify`: 문장/문서 분류
- `POST /embed`: 텍스트 임베딩 벡터 생성
- `POST /report`: 요약 + 액션아이템 형태의 종합 리포트 생성

### 2-2. 요청/응답 스키마
- Pydantic 모델로 필수/선택 필드 검증
- 입력 포맷 오류를 422로 명확히 반환
- 응답 JSON 구조가 고정되어 클라이언트 자동화에 유리

## 3. Python 런타임 및 패키지
- Python 3.10+ 권장
- 주요 패키지: `fastapi`, `uvicorn`, `transformers`, `torch`, `sentence-transformers`
- 설치는 `server/requirements.txt` 중심으로 관리

## 4. 운영 관점 체크포인트
- 서버 시작: `uvicorn app:app --host 127.0.0.1 --port 8000`
- 로컬 테스트: `/docs`에서 Swagger 기반 API 호출
- 버전 고정: 모델/라이브러리 호환성 이슈를 피하기 위해 특정 버전 고정(예: transformers)

## 5. 권장 개선사항
- 프로덕션용 ASGI 서버 튜닝(worker 수, timeout)
- 요청량 증가 시 model warm-up 및 캐시 전략 도입
- 로깅 구조화(JSON logging) 및 추적 ID 부여
