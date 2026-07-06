# Readme07 - 컨테이너/배포 스택(Docker & Docker Compose)

## 1. Docker 도입 목적
- 개발 환경 차이를 최소화
- Python/Node 런타임 버전 충돌 방지
- 재현 가능한 배포 단위 제공

## 2. 구성 파일
- `server/Dockerfile`: FastAPI 서버 이미지 정의(`client/public`을 정적 파일로 함께 COPY)
- `docker-compose.yml`: `api` 서비스 실행(별도 프론트엔드 컨테이너 없음, FastAPI가 정적 파일도 함께 서빙)

## 3. Compose 기반 실행 구조
- API 서비스(포트 8000) 1개만 존재 — 정적 프론트엔드도 같은 컨테이너에서 서빙
- `hf-cache` 볼륨: HuggingFace 모델 캐시 영속화
- `shared-net`(외부 네트워크, `external: true`): 애널리스트/콜 QA 이력을 저장하는 공유 Postgres 컨테이너(`postgres`)에 접근하기 위해 연결. 이 네트워크와 postgres 컨테이너는 이 레포 밖에서 이미 떠 있어야 하며(`docker network ls`로 확인), 없으면 `docker network create shared-net` 후 postgres 컨테이너를 붙여야 함
- Postgres 접속 정보(`POSTGRES_HOST/PORT/USER/PASSWORD/DB`)는 `environment`에서 주입되며, 앱이 시작 시 전용 DB(`meeting_agent`)와 테이블을 자동 생성

## 4. 운영 체크리스트
- `docker compose up --build`로 초기 빌드 + 실행
- `docker compose down`으로 정리
- 포트 충돌 점검(`8000`)
- Postgres 연결 실패 시 회의록 핵심 기능은 정상 동작하되, `/analyst-report*`/`/call-summary*`는 500 응답 — 로그에서 `Postgres 초기화 실패` 경고 확인
- 실패 컨테이너 정리 후 재기동

## 5. 고도화 방향
- 멀티스테이지 빌드로 이미지 경량화
- 헬스체크 추가로 자동 복구 안정화
- 환경별 compose override(dev/stage/prod) 분리
