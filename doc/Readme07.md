# Readme07 - 컨테이너/배포 스택(Docker & Docker Compose)

## 1. Docker 도입 목적
- 개발 환경 차이를 최소화
- Python/Node 런타임 버전 충돌 방지
- 재현 가능한 배포 단위 제공

## 2. 구성 파일
- `server/Dockerfile`: FastAPI 추론 서버 이미지 정의
- `client/Dockerfile`: 웹/클라이언트 런타임 이미지 정의
- `docker-compose.yml`: 다중 서비스 통합 실행

## 3. Compose 기반 실행 구조
일반적으로 다음 서비스를 포함합니다.

1. API 서비스(포트 8000)
2. Web/Client 서비스(포트 3000)

서비스 간 네트워크가 자동 구성되므로 로컬 개발에서 인프라 설정 부담이 줄어듭니다.

## 4. 운영 체크리스트
- `docker compose up --build`로 초기 빌드 + 실행
- `docker compose down`으로 정리
- 포트 충돌 점검(`8000`, `3000`)
- 실패 컨테이너 정리 후 재기동

## 5. 고도화 방향
- 멀티스테이지 빌드로 이미지 경량화
- 헬스체크 추가로 자동 복구 안정화
- 환경별 compose override(dev/stage/prod) 분리
