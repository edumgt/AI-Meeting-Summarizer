# Readme05 - 프론트엔드/클라이언트 스택(Node.js + 스크립트)

## 1. 클라이언트 역할
이 프로젝트의 Node.js 계층은 전통적인 SPA 프론트엔드라기보다, API를 쉽게 검증하고 자동화하는 **실행 클라이언트 레이어**에 가깝습니다.

- 샘플 입력을 서버로 전송
- 결과를 콘솔 또는 파일로 출력
- 기능별(요약/분류/임베딩/리포트) 시나리오 분리

## 2. 핵심 구성
- `client/scripts/summarize_sample.js`
- `client/scripts/classify_sample.js`
- `client/scripts/embed_sample.js`
- `client/scripts/report_to_md.js`
- 공통 HTTP 유틸: `client/scripts/_http.js`

## 3. CommonJS 기반 장점
- 러닝커브가 낮고 빠르게 실행 가능
- 단일 파일 스크립트 운영이 쉬움
- CI/CD에서 작업 스텝으로 넣기 간단

## 4. 운영 활용 예시
- 배포 후 API 회귀 테스트
- 샘플 데이터 주기 실행으로 모델 변경 영향도 확인
- 결과 Markdown 자동 생성 후 사내 위키 업로드

## 5. 확장 방향
- TypeScript로 전환해 타입 안정성 강화
- CLI 옵션화(`--input`, `--output`, `--model`)로 재사용성 향상
- React/Vue UI 연동 시 현재 스크립트를 서비스 레이어로 재사용
