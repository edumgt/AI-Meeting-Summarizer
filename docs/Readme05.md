# Readme05 - 프론트엔드/클라이언트 스택(정적 HTML/JS)

> 과거에는 별도 Node.js 클라이언트(`client/server.js` 등)가 있었으나 제거되었습니다. 현재는 빌드 도구 없는 단일 정적 페이지가 FastAPI 서버에 의해 직접 서빙됩니다.

## 1. 클라이언트 역할
`client/public`은 전통적인 SPA 프레임워크 없이 순수 HTML/CSS/JS로 작성된 단일 페이지 워크스페이스입니다.

- 오프캔버스 메뉴로 기능별(회의록/애널리스트 리포트/콜 QA) 모듈 전환
- 텍스트 입력, 브라우저 녹음(`MediaRecorder`), 오디오/PDF 업로드를 한 화면에서 처리
- 결과(markdown, 전사문, 구조화 JSON)를 같은 화면에서 바로 확인

## 2. 핵심 구성
- `client/public/index.html`: 레이아웃과 각 기능 카드(모듈) 마크업
- `client/public/main.js`: DOM 이벤트 바인딩, `fetch` 기반 API 호출, 결과 렌더링
- `marked`(CDN)로 서버가 반환한 markdown을 HTML로 렌더링
- `/config.js`(서버 제공): 배포 환경의 `API_BASE`를 `window.API_BASE`로 주입

## 3. 서빙 방식
- `server/Dockerfile`이 `client/public`을 `server/static`으로 복사
- `app.py`가 `StaticFiles(directory="static", html=True)`를 `/`에 마운트해 같은 FastAPI 프로세스가 API와 정적 파일을 함께 제공
- 별도 프론트엔드 컨테이너/포트가 없어 배포 단위가 단순함(이미지 1개, 포트 1개)

## 4. 신규 기능 UI
- **애널리스트 리포트** 모듈: 텍스트/PDF 입력 → 추천 추출 결과 표시, 종목코드로 컨센서스 조회
- **상담 통화 QA** 모듈: 상담 녹음 업로드 → markdown QA 리포트 + 구조화 JSON(문의유형/리스크등급/컴플라이언스 플래그) 표시

## 5. 확장 방향
- 정적 페이지가 커지면 컴포넌트 단위 분리(예: Vite + 바닐라 JS 모듈) 고려
- 오디오 녹음 UI를 콜 QA 모듈에도 재사용(현재는 콜 QA가 파일 업로드만 지원)
- 종목 자동완성 UI(현재 `tickers_kr.json` 정적 사전 기반)
