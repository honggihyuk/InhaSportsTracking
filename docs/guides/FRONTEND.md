# 웹 프론트엔드 가이드

React 19 + Vite + Three.js(React Three Fiber) 기반입니다. 실행 방법은 [설치 · 서버 실행 가이드](./SETUP.md#3-서버-실행).

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173 (백엔드 http://localhost:8000 필요)
npm run build     # dist/ 정적 빌드
npm run lint
```

## 구성

```
frontend/
├── index.html
├── vite.config.js
├── public/                      # 파비콘·아이콘
└── src/
    ├── main.jsx                 # 진입점 (Pretendard 서체를 npm 패키지로 번들)
    ├── App.jsx                  # 화면 구성 · 영상 ↔ 분석 ↔ 3D 데이터 흐름 · 분석/경기 지표 카드
    ├── analysis.js              # 보정 기준점, 분석 프레임 → 3D 데이터, 경기장 라인 투영
    ├── index.css                # 디자인 토큰 · 스타일
    └── components/
        ├── api.js               # REST / WebSocket 클라이언트 (API_BASE_URL)
        ├── VideoOverlay.jsx     # 영상 위 탐지 박스 · 보정 라인 · 보정점 canvas
        └── CalibrationPanel.jsx # 수동 보정용 경기장 도면
```

## 화면 기능

| 영역 | 기능 |
|---|---|
| 레이아웃 | 분할 / 경기 영상 / 3D 트윈 |
| 경기 영상 | 업로드·라이브러리, 탐지 박스(팀 색·`#ID`·`BALL`, 보간 박스는 점선), **보정된 경기장 라인 표시** |
| 3D 트윈 | 자유 시점 / 탑다운 / 선수 시점, 영상과 프레임 단위 동기화(`requestVideoFrameCallback`), 화면이 바뀔 때만 렌더링 |
| 영상 분석 카드 | 모드 선택(고정밀 / 빠른 미리보기), 진행 단계·취소, 자동 보정 · 기준점 직접 지정, 키프레임(자동·수동) 수 |
| 경기 지표 카드 | 점유율·이동 거리·스프린트·패스, 선수 표, 이벤트 목록(클릭 시 해당 장면으로 이동) |

## 백엔드 주소 바꾸기

`src/components/api.js` 의 `API_BASE_URL` (기본 `http://localhost:8000`). WebSocket 주소도 여기서 만들어집니다.

## 외부 요청 없음

글꼴(Pretendard)은 npm 패키지로 번들되고, 3D 선수 번호 라벨은 canvas 로 그립니다. CDN 이 막힌 환경(교내망·오프라인)에서도 동작합니다.
