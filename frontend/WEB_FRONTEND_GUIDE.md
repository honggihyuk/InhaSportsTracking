# ⚽ Soccer 3D Digital Twin - Web 프론트엔드 가이드

## 🎯 개요

React + Three.js 기반의 웹 프론트엔드로, 축구 경기 영상을 3D 디지털 트윈으로 시각화합니다.

## 🏗️ 아키텍처

```
frontend/
├── src/
│   ├── App.jsx          # 메인 애플리케이션 (3D 렌더링 + UI)
│   ├── main.jsx         # 엔트리 포인트
│   └── components/      # 재사용 가능 컴포넌트 (예정)
│       ├── VideoPlayer.jsx
│       ├── Field3D.jsx
│       ├── PlayerList.jsx
│       └── StatsPanel.jsx
├── public/              # 정적 자산
└── package.json         # 의존성 관리
```

## 🚀 기능

### 1. **비디오 업로드 및 재생**
- 로컬 비디오 파일 업로드
- 재생/일시정지 컨트롤
- 타임라인 시크 바

### 2. **3D 뷰어 (Three.js)**
- 실시간 3D 축구장 렌더링
- 선수 및 공 위치 시각화
- 자유로운 카메라 조작

### 3. **다중 뷰 모드**
- **3D 뷰**: 입체적인 경기장 뷰
- **탑다운 뷰**: 전술 분석용 평면 뷰
- **선수 시점**: 특정 선수 1 인칭 뷰

### 4. **실시간 통계**
- 선수 수 및 팀별 인원
- 공 위치 좌표
- 선수별 이동 속도
- 프레임 및 시간 정보

## 📦 설치 및 실행

```bash
# 의존성 설치
cd frontend
npm install

# 개발 서버 시작
npm run dev

# 프로덕션 빌드
npm run build
```

서버가 시작되면 `http://localhost:5173` 에서 확인 가능합니다.

## 🔧 주요 컴포넌트

### App.jsx

```jsx
// 상태 관리
const [videoUrl, setVideoUrl] = useState(null);
const [players, setPlayers] = useState([]);
const [ball, setBall] = useState(null);
const [viewMode, setViewMode] = useState('3d');

// Three.js 3D 렌더링
useEffect(() => {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(...);
  const renderer = new THREE.WebGLRenderer(...);
  // ... 경기장, 선수, 공 렌더링
}, [players, ball, viewMode]);
```

## 🔌 백엔드 연동 (예정)

### WebSocket 실시간 데이터 수신

```javascript
// Python 파이프라인 → Frontend 데이터 스트리밍
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  setPlayers(data.players);
  setBall(data.ball);
};
```

### REST API 데이터 조회

```javascript
// 특정 프레임 데이터 조회
const response = await fetch('/api/frame/1234');
const data = await response.json();
```

## 🎨 커스터마이징

### 색상 테마 변경

```javascript
// App.jsx 내 색상 수정
const HOME_TEAM_COLOR = 0x0066cc;  // 홈 팀 (파랑)
const AWAY_TEAM_COLOR = 0xcc0000;  // 어웨이 팀 (빨강)
const FIELD_COLOR = 0x2d5a27;      // 잔디색
```

### 축구장 규격 조정

```javascript
const FIELD_LENGTH = 105;  // 미터
const FIELD_WIDTH = 68;    // 미터
```

## 📊 성능 최적화

- ** instanced Mesh**: 다수 선수 렌더링 최적화
- **LOD(Level of Detail)**: 거리에 따른 디테일 조절
- **Web Workers**: 무거운 계산 백그라운드 처리

## 🚧 향후 개발 계획

1. **SMPL 인체 모델 연동**: 실제 선수 포즈 복원
2. **3D Gaussian Splatting 렌더러**: 고품질 볼류메트릭 비디오
3. **실시간 데이터 파이프라인**: Python 백엔드와 WebSocket 연동
4. **전술 분석 도구**: 패스 라인, 히트맵, 오프사이드 라인
5. **모바일 대응**: 반응형 디자인 및 터치 컨트롤

## 📝 참고 자료

- [Three.js 문서](https://threejs.org/docs/)
- [React 공식 문서](https://react.dev/)
- [Vite 공식 문서](https://vitejs.dev/)
- [Roboflow Sports](https://github.com/roboflow/sports)
