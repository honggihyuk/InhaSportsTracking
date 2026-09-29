# ⚽ Soccer 3D Digital Twin

**축구 영상 기반 3D 디지털 트윈 분석 플랫폼**

축구 경기 영상에서 선수와 공을 탐지·추적하고, 경기장 좌표로 변환해 3D 디지털 트윈으로 재현합니다. 웹에서 경기 영상과 3D 트윈을 같은 타임라인으로 보며 자유로운 시점에서 경기를 분석할 수 있습니다.

> 📘 사용 기술, 구현된 기능, 향후 계획의 상세 설명은 **[docs/PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md)** 를 참고하세요.

---

## 🎯 프로젝트 개요

### 핵심 기술 스택

- **Object Detection**: YOLO11 + Roboflow 축구 전문 모델
- **Tracking**: BoT-SORT + 카메라 움직임 보정 (boxmot), 공 전용 BallTracker
- **Camera Motion / Calibration**: 광류 기반 프레임 간 호모그래피 + 키프레임 경기장 보정 전파
- **Coordinate Mapping**: Homography (픽셀 → 경기장 미터 좌표)
- **Backend**: FastAPI + WebSocket (30 FPS 트래킹 스트림)
- **Frontend**: React 19 + Vite + Three.js (React Three Fiber)
- **3D Gaussian Splatting**: 표준 공간 · 변형 MLP 모듈 (렌더링 연동 예정)

### 주요 기능

1. **업로드 영상 분석**: 선수·공 탐지 → 관중 제거 → 추적(공은 항상 ID 0) → 유니폼 색 팀 분류 → 카메라 움직임 추정 (백그라운드 작업, 진행률 표시)
2. **탐지 박스 오버레이**: 경기 영상 위에 어떤 선수(`#ID`, 팀 색)와 공(`BALL`)을 인식했는지 프레임 단위로 표시
3. **경기장 보정**: 한 프레임에서 경기장 기준점 4 개 이상을 지정하면 카메라 팬·줌을 따라 전 프레임에 전파
4. **3D 디지털 트윈 연동**: 분석·보정된 선수·공 위치를 영상과 같은 프레임으로 3D 재현 (자유 시점 / 탑다운 / 선수 시점)
5. **통계**: 팀별 선수 수·평균/최고 속도, 볼 소유, 공 속도·위치

---

## 📁 프로젝트 구조

```
InhaSportsTracking/
├── backend/
│   └── server.py           # FastAPI: 업로드·영상 제공·WebSocket 스트림
├── tracking/               # 객체 탐지 및 추적 모듈
│   ├── detector.py         # YOLO 탐지, Detection 데이터클래스
│   ├── tracker.py          # make_tracker, SimpleTracker, BallTracker, SoccerTracker
│   ├── homography.py       # 호모그래피 변환
│   └── coordinate_mapper.py # 2D → 3D 좌표 매핑
├── pipeline/
│   └── main_pipeline.py    # 탐지 → 추적 → 좌표 변환 통합 파이프라인
├── gs_model/               # 3D Gaussian Splatting 모델 (선택, torch 필요)
│   ├── canonical_gs.py     # 표준 공간 가우시안
│   └── deformation_mlp.py  # 변형 MLP 네트워크
├── frontend/               # Web 프론트엔드 (React + Vite)
│   ├── index.html
│   └── src/
│       ├── App.jsx         # 영상 플레이어, 3D 트윈, 통계 UI
│       ├── index.css       # 디자인 토큰 및 스타일
│       └── components/api.js # REST / WebSocket 클라이언트
├── tests/
│   ├── test_core.py        # 탐지·추적·API 테스트 (pytest)
│   └── test_analysis.py    # 영상 분석·경기장 보정 테스트
├── docs/
│   └── PROJECT_OVERVIEW.md # 기술·기능 명세 및 로드맵
├── configs/
│   ├── model_config.yaml   # 모델 및 파이프라인 설정
│   └── field_dimensions.yaml # 축구장 규격
├── data/raw/               # 원본 비디오
├── models/                 # 사전 학습 YOLO11 모델 (yolo11s.pt, yolo11n.pt)
├── uploads/                # 업로드된 경기 영상 (git 제외, 서버 실행 시 자동 생성)
└── requirements.txt        # Python 의존성
```

---

## 🚀 빠른 시작

### 1. 백엔드 서버

```bash
# 의존성 설치 (가상환경 권장)
pip install -r requirements.txt

# API 서버 실행 → http://localhost:8000
python -m uvicorn backend.server:app --port 8000
```

### 2. 웹 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

브라우저에서 `http://localhost:5173` 에 접속합니다.

1. 라이브러리에서 **업로드**로 경기 영상을 올리고 선택합니다.
2. **영상 분석** 카드에서 **분석 시작** — 완료되면 영상 위에 탐지 박스가 표시됩니다.
3. **경기장 보정** — 오른쪽 도면에서 기준점(코너, 페널티박스 모서리, 센터서클과 하프라인 교점 등)을 선택하고 영상에서 같은 지점을 클릭합니다. 한 직선 위에 있지 않은 4 점 이상이면 저장할 수 있습니다.
4. 3D 트윈이 분석 데이터로 전환되어 영상과 함께 재생됩니다. 장면 전환 이후 구간은 그 구간에서 보정을 추가하세요.

> CPU 에서는 분석이 영상 길이의 수십 배 걸립니다 (5 코어 기준 약 1 초/프레임). GPU 가 있으면 `detection.device: cuda`, `analysis.stride: 1` 을 권장합니다.

> 브라우저가 재생할 수 있는 **H.264(MP4)** 또는 **VP9/AV1(WebM)** 영상을 사용하세요. 다른 코덱은 다음과 같이 변환합니다.
>
> ```bash
> ffmpeg -i input.mp4 -c:v libx264 -c:a aac -movflags +faststart output.mp4
> ```

### 3. 분석 파이프라인 (선택)

```bash
# Roboflow 전문 모델 다운로드 (.env 에 ROBOFLOW_API_KEY 필요)
python setup_roboflow.py

# 영상 분석 → data/processed/processed_data.pkl, trajectories.csv
python -m pipeline.main_pipeline --config configs/model_config.yaml --video <영상경로>
```

> CLI 실행은 현재 테스트용으로 **앞 50 프레임만** 처리합니다 (`main_pipeline.py` 의 `max_frames=50`).

### 4. 테스트

```bash
python -m pytest tests -q
```

YOLO·GPU 없이 합성 영상과 가짜 탐지기로 실행됩니다.

---

## 📊 사용된 데이터셋

### Roboflow 전문 모델

1. **[Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)** — 선수, 심판, 골키퍼 탐지
2. **[Football Ball Detection](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)** — 축구공 탐지
3. **[Football Field Detection](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)** — 경기장 라인·영역 탐지 (캘리브레이션용)

### 기본 모델 (COCO)

- **YOLO11s / YOLO11n**: 80 클래스 일반 객체 탐지 — person (Class 0), sports ball (Class 32)
- Roboflow 모델이 없으면 자동으로 대체 사용됩니다. 이 경우 공 모델이 사람도 공 후보로 탐지하므로 Roboflow 모델 사용을 권장합니다.

---

## 🔧 설정 가이드

### `.env` 파일

```bash
# Roboflow API 키 (Roboflow 모델 다운로드 시 필요)
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

API 키 발급: https://app.roboflow.com/settings/api

### `configs/model_config.yaml`

```yaml
detection:
  model_path: "models/yolo11n.pt"   # CPU 기본값, GPU 에서는 yolo11s 권장
  use_roboflow_models: false   # true 면 roboflow_models 경로의 전문 모델 사용
  roboflow_models:
    players: "data/roboflow_datasets/football-players-detection-3zvbc/weights/best.pt"
    ball: "data/roboflow_datasets/football-ball-detection-rejhg/weights/best.pt"
  confidence_threshold: 0.25
  device: "cpu"                # cpu 또는 cuda

analysis:
  stride: 3                    # N 프레임마다 탐지, 사이는 보간 (GPU 면 1)
  ball_imgsz: 960              # 공 모델 추론 해상도

tracking:
  type: "botsort"              # botsort(카메라 움직임 보정, 권장) | bytetrack | simple
  track_threshold: 0.3
  match_threshold: 0.8         # 매칭 비용(1 - IoU) 상한
  max_age: 30
```

---

## 📈 아키텍처

```
업로드 영상 ──▶ YOLO11 탐지 ──▶ 잔디 필터 ──▶ BoT-SORT + BallTracker ──▶ 유니폼 색 팀 분류
     │                                                │
     └──▶ 광류 카메라 움직임 (프레임 간 호모그래피) ──┤
                                                      ▼
                                   uploads/analysis/{영상}.json
                                                      │
     경기장 보정 (기준점 4+) ──▶ 키프레임 호모그래피 × 카메라 움직임 ──▶ 프레임별 경기장 좌표
                                                      │
                                                      ▼
            Web: 경기 영상 + 탐지 박스 오버레이  ⇄  3D 디지털 트윈 · 통계  (영상 프레임 단위 동기화)
```

---

## 🎨 Web 프론트엔드 기능

### 화면 구성

- **분할 / 경기 영상 / 3D 트윈** 레이아웃 전환
- **경기 영상 플레이어**: 업로드, 라이브러리 선택, 재생 불가 코덱 안내
- **영상 ↔ 3D 동기화**: 재생·정지·탐색 시 3D 트윈이 같은 프레임을 따라가며, 어긋나면 자동 재동기화

### 3D 카메라 시점

1. **자유 시점**: 궤도 회전·줌
2. **탑다운**: 전술 분석용 평면 뷰
3. **선수 시점**: 선수 위치에서 공을 바라보는 1인칭 시점

### 통계 패널

- 경기 시간·프레임, 팀별 선수 수·평균/최고 속도 비교, 볼 소유, 공 속도·위치

---

## 📝 문서

- **[docs/PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md)**: 사용 기술·구현 기능·API·로드맵 상세 명세
- **[docs/PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md](./docs/PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md)**: 영상 분석·경기장 보정·3D 트윈 연동 기술 상세와 실측 결과 (2부)
- **[docs/PROJECT_OVERVIEW_3_VLM.md](./docs/PROJECT_OVERVIEW_3_VLM.md)**: 컴퓨터 비전 고도화(RF-DETR·SAM3·SigLIP2·GLM-OCR)와 로컬 VLM 분석 층 조사·설계 (3부)
- **[TESTING_GUIDE.md](./TESTING_GUIDE.md)**: 테스트 및 사용 가이드
- **[WEB_FRONTEND_GUIDE.md](./frontend/WEB_FRONTEND_GUIDE.md)**: 웹 프론트엔드 가이드
- **[MODEL_SETUP_GUIDE.md](./MODEL_SETUP_GUIDE.md)**: 모델 설정 및 커스터마이징

---

## 🚧 개발 현황

- ✅ YOLO11 모델 통합 · Roboflow 데이터셋 연동
- ✅ 객체 탐지 및 추적 (BoT-SORT 카메라 움직임 보정, 공 단일 ID)
- ✅ 업로드 영상 분석 작업 · 탐지 박스 오버레이 · 팀 자동 분류
- ✅ 경기장 보정 (키프레임 + 카메라 움직임 전파) → 3D 트윈 연동
- ✅ FastAPI 백엔드 · 영상 업로드/제공 · Web 프론트엔드
- ✅ 단위·통합 테스트
- 🔶 3D Gaussian Splatting — 표준 공간·변형 MLP 모듈만 구현, 렌더링 미연동
- ⏳ 자동 경기장 보정 · 공 전용 탐지 모델 · GPU 추론
- ⏳ 히트맵 · 패스 네트워크 등 전술 분석 도구
- ⏳ SMPL 포즈 복원

자세한 로드맵은 [향후 구현 계획](./docs/PROJECT_OVERVIEW.md#6-향후-구현-계획)을 참고하세요.

---

## 📚 참고 자료

- [Roboflow Sports](https://github.com/roboflow/sports)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [boxmot](https://github.com/mikel-brostrom/boxmot)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Three.js 문서](https://threejs.org/docs/) · [React Three Fiber](https://r3f.docs.pmnd.rs/)
- [Deformable 3DGS](https://arxiv.org/abs/2312.00106)
- [Material Design – Dark theme](https://m2.material.io/design/color/dark-theme.html)
- [Pretendard](https://github.com/orioncactus/pretendard)

---

## 👥 기여

이슈 및 PR 환영합니다!

---

## 📄 라이선스

MIT License
