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
3. **경기장 보정**: **자동 보정**(기준점 입력 없이 경기장 라인·센터서클로 키프레임 검출 — [5부 문서](./docs/PROJECT_OVERVIEW_5_AUTO_CALIBRATION.md)) 또는 한 프레임에서 기준점 4 개 이상 직접 지정 → 카메라 팬·줌을 따라 전 프레임에 전파
4. **3D 디지털 트윈 연동**: 분석·보정된 선수·공 위치를 영상과 같은 프레임으로 3D 재현 (자유 시점 / 탑다운 / 선수 시점)
5. **통계**: 팀별 선수 수·평균/최고 속도, 볼 소유, 공 속도·위치
6. **고정밀 분석 모드 (기본)**: 매 프레임 탐지 + 공 타일 탐지, 끊긴 트랙 잇기·가려진 구간 보간, **경기장 라인 정렬로 보정 누적 오차 제거**, 좌표 칼만+RTS 평활화, 골키퍼 판정, 경기 지표(거리·스프린트·점유율·패스·대형·히트맵) — [4부 문서](./docs/PROJECT_OVERVIEW_4_PRECISE_ANALYSIS.md)

---

## 📁 프로젝트 구조

```
InhaSportsTracking/                 ← 모든 명령은 여기(저장소 루트)에서 실행
├── backend/
│   └── server.py                   # FastAPI: 업로드·영상 제공·분석 작업·보정·경기 지표 API, WebSocket
├── pipeline/                       # 영상 분석
│   ├── video_analysis.py           # 분석 작업(탐지·추적·카메라 움직임·팀), 보정 전파·저장
│   ├── pitch.py                    # 경기장 라인 모델·라인 검출·라인 정렬 보정
│   ├── autocalib.py                # 자동 초기 보정 (기준점 입력 없이)
│   ├── postprocess.py              # 트랙 잇기·보간·공 정리·좌표 평활화·골키퍼
│   ├── synthetic.py                # 정답이 있는 합성 중계 영상 (테스트·벤치마크)
│   └── main_pipeline.py            # 설정 기반 파이프라인, 분석 모드 프로필, CLI
├── tracking/                       # 탐지·추적·좌표 변환
│   ├── detector.py                 # YOLO 탐지(Detection), 공 타일 추론, 모델 경로 해석
│   ├── tracker.py                  # BoT-SORT/ByteTrack/Simple, BallTracker, SoccerTracker
│   ├── homography.py
│   └── coordinate_mapper.py
├── analysis/
│   └── stats.py                    # 경기 지표 (거리·스프린트·점유율·패스·대형·히트맵)
├── gs_model/                       # 3D Gaussian Splatting 모듈 (렌더링 미연동)
├── frontend/                       # 웹 화면 (React + Vite + Three.js)
├── configs/
│   ├── model_config.yaml           # 모델·분석 모드·추적 설정
│   └── field_dimensions.yaml       # 경기장 규격
├── models/                         # YOLO11 가중치 (yolo11n.pt, yolo11s.pt)
├── scripts/                        # 실행용 스크립트
│   ├── benchmark_precise.py        # 분석 모드·자동 보정 정량 비교
│   ├── setup_roboflow.py           # Roboflow 데이터셋 다운로드
│   ├── download_roboflow_datasets.py
│   ├── test_roboflow_integration.py
│   └── sample_test.py              # 모듈 점검 + 샘플 영상 생성
├── tests/                          # pytest (68 개)
├── docs/
│   ├── PROJECT_OVERVIEW*.md        # 기술·기능 명세 1~5부
│   └── guides/                     # 설치·실행, 모델, Roboflow, 테스트, 프론트엔드 가이드
├── data/raw/                       # 샘플 영상
├── uploads/                        # 업로드 영상·분석 결과 (git 제외, 서버가 자동 생성)
├── .env.example                    # Roboflow API 키 템플릿 (.env 로 복사, git 제외)
└── requirements.txt
```

---

## 🚀 빠른 시작

> 자세한 설치(가상환경·CPU 전용 PyTorch 등)와 문제 해결은 **[설치 · 서버 실행 가이드](./docs/guides/SETUP.md)**.

### 1. 설치

```bash
pip install -r requirements.txt     # 저장소 루트에서 (가상환경 권장)
cd frontend && npm install && cd ..
```

### 2. 서버 실행 (터미널 두 개)

```bash
# 백엔드 → http://localhost:8000  (저장소 루트에서)
python -m uvicorn backend.server:app --port 8000

# 프론트엔드 → http://localhost:5173
cd frontend
npm run dev
```

브라우저에서 `http://localhost:5173` 에 접속합니다.

### 3. 사용

1. 라이브러리에서 **업로드**로 경기 영상을 올리고 선택합니다.
2. **영상 분석** 카드에서 모드(**고정밀** / 빠른 미리보기)를 고르고 **분석 시작** — 진행 중에는 취소할 수 있습니다.
3. **경기장 보정** — 고정밀 분석은 끝나면 **자동으로 보정**합니다. 빠른 미리보기 결과는 **자동 보정** 버튼을 누르세요. 자동 보정이 안 되는 구간은 **기준점 직접 지정**: 오른쪽 도면에서 기준점(코너, 페널티박스 모서리, 센터서클과 하프라인 교점 등)을 고르고 영상에서 같은 지점을 클릭합니다 (한 직선 위에 있지 않은 4 점 이상).
4. 탐지 박스·3D 트윈·**경기 지표**(점유율·거리·스프린트·패스, 이벤트 클릭 시 해당 장면으로 이동)가 영상과 같은 타임라인으로 재생됩니다. **보정된 경기장 라인 표시**로 보정 정확도를 눈으로 확인할 수 있습니다.

> CPU 에서는 분석이 영상 길이의 수십 배 걸립니다 (빠른 미리보기 약 1 초/프레임, 고정밀은 그 이상). GPU 가 있으면 `detection.device: cuda` 를 권장합니다.

> 브라우저가 재생할 수 있는 **H.264(MP4)** 또는 **VP9/AV1(WebM)** 영상을 사용하세요: `ffmpeg -i input.mp4 -c:v libx264 -c:a aac -movflags +faststart output.mp4`

### 4. 테스트

```bash
python -m pytest tests -q            # 저장소 루트에서
```

YOLO·GPU 없이 합성 영상과 가짜 탐지기로 실행됩니다.

```bash
# 실시간 vs 고정밀 모드 정량 비교 (정답이 있는 합성 중계 영상)
python scripts/benchmark_precise.py --frames 300
```

---

## 📊 사용된 데이터셋

### Roboflow 전문 모델

1. **[Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)** — 선수, 심판, 골키퍼 탐지
2. **[Football Ball Detection](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)** — 축구공 탐지
3. **[Football Field Detection](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)** — 경기장 라인·영역 탐지 (캘리브레이션용)

### 기본 모델 (COCO)

- **YOLO11s / YOLO11n**: 80 클래스 일반 객체 탐지 — person (Class 0), sports ball (Class 32)
- Roboflow 모델이 없으면 자동으로 대체 사용됩니다 (COCO 모델은 `person`·`sports ball` 클래스만 추론). 작은 공 탐지 정확도가 필요하면 Roboflow 모델 학습을 권장합니다 — [모델 가이드](./docs/guides/MODELS.md)

---

## 🔧 설정 가이드

### `.env` 파일 (선택 — Roboflow 모델을 쓸 때만)

```bash
cp .env.example .env                 # Windows: copy .env.example .env
# .env 에 ROBOFLOW_API_KEY=발급받은_키 입력 → python scripts/setup_roboflow.py
```

`.env` 는 git 에 올라가지 않습니다. 자세한 내용: [Roboflow 가이드](./docs/guides/ROBOFLOW.md)

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
  mode: precise                # 기본 분석 모드 (API ?mode=realtime|precise 로 영상별 선택)
  profiles:
    realtime: {stride: 3, img_size: 640, ball_imgsz: 960, ball_tile: null, postprocess: false, line_refine: false}
    precise:  {stride: 1, img_size: 960, ball_imgsz: 1280, ball_tile: 640, postprocess: true, line_refine: true}

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
     경기장 보정 (자동: 라인·센터서클 / 수동: 기준점 4+) ──▶ 키프레임 × 카메라 움직임 + 라인 정렬 ──▶ 프레임별 경기장 좌표
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

**기술·기능 명세** (`docs/`)

- **[docs/PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md)**: 사용 기술·구현 기능·API·로드맵 상세 명세 (1부)
- **[docs/PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md](./docs/PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md)**: 영상 분석·경기장 보정·3D 트윈 연동 기술 상세와 실측 결과 (2부)
- **[docs/PROJECT_OVERVIEW_3_VLM.md](./docs/PROJECT_OVERVIEW_3_VLM.md)**: 컴퓨터 비전 고도화(RF-DETR·SAM3·SigLIP2·GLM-OCR)와 로컬 VLM 분석 층 조사·설계 (3부)
- **[docs/PROJECT_OVERVIEW_4_PRECISE_ANALYSIS.md](./docs/PROJECT_OVERVIEW_4_PRECISE_ANALYSIS.md)**: 고정밀 분석 모드 — 라인 정렬 보정·오프라인 후처리·경기 지표 구현과 정량 비교 (4부)
- **[docs/PROJECT_OVERVIEW_5_AUTO_CALIBRATION.md](./docs/PROJECT_OVERVIEW_5_AUTO_CALIBRATION.md)**: 자동 초기 보정 — 기준점 입력 없이 경기장 라인·센터서클 기하로 보정 (5부)

**가이드** (`docs/guides/`)

- **[SETUP.md](./docs/guides/SETUP.md)**: 설치 · 서버 실행 · 실행 중 생기는 폴더 · 문제 해결
- **[MODELS.md](./docs/guides/MODELS.md)**: 탐지 모델 · 설정 · 코드에서 사용
- **[ROBOFLOW.md](./docs/guides/ROBOFLOW.md)**: Roboflow 축구 전용 데이터셋 다운로드 · 학습 · 적용
- **[TESTING.md](./docs/guides/TESTING.md)**: 테스트 · 벤치마크 · 점검 스크립트
- **[FRONTEND.md](./docs/guides/FRONTEND.md)**: 웹 화면 구성 · 기능

---

## 🚧 개발 현황

- ✅ YOLO11 모델 통합 · Roboflow 데이터셋 연동
- ✅ 객체 탐지 및 추적 (BoT-SORT 카메라 움직임 보정, 공 단일 ID)
- ✅ 업로드 영상 분석 작업 · 탐지 박스 오버레이 · 팀 자동 분류
- ✅ 경기장 보정 (키프레임 + 카메라 움직임 전파) → 3D 트윈 연동
- ✅ FastAPI 백엔드 · 영상 업로드/제공 · Web 프론트엔드
- ✅ 고정밀 분석 모드 — 경기장 라인 정렬 보정, 트랙 잇기·보간, 좌표 평활화, 골키퍼 판정
- ✅ 경기 지표 — 이동 거리·스프린트·점유율·패스/턴오버·팀 대형·히트맵 (`/analysis/{영상}/stats`)
- ✅ 자동 경기장 보정 — 기준점 입력 없이 라인·센터서클 기하로 키프레임 검출, 교차 검증 (`/analysis/{영상}/calibration/auto`)
- ✅ 단위·통합 테스트 (68 개)
- 🔶 3D Gaussian Splatting — 표준 공간·변형 MLP 모듈만 구현, 렌더링 미연동
- ⏳ 경기장 키포인트 모델(자동 보정 가속) · 공 전용 탐지 모델 · GPU 추론
- ⏳ 히트맵·패스 네트워크 화면 표시 (데이터는 생성됨) · VLM 질의 층
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
