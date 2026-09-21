# ⚽ Soccer 3D Digital Twin

**축구 영상 기반 3D 디지털 트윈 분석 플랫폼**

2D 축구 영상을 실시간으로 3D 디지털 트윈으로 변환하고, 선수 및 공의 움직임을 추적하여 자유로운 시점에서 경기를 분석할 수 있는 솔루션입니다.

---

## 🎯 프로젝트 개요

### 핵심 기술 스택

- **Object Detection**: YOLO11 + Roboflow 전문 모델
- **Tracking**: ByteTrack/BoT-SORT (또는 Simple Tracker)
- **Coordinate Mapping**: Homography + Camera Calibration
- **3D Rendering**: Three.js (Web), 3D Gaussian Splatting (예정)
- **Deformation Network**: MLP 기반 Dynamic Gaussian 변형
- **Frontend**: React + Vite + Three.js

### 주요 기능

1. **실시간 객체 탐지 및 추적**: 선수, 공, 심판 자동 식별
2. **2D → 3D 좌표 변환**: 호모그래피를 통한 경기장 월드 좌표 매핑
3. **3D 디지털 트윈**: 가상 공간에서 경기 재현
4. **Free-Viewpoint Rendering**:任意 시점에서의 경기 관람
5. **실시간 통계 시각화**: 선수 위치, 속도, 이동 거리 등

---

## 📁 프로젝트 구조

```
InhaSportsTracking/
├── tracking/               # 객체 탐지 및 추적 모듈
│   ├── detector.py         # YOLO 기반 객체 탐지
│   ├── tracker.py          # 객체 추적 (ByteTrack/Simple)
│   ├── homography.py       # 호모그래피 변환
│   └── coordinate_mapper.py # 2D → 3D 좌표 매핑
├── gs_model/               # 3D Gaussian Splatting 모델
│   ├── canonical_gs.py     # 표준 공간 가우시안 정의
│   └── deformation_mlp.py  # 변형 MLP 네트워크
├── pipeline/               # 통합 처리 파이프라인
│   └── main_pipeline.py    # 메인 파이프라인 실행기
├── frontend/               # Web 프론트엔드 (React)
│   ├── src/
│   │   ├── App.jsx         # 메인 애플리케이션
│   │   └── main.jsx        # 엔트리 포인트
│   └── package.json
├── configs/                # 설정 파일
│   ├── model_config.yaml   # 모델 및 파이프라인 설정
│   └── field_dimensions.yaml # 축구장 규격
├── data/                   # 데이터 저장소
│   ├── raw/                # 원본 비디오
│   └── roboflow_datasets/  # Roboflow 다운로드 데이터
├── models/                 # 사전 학습 모델
│   ├── yolo11s.pt          # YOLO11s 모델
│   └── yolo11n.pt          # YOLO11n 경량 모델
├── .env                    # 환경 변수 (API 키 등)
├── requirements.txt        # Python 의존성
└── README.md               # 이 파일
```

---

## 🚀 빠른 시작

### 1. Python 백엔드 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# Roboflow API 키 설정 (.env 파일 수정)
echo "ROBOFLOW_API_KEY=your_api_key" > .env

# (선택사항) Roboflow 전문 모델 다운로드
python setup_roboflow.py

# 파이프라인 실행
python -m pipeline.main_pipeline --config configs/model_config.yaml
```

### 2. Web 프론트엔드 설정

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 시작
npm run dev
```

브라우저에서 `http://localhost:5173` 접속

---

## 📊 사용된 데이터셋

### Roboflow 전문 모델

1. **[Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)**
   - 선수, 심판, 골키퍼 탐지
   - 22,000+ 라벨링 이미지

2. **[Football Ball Detection](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)**
   - 축구공 탐지
   - 다양한 각도 및 조명 조건

3. **[Football Field Detection](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)**
   - 경기장 라인 및 영역 탐지
   - 호모그래피 캘리브레이션용

### 기본 모델 (COCO)

- **YOLO11s**: 80 클래스 일반 객체 탐지
- person (Class 0), sports ball (Class 32) 사용

---

## 🔧 설정 가이드

### `.env` 파일

```bash
# Roboflow API 키 (필수)
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

API 키 발급: https://app.roboflow.com/settings/api

### `configs/model_config.yaml`

```yaml
device: cpu  # cpu 또는 cuda
use_roboflow_models: false  # true 로 변경시 Roboflow 전문 모델 사용

detector:
  model_path: models/yolo11s.pt
  confidence: 0.25
  classes: [0, 32]  # person, ball

tracker:
  type: simple  # simple 또는 boxmot
  track_threshold: 0.5
```

---

## 📈 아키텍처 다이어그램

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Video Input    │────▶│  Object Detection │────▶│   Object Track  │
│  (MP4, AVI)     │     │  (YOLO11)         │     │  (ByteTrack)    │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  3D Rendering   │◀────│  Gaussian Space  │◀────│  3D Coordinate  │
│  (Three.js)     │     │  + Deformation   │     │    Mapping      │
└─────────────────┘     └──────────────────┘     └──────────────────┘
        │
        ▼
┌─────────────────┐
│  Web Frontend   │
│  (React + UI)   │
└─────────────────┘
```

---

## 🎨 Web 프론트엔드 기능

### 뷰 모드

1. **3D 뷰**: 입체적인 경기장 렌더링 (Three.js)
2. **탑다운 뷰**: 전술 분석용 평면 뷰
3. **선수 시점**: 특정 선수 1 인칭 시점

### 컨트롤 패널

- 비디오 업로드 및 재생 제어
- 타임라인 시크 바
- 실시간 통계 (선수 수, 공 위치, 속도)
- 선수 목록 및 팀별 분류

### 향후 추가 예정

- WebSocket 실시간 데이터 연동
- SMPL 인체 모델 기반 포즈 복원
- 3D Gaussian Splatting 고품질 렌더링
- 전술 분석 도구 (패스 라인, 히트맵)

---

## 📝 문서

- **[TESTING_GUIDE.md](./TESTING_GUIDE.md)**: 테스트 및 사용 가이드
- **[WEB_FRONTEND_GUIDE.md](./frontend/WEB_FRONTEND_GUIDE.md)**: 웹 프론트엔드 상세 가이드
- **[MODEL_SETUP_GUIDE.md](./MODEL_SETUP_GUIDE.md)**: 모델 설정 및 커스터마이징

---

## 🚧 개발 현황

- ✅ YOLO11 모델 통합
- ✅ Roboflow 데이터셋 연동
- ✅ 객체 탐지 및 추적 모듈
- ✅ 호모그래피 좌표 변환
- ✅ Canonical 3D Gaussian Space
- ✅ Deformation MLP 구현
- ✅ Web 프론트엔드 (React + Three.js)
- ⏳ 실시간 WebSocket 연동
- ⏳ 3D Gaussian Splatting 렌더러
- ⏳ SMPL 포즈 추정 연동

---

## 📚 참고 자료

- [Roboflow Sports](https://github.com/roboflow/sports)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Three.js 문서](https://threejs.org/docs/)
- [Deformable 3DGS](https://arxiv.org/abs/2312.00106)

---

## 👥 기여

이슈 및 PR 환영합니다!

---

## 📄 라이선스

MIT License
