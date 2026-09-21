# 🎉 Soccer 3D Digital Twin 프로젝트 설정 완료

## ✅ 현재 상태

모든 핵심 컴포넌트가 정상적으로 초기화되고 작동합니다!

### 작동 확인된 모듈
- ✅ **YOLO11s/n 객체 탐지기** - CPU 에서 정상 작동
- ✅ **SoccerTracker** - SimpleTracker 모드로 자동 전환 (boxmot 없이 IoU 기반 추적)
- ✅ **HomographyTransformer** - 2D → 월드 좌표 변환
- ✅ **CoordinateMapper** - 3D 좌표 매핑
- ✅ **CanonicalGaussianSpace** - 10,000 개 가우시안 초기화
- ✅ **DeformationMLP** - 변형 네트워크
- ✅ **DynamicGaussianRenderer** - 동적 렌더러

### 파이프라인 테스트 결과
```
=== Soccer 3D Digital Twin Pipeline ===
✅ 파이프라인 초기화 성공!
✅ 테스트 완료!
```

---

## 🔧 해결된 문제들

### 1. **PyTorch CPU 버전 설치**
- **문제**: GPU 버전 PyTorch 가 CPU-only 환경에서 Bus error 발생
- **해결**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- **현재 버전**: PyTorch 2.14.0+cpu

### 2. **NumPy 버전 호환성**
- **문제**: NumPy 2.x 와 일부 패키지의 호환성 문제
- **해결**: `pip install numpy==1.26.4 --force-reinstall`

### 3. **boxmot 추적기 대안 마련**
- **문제**: boxmot v19+ 의 import 경로 변경 및 CUDA 의존성
- **해결**: 
  - `SimpleTracker` 클래스 추가 (IoU 기반 간단한 추적)
  - boxmot 이 없으면 자동으로 SimpleTracker 사용
  - boxmot 설치시: `pip install boxmot==10.0.84`

### 4. **Roboflow 데이터셋 다운로드**
- **완료**: Football Players Detection (v3)
- **완료**: Football Ball Detection (v1)
- **실패**: Football Field Detection (keypoint-detection 유형은 YOLOv11 지원하지 않음)
  - 대체: 기본 YOLO11 모델로 필드 라인 탐지 가능

---

## 📁 프로젝트 구조

```
/workspace/
├── pipeline/
│   └── main_pipeline.py      # 통합 파이프라인 (테스트 완료 ✅)
├── tracking/
│   ├── detector.py           # YOLO 기반 객체 탐지
│   ├── tracker.py            # SimpleTracker + SoccerTracker
│   ├── homography.py         # 호모그래피 변환
│   └── coordinate_mapper.py  # 3D 좌표 매핑
├── gs_model/
│   ├── canonical_gs.py       # 표준 공간 가우시안
│   └── deformation_mlp.py    # 변형 MLP
├── configs/
│   └── model_config.yaml     # 설정 파일
├── models/
│   ├── yolo11s.pt           # 메인 탐지 모델
│   └── yolo11n.pt           # 경량 탐지 모델
├── data/
│   ├── raw/                 # 입력 영상 폴더
│   └── roboflow_datasets/   # Roboflow 데이터셋
├── .env                     # API 키 설정
└── README.md                # 프로젝트 문서
```

---

## 🚀 다음 단계

### 1. **실제 축구 영상 처리**
```bash
# data/raw/ 폴더에 축구 영상 추가 (.mp4, .avi 등)
python -m pipeline.main_pipeline --config configs/model_config.yaml --video data/raw/match.mp4
```

### 2. **Roboflow 전문 모델 사용 (선택사항)**
```bash
# 1. .env 파일에 API 키 입력
echo "ROBOFLOW_API_KEY=your_key_here" > .env

# 2. 데이터셋 다운로드
python setup_roboflow.py

# 3. configs/model_config.yaml 에서 use_roboflow_models: true 변경
```

### 3. **고도화 작업**
- [ ] SMPL 포즈 추정 연동 (RTMPose/OpenPose)
- [ ] diff-gaussian-rasterization CUDA 렌더러 통합
- [ ] 웹 비주얼라이저 (Three.js/deck.gl)
- [ ] 공간 DB 적재 (PostGIS/SQLite)
- [ ] LLM 기반 데이터 질의 인터페이스

---

## 💡 사용 팁

### SimpleTracker vs ByteTrack
- **SimpleTracker** (현재 기본): 
  - ✅ 별도 설치 불필요
  - ✅ CPU 에서 빠름
  - ⚠️ IoU 기반 단순 매칭 (복잡한 움직임에서 ID 스위치 가능성)
  
- **ByteTrack** (설치 시):
  - ✅ 더 정확한 ID 유지
  - ✅ 저신뢰도 탐지도 활용
  - ⚠️ `pip install boxmot==10.0.84` 필요
  - ⚠️ CUDA 권장

### 성능 최적화
- 작은 영상으로 테스트: `--video data/raw/sample_short.mp4`
- 탐지 임계값 조정: `configs/model_config.yaml` 에서 `conf_threshold` 변경
- 프레임 스킵: 긴 영상은 `frame_skip` 파라미터 사용

---

## 📊 기술 스택

| 분야 | 기술 |
|------|------|
| 객체 탐지 | YOLO11s/n (Ultralytics) |
| 객체 추적 | SimpleTracker (IoU 기반) / ByteTrack (선택) |
| 좌표 변환 | OpenCV Homography |
| 3DGS | PyTorch 기반 Custom 구현 |
| 변형 MLP | 8 레이어 Positional Encoding MLP |
| 렌더링 | CPU Rasterization (CUDA 버전 준비 중) |

---

## 🔗 참고 링크

- [Roboflow Sports](https://github.com/roboflow/sports)
- [Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- [Football Ball Detection](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)
- [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

**프로젝트가 준비되었습니다! 실제 축구 영상을 넣어 테스트해보세요.** ⚽🏃
