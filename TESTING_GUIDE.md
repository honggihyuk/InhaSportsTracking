# Soccer 3D Digital Twin 테스트 가이드

## 프로젝트 개요

이 프로젝트는 축구 영상 (2D) 에 움직임 트랙킹과 3D Gaussian Splatting 기술을 통합하여 스포츠 분석 시각화 솔루션을 구축합니다.

## 핵심 기술 스택

- **객체 탐지**: YOLOv8/v10 (Roboflow Sports 호환)
- **객체 추적**: ByteTrack / BoT-SORT
- **좌표 변환**: Homography 기반 2D→3D 매핑
- **3D 표현**: Canonical 3D Gaussian + Deformation MLP
- **렌더링**: Dynamic 3DGS (diff-gaussian-rasterization 연동 예정)

## 설치 방법

### 1. 기본 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 선택적 의존성 (CUDA 환경)

```bash
# 3D Gaussian Splatting 렌더러 (CUDA 필요)
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install git+https://github.com/graphdeco-inria/simple-knn.git
```

### 3. 모델 다운로드

Roboflow 에서 soccer detection 모델 다운로드:
- https://universe.roboflow.com/roboflow-jvuqo/soccer-detection-5nwp6

```bash
mkdir -p data/models
# yolo_soccer.pt 파일을 data/models/ 디렉토리에 배치
```

## 빠른 시작

### 1. 파이프라인 초기화 테스트

```bash
cd /workspace
PYTHONPATH=/workspace:$PYTHONPATH python pipeline/main_pipeline.py
```

### 2. Deformation MLP 테스트

```bash
PYTHONPATH=/workspace:$PYTHONPATH python gs_model/deformation_mlp.py
```

### 3. 개별 모듈 테스트

#### 객체 탐지 테스트
```python
from tracking.detector import SoccerDetector

detector = SoccerDetector(confidence_threshold=0.5)
# model_path 에 실제 YOLO 모델 파일 경로 지정 필요
```

#### 호모그래피 변환 테스트
```python
from tracking.homography import HomographyTransformer

transformer = HomographyTransformer()
pixel_corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
transformer.compute_from_field_lines(pixel_corners)

# 픽셀 → 월드 좌표 변환
world_coord = transformer.pixel_to_world(640, 360)
print(f"월드 좌표: {world_coord} 미터")
```

## 프로젝트 구조

```
/workspace/
├── README.md                    # 프로젝트 문서
├── requirements.txt             # Python 의존성
├── configs/
│   ├── model_config.yaml        # 메인 설정
│   └── field_dimensions.yaml    # 축구장 규격
├── tracking/
│   ├── detector.py              # YOLO 객체 탐지
│   ├── tracker.py               # ByteTrack 추적
│   ├── homography.py            # 호모그래피 변환
│   └── coordinate_mapper.py     # 3D 좌표 매핑
├── gs_model/
│   ├── canonical_gs.py          # 표준 공간 가우시안
│   └── deformation_mlp.py       # 변형 MLP 네트워크
├── pipeline/
│   └── main_pipeline.py         # 통합 파이프라인
├── visualizer/                  # 웹 시각화 (Three.js)
├── data/
│   ├── raw/                     # 원본 영상
│   ├── processed/               # 처리된 데이터
│   └── models/                  # 학습 모델
└── tests/                       # 테스트 코드
```

## 사용 예제

### 비디오 처리

```python
from pipeline.main_pipeline import Soccer3DPipeline

# 파이프라인 생성
pipeline = Soccer3DPipeline('configs/model_config.yaml')

# 컴포넌트 초기화
pipeline.initialize_components()

# 캘리브레이션 데이터 로드
pipeline.load_calibration('data/calibration.json')

# 비디오 처리
processed_data = pipeline.process_video(
    video_path='data/raw/match_video.mp4',
    output_dir='data/processed',
    max_frames=300  # 처음 300 프레임만 처리
)

# 추적 결과 시각화
import cv2
cap = cv2.VideoCapture('data/raw/match_video.mp4')

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_count >= len(processed_data):
        break
        
    frame_data = processed_data[frame_count]
    vis_frame = pipeline.visualize_tracking(frame, frame_data)
    
    cv2.imshow('Tracking', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_count += 1
    
cap.release()
cv2.destroyAllWindows()
```

### 궤적 데이터 내보내기

```python
# CSV 로 저장
pipeline.export_trajectory(
    processed_data=processed_data,
    output_file='data/processed/trajectories.csv'
)
```

## 설정 파일 예제

### configs/model_config.yaml

```yaml
device: cuda  # 또는 cpu

detector:
  model_path: data/models/yolo_soccer.pt
  confidence_threshold: 0.5
  iou_threshold: 0.45

tracker:
  type: bytetrack  # 또는 botsort
  track_threshold: 0.3
  high_threshold: 0.6
  low_threshold: 0.1
  max_age: 30

gaussian:
  num_gaussians: 10000
  sh_degree: 3
  feature_dim: 256

deformation_mlp:
  input_dim: 20
  feature_dim: 256
  num_layers: 8
  use_positional_encoding: true
  num_freqs: 10
```

## 캘리브레이션

### 호모그래피 행렬 계산

경기장의 4 개 코너 점을 이미지 상에서 찾아 대응시킵니다:

```python
from tracking.homography import HomographyTransformer

transformer = HomographyTransformer()

# 이미지 상의 경기장 4 개 코너 (픽셀 좌표)
image_corners = [
    (100, 100),   # top_left
    (1180, 100),  # top_right
    (1180, 620),  # bottom_right
    (100, 620)    # bottom_left
]

# 자동 계산 (FIFA 규격 축구장 사용)
success = transformer.compute_from_field_lines(image_corners)

# 캘리브레이션 저장
transformer.save_calibration('data/calibration.json')
```

## 기술적 특징

### 1. Deformation MLP

시간 및 포즈 정보를 기반으로 Canonical Gaussian 을 변형:

- **입력**: 가우시안 위치 + 시간 + SMPL 포즈 파라미터
- **출력**: Δ위치, Δ회전, Δ크기, Δ불투명도
- **구조**: 8 레이어 MLP + Positional Encoding

### 2. Human Gaussian Template

SMPL 메쉬 기반 인간 형태 가우시안:

- 정점당 5 개 가우시안 할당
- 자연스러운 인간 동작 표현
- GauHuman, Animatable Gaussians 기술 참조

### 3. 실시간 렌더링

- diff-gaussian-rasterization CUDA 커널 사용
- 100+ FPS 가능 (경량 MLP 적용시)
- WebGPU/Three.js 연동으로 웹 브라우저에서도 실행 가능

## 다음 단계

1. **실제 축구 영상 수집 및 처리**
   - 멀티뷰 카메라 또는 방송용 영상 확보
   - 카메라 캘리브레이션 수행

2. **SMPL 포즈 추정 연동**
   - RTMPose / OpenPose 도입
   - 3D Skeleton Keypoints 추출

3. **3DGS 렌더러 완성**
   - diff-gaussian-rasterization 연동
   - 실시간 Free-viewpoint Rendering 구현

4. **웹 비주얼라이저 개발**
   - Three.js 기반 3D 뷰어
   - 자유 시점 전환 UI

5. **데이터 분석 기능**
   - 선수 이동 거리, 속도 분석
   - 패스 네트워크 시각화
   - LLM 기반 질의 응답

## 참고 자료

- [Roboflow Sports](https://github.com/roboflow/sports)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Deformable 3DGS](https://arxiv.org/abs/2312.11243)
- [GauHuman](https://github.com/johannwagner/gauhuman)
- [BoxMOT Tracking](https://github.com/mikel-brostrom/boxmot)

## 라이선스

MIT License
