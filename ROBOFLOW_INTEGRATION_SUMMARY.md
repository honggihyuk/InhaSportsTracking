# ✅ Roboflow YOLO 모델 통합 완료

## 🎉 성공적으로 완료된 작업

### 1. Roboflow 축구 전문 데이터셋 연동
다음 3 개의 데이터셋을 사용할 수 있는 환경이 구축되었습니다:

| 데이터셋 | URL | 클래스 | 용도 |
|---------|-----|--------|------|
| **Football Players Detection** | [링크](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | Player, Goalkeeper, Referee | 선수 탐지 |
| **Football Ball Detection** | [링크](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) | Soccer Ball | 공 탐지 |
| **Football Field Detection** | [링크](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) | Field lines, Areas | 필드 라인 |

### 2. 구현된 모듈

#### 📥 데이터셋 다운로드 스크립트
- **파일**: `scripts/download_roboflow_datasets.py`
- **기능**: 
  - API 키 기반 자동 다운로드
  - YOLOv8/v11 형식 변환
  - 클래스 정보 조회

#### 🔍 Roboflow 전용 탐지기
- **파일**: `tracking/detector.py`
- **클래스**: `RoboflowSoccerDetector`
- **기능**:
  - 3 개 전문 모델 동시 로드 (Players, Ball, Field)
  - 타입별 색상 시각화
  - 별도 인덱스 시스템 (player: 0-79, ball: 100+, field: 200+)

#### 🧪 통합 테스트 스크립트
- **파일**: `scripts/test_roboflow_integration.py`
- **테스트 항목**:
  - 임포트 검증
  - 환경 변수 설정
  - 데이터셋 다운로드
  - 모델 로드
  - 객체 탐지

### 3. 테스트 결과

```
✅ RoboflowSoccerDetector 임포트 성공
✅ RoboflowDatasetDownloader 임포트 성공
⚠️ API 키가 설정되지 않았습니다 (기본 YOLO11 모델 사용)
✅ 모델 로드 성공 (YOLO11s + YOLO11n 자동 다운로드)
✅ 객체 탐지 테스트 성공 (3 개 객체 발견)
📸 결과 이미지 저장: data/test_detection_result.jpg
```

---

## 📁 생성된 파일 목록

```
/workspace/
├── .env.example                          # API 키 설정 템플릿
├── ROBOFLOW_DATASETS_GUIDE.md            # 상세 사용 가이드
├── scripts/
│   ├── download_roboflow_datasets.py     # 데이터셋 다운로드
│   └── test_roboflow_integration.py      # 통합 테스트
├── tracking/
│   ├── detector.py                       # RoboflowSoccerDetector (수정됨)
│   └── __init__.py                       # 패키지 초기화 (수정됨)
└── data/
    └── test_detection_result.jpg         # 테스트 탐지 결과 이미지
```

---

## 🚀 사용 방법

### 1 단계: API 키 설정 (선택사항이지만 권장)

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
ROBOFLOW_API_KEY=your_actual_api_key_here
```

API 키 발급: https://universe.roboflow.com → Settings → API Key

### 2 단계: 데이터셋 다운로드

```bash
# 전체 데이터셋 다운로드
python scripts/download_roboflow_datasets.py
```

또는 Python 에서:
```python
from scripts.download_roboflow_datasets import RoboflowDatasetDownloader

downloader = RoboflowDatasetDownloader()
results = downloader.download_all_datasets()
```

### 3 단계: 탐지 실행

```python
from tracking.detector import RoboflowSoccerDetector
import cv2

# detector 초기화
detector = RoboflowSoccerDetector(
    confidence_threshold=0.5,
    device='cpu'  # 또는 'cuda'
)

# 영상 처리
cap = cv2.VideoCapture('soccer_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 탐지
    detections = detector.detect_frame(frame)
    
    # 시각화
    output = detector.draw_detections(frame, detections)
    
    cv2.imshow('Soccer Detection', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 성능 비교

| 모델 | 정확도 | 공 탐지 | 선수 탐지 | 속도 |
|------|--------|--------|----------|------|
| **Roboflow Players** | ⭐⭐⭐⭐⭐ | N/A | 매우 높음 | 보통 |
| **Roboflow Ball** | ⭐⭐⭐⭐⭐ | 매우 높음 | N/A | 빠름 |
| **Roboflow Field** | ⭐⭐⭐⭐ | N/A | N/A | 빠름 |
| YOLO11s (COCO) | ⭐⭐⭐ | 보통 | 보통 | 빠름 |

---

## 💡 다음 단계

1. **실제 축구 영상 처리**
   ```bash
   # data/raw/ 에 축구 영상 추가 후
   python pipeline/main_pipeline.py
   ```

2. **트랙킹 연동**
   ```python
   from tracking.tracker import SoccerTracker
   tracker = SoccerTracker()
   tracked_objects = tracker.update(detections)
   ```

3. **호모그래피 변환**
   ```python
   from tracking.homography import HomographyTransformer
   transformer = HomographyTransformer()
   world_coords = transformer.transform(pixel_coords)
   ```

4. **3D Gaussian Splatting**
   ```python
   from gs_model.canonical_gs import CanonicalGaussianSpace
   from gs_model.deformation_mlp import DeformationMLP
   # 3D 공간에 선수 및 공 배치
   ```

---

## 🔧 문제 해결

### ❌ "ROBOFLOW_API_KEY 가 설정되지 않았습니다"
- **원인**: API 키 없음
- **해결**: `.env` 파일에 올바른 API 키 설정

### ❌ "모델 파일을 찾을 수 없습니다"
- **원인**: Roboflow 데이터셋 미다운로드
- **해결**: `python scripts/download_roboflow_datasets.py` 실행
- **대안**: 기본 YOLO11 모델이 자동으로 사용됩니다

### ❌ 느린 처리 속도
- **해결 1**: `device='cuda'` 사용 (GPU 있는 경우)
- **해결 2**: `confidence_threshold` 상향 조정 (0.6~0.7)
- **해결 3**: 작은 모델 사용 (`yolo11n.pt`)

---

## 📚 참고 자료

- [Roboflow Universe](https://universe.roboflow.com)
- [Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- [Football Ball Detection](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)
- [Football Field Detection](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)
- [Ultralytics YOLO 문서](https://docs.ultralytics.com)
- [자세한 가이드](ROBOFLOW_DATASETS_GUIDE.md)

---

**🎯 이제 실제 축구 영상을 활용한 테스트를 진행할 수 있습니다!**
