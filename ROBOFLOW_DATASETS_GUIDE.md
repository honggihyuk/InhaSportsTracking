# Roboflow 축구 데이터셋 사용 가이드

## 📦 다운로드된 데이터셋

Roboflow 에서 제공하는 3 개의 축구 전문 데이터셋을 사용합니다:

### 1. Football Players Detection
- **URL**: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
- **클래스**: Player, Goalkeeper, Referee
- **용도**: 경기장 내 모든 인물 탐지

### 2. Football Ball Detection  
- **URL**: https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg
- **클래스**: Soccer Ball
- **용도**: 공의 정확한 위치 추적

### 3. Football Field Detection
- **URL**: https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi
- **클래스**: Field lines, Penalty area, Center circle 등
- **용도**: 경기장 라인 인식 및 호모그래피 계산

---

## 🔑 API 키 설정 방법

### 1 단계: API 키 발급
1. https://universe.roboflow.com 접속
2. 우측 상단 프로필 아이콘 클릭 → **Settings**
3. **API Key** 섹션에서 키 복사

### 2 단계: .env 파일 생성
```bash
cd /workspace
cp .env.example .env
```

`.env` 파일 편집:
```env
ROBOFLOW_API_KEY=your_actual_api_key_here
FOOTBALL_PLAYERS_VERSION=3
FOOTBALL_BALL_VERSION=1
FOOTBALL_FIELD_VERSION=1
```

---

## 📥 데이터셋 다운로드

### 방법 1: 스크립트 실행 (권장)
```bash
python scripts/download_roboflow_datasets.py
```

### 방법 2: Python 코드에서 직접
```python
from scripts.download_roboflow_datasets import RoboflowDatasetDownloader

downloader = RoboflowDatasetDownloader()

# 개별 데이터셋 다운로드
players_path = downloader.download_dataset('players')
ball_path = downloader.download_dataset('ball')
field_path = downloader.download_dataset('field')

# 또는 한 번에 모두 다운로드
results = downloader.download_all_datasets()
```

---

## 🚀 모델 사용 예제

### 기본 사용법
```python
from tracking.detector import RoboflowSoccerDetector

# detector 초기화 (자동으로 다운로드된 모델 로드)
detector = RoboflowSoccerDetector(
    confidence_threshold=0.5,
    device='cpu'  # 'cuda' 사용 가능
)

# 영상 프레임 읽기
import cv2
cap = cv2.VideoCapture('soccer_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 객체 탐지
    detections = detector.detect_frame(frame)
    
    # 결과 시각화
    output = detector.draw_detections(frame, detections)
    
    cv2.imshow('Detection', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 고급 사용법 (커스텀 모델 경로)
```python
detector = RoboflowSoccerDetector(
    players_model_path='data/roboflow_datasets/football-players-detection/weights/best.pt',
    ball_model_path='data/roboflow_datasets/football-ball-detection/weights/best.pt',
    field_model_path='data/roboflow_datasets/football-field-detection/weights/best.pt',
    confidence_threshold=0.6,
    iou_threshold=0.45,
    device='cuda'
)

# 선수만 탐지
player_dets = detector.detect(frame, detect_players=True, detect_ball=False)

# 공만 탐지
ball_dets = detector.detect(frame, detect_players=False, detect_ball=True)

# 필드 라인 탐지 (호모그래피용)
field_dets = detector.detect(frame, detect_players=False, detect_ball=False, detect_field=True)
```

### 탐지 결과 처리
```python
detections = detector.detect_frame(frame)

for det in detections:
    print(f"타입: {det['type']}")
    print(f"클래스: {det['class_name']}")
    print(f"신뢰도: {det['confidence']:.2f}")
    print(f"바운딩 박스: {det['bbox']}")
    print(f"중심점: {det['center']}")
    print(f"면적: {det['area']}")
    print("---")
```

---

## 📊 데이터셋 구조

다운로드 후 디렉토리 구조:
```
data/
└── roboflow_datasets/
    ├── football-players-detection/
    │   ├── train/
    │   ├── valid/
    │   ├── test/
    │   ├── weights/
    │   │   └── best.pt      # 학습된 모델
    │   └── data.yaml        # 클래스 정보
    ├── football-ball-detection/
    │   └── ... (동일한 구조)
    └── football-field-detection/
        └── ... (동일한 구조)
```

---

## 🔧 문제 해결

### ❌ "ROBOFLOW_API_KEY 가 설정되지 않았습니다"
**해결**: `.env` 파일에 올바른 API 키를 설정했는지 확인

### ❌ "모델 파일을 찾을 수 없습니다"
**해결**: 
1. `python scripts/download_roboflow_datasets.py` 재실행
2. 또는 기본 YOLO11 모델이 자동으로 사용됩니다

### ❌ "CUDA out of memory"
**해결**: 
```python
detector = RoboflowSoccerDetector(device='cpu')  # CPU 사용
# 또는
detector = RoboflowSoccerDetector(confidence_threshold=0.7)  # 임계값上调
```

---

## 📈 성능 비교

| 모델 | 정확도 | 속도 | 용도 |
|------|--------|------|------|
| **Roboflow Players** | 매우 높음 ⭐⭐⭐⭐⭐ | 보통 | 전문 축구 선수 탐지 |
| **Roboflow Ball** | 매우 높음 ⭐⭐⭐⭐⭐ | 빠름 | 작은 공도 정확 탐지 |
| **Roboflow Field** | 높음 ⭐⭐⭐⭐ | 빠름 | 라인/영역 탐지 |
| YOLO11s (COCO) | 보통 ⭐⭐⭐ | 빠름 | 일반 객체 탐지 |

---

## 💡 팁

1. **정확도 향상**: `confidence_threshold=0.6~0.7` 로 설정하면 오탐지 감소
2. **속도 향상**: `device='cuda'` 사용 시 GPU 가속
3. **공 탐지 강화**: 공은 작아서 `confidence_threshold=0.4` 로 낮추는 것 추천
4. **멀티뷰**: 여러 카메라 각도에서 필드 라인을 탐지하면 호모그래피 정확도 향상

---

## 📚 참고 자료

- [Roboflow Universe](https://universe.roboflow.com)
- [Football Players Detection Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- [Football Ball Detection Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)
- [Football Field Detection Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)
- [Ultralytics YOLO 문서](https://docs.ultralytics.com)
