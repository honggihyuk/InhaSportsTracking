# YOLO 모델 설정 가이드

## 📦 다운로드된 모델

### 현재 사용 가능한 모델

| 모델 | 크기 | 용도 | 경로 |
|------|------|------|------|
| **YOLO11s** (Primary) | 18.4 MB | 축구 선수/공 탐지 (기본) | `models/yolo11s.pt` |
| **YOLO11n** (Fast) | 5.4 MB | 빠른 추론, 낮은 정확도 | `models/yolo11n.pt` |

### 모델 특징

#### YOLO11s (추천)
- **아키텍처**: Ultralytics YOLO11 Small
- **학습 데이터**: COCO dataset (80 클래스)
- **축구 관련 클래스**:
  - Class 0: `person` (선수, 심판, 골키퍼)
  - Class 32: `sports ball` (축구공)
- **성능**: 
  - CPU: ~30 FPS
  - GPU: ~60+ FPS
- **정확도**: mAP 50-59% (COCO 기준)

#### YOLO11n (경량)
- **용도**: 실시간 처리가 중요한 경우
- **정확도**: YOLO11s 대비 약 10% 낮음
- **속도**: YOLO11s 대비 2 배 빠름

---

## 🔧 Roboflow 전문 모델 (선택사항)

Roboflow 에서 제공하는 축구 전용 모델을 사용하려면:

### 1. Roboflow 계정 생성
1. https://universe.roboflow.com 접속
2. 무료 계정 생성

### 2. API 키 발급
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
```

### 3. 축구 전용 모델 다운로드

#### 옵션 A: Football Players Detection
```python
project = rf.workspace("roboflow").project("football-players-detection-3zv9t")
version = project.version(3)
model = version.download("yolov8", location="models/")
```
- **클래스**: player, goalkeeper, referee, ball, soccer field
- **URL**: https://universe.roboflow.com/roboflow/football-players-detection-3zv9t

#### 옵션 B: Soccer Ball Tracking
```python
project = rf.workspace("roboflow").project("soccer-ball-tracking")
version = project.version(1)
model = version.download("yolov8", location="models/")
```
- **클래스**: soccer ball (공 추적 특화)
- **URL**: https://universe.roboflow.com/roboflow/soccer-ball-tracking

---

## 💻 사용 예제

### 기본 usage (COCO 모델)
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO("models/yolo11s.pt")

# 비디오 추론
results = model.predict(
    source="data/raw/match_video.mp4",
    conf=0.25,          # 신뢰도 임계값
    iou=0.45,           # NMS IoU 임계값
    classes=[0, 32],    # person, sports ball 만 탐지
    imgsz=640,          # 입력 이미지 크기
    show=True           # 결과 표시
)
```

### 프로젝트 파이프라인 통합
```python
from tracking.detector import SoccerDetector

# 설정 파일에서 자동 로드
detector = SoccerDetector(config_path="configs/model_config.yaml")

# 또는 수동 설정
detector = SoccerDetector(
    model_path="models/yolo11s.pt",
    conf_threshold=0.25,
    classes=[0, 32]
)

# 탐지 실행
detections = detector.detect(frame)
```

### 배치 처리
```python
# 여러 프레임 동시 처리
results = model.predict(
    source="data/raw/",
    batch=8,
    save_txt=True,      # 결과를 텍스트로 저장
    save_crop=True,     # 탐지된 객체 크롭 저장
    project="runs/detect",
    name="soccer_match"
)
```

---

## ⚙️ 설정 파일 (config/model_config.yaml)

```yaml
detection:
  model_path: "models/yolo11s.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.45
  img_size: 640
  device: "cpu"  # 또는 "cuda"
  
  # 탐지할 클래스 (COCO index)
  classes: [0, 32]
  class_names:
    0: "person"
    32: "sports ball"
```

---

## 📊 성능 최적화 팁

### 1. GPU 사용 (CUDA)
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

```python
# GPU 로 모델 로드
model = YOLO("models/yolo11s.pt").to("cuda")
```

### 2. 추론 속도 향상
```python
# Half precision (FP16) 사용
results = model.predict(source="video.mp4", half=True)

# 더 작은 입력 크기 사용
results = model.predict(source="video.mp4", imgsz=416)

# 배치 사이즈 증가 (GPU 메모리 허용 시)
results = model.predict(source="video.mp4", batch=16)
```

### 3. 클래스 필터링
```python
# person 과 sports ball 만 탐지 (속도 향상)
results = model.predict(
    source="video.mp4",
    classes=[0, 32]  # 다른 클래스는 무시
)
```

---

## 🔍 문제 해결

### Model not found 오류
```bash
# 모델 재다운로드
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
mv yolo11s.pt models/
```

### Out of memory 오류
```python
# 배치 사이즈 축소
results = model.predict(source="video.mp4", batch=1)

# 입력 이미지 크기 축소
results = model.predict(source="video.mp4", imgsz=416)
```

### 느린 추론 속도
1. GPU 사용 확인 (`torch.cuda.is_available()`)
2. 경량 모델 사용 (YOLO11n)
3. 입력 이미지 크기 축소
4. 탐지 클래스 수 축소

---

## 📚 참고 자료

- **Ultralytics 문서**: https://docs.ultralytics.com/
- **YOLO11 블로그**: https://www.ultralytics.com/blog/ultralytics-yolo11
- **Roboflow Universe**: https://universe.roboflow.com/
- **COCO Dataset**: https://cocodataset.org/

---

**최종 업데이트**: 2024 년
**모델 버전**: Ultralytics YOLO11 (v8.4.150)
