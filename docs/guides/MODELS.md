# 탐지 모델 가이드

## 1. 포함된 모델

| 모델 | 경로 | 용도 |
|---|---|---|
| **YOLO11n** | `models/yolo11n.pt` (5.4 MB) | CPU 기본값 — 선수 탐지율은 11s 와 비슷하고 약 3 배 빠름 |
| **YOLO11s** | `models/yolo11s.pt` (18.4 MB) | GPU 권장 · Roboflow 모델이 없을 때 선수 모델 대체값 |

두 모델 모두 COCO(80 클래스) 사전학습 모델입니다. 축구에는 `person`(0)과 `sports ball`(32)만 쓰며, 탐지기가 COCO 모델을 알아보고 **필요한 클래스만** 추론합니다(선수 모델은 `person`, 공 모델은 `sports ball`).

> COCO 범용 모델은 중계 화면의 10~15 px 공을 자주 놓칩니다. 고정밀 모드의 공 타일 추론이 이를 보완하지만, 정확도가 필요하면 [Roboflow 축구 전용 모델](./ROBOFLOW.md)을 권장합니다.

## 2. 설정 — `configs/model_config.yaml`

```yaml
detection:
  model_path: "models/yolo11n.pt"   # 기본 모델 (CPU). GPU 면 yolo11s 권장
  use_roboflow_models: false         # true 면 아래 전용 모델 사용
  roboflow_models:
    players: "data/roboflow_datasets/football-players-detection-3zvbc/weights/best.pt"
    ball: "data/roboflow_datasets/football-ball-detection-rejhg/weights/best.pt"
  confidence_threshold: 0.25
  img_size: 640
  device: "cpu"                      # cpu | cuda

analysis:
  mode: precise                      # 기본 분석 모드
  profiles:
    realtime: {stride: 3, img_size: 640, ball_imgsz: 960, ball_tile: null}
    precise:  {stride: 1, img_size: 960, ball_imgsz: 1280, ball_tile: 640}
```

- 경로는 프로젝트 루트 기준으로 해석됩니다 (실행 위치와 무관).
- 분석 모드별 추론 해상도·공 타일 크기는 `analysis.profiles` 가 덮어씁니다. 탐지 모델은 한 번만 로드하고 모드에 따라 추론 설정만 바꿉니다.

## 3. 코드에서 사용

```python
from tracking.detector import SoccerDetector

detector = SoccerDetector(
    players_model_path="models/yolo11n.pt",
    confidence_threshold=0.25,
    device="cpu",
    imgsz=640,          # 선수 모델 추론 해상도
    ball_imgsz=960,     # 공 모델 추론 해상도
    ball_tile=640,      # 공 타일 추론 (None 이면 끔)
)
detections = detector.detect(frame)          # List[Detection]
for d in detections:
    print(d.label, d.confidence, d.bbox, d.center, d.is_ball)
```

`Detection` 은 데이터클래스입니다 — `bbox (x1, y1, x2, y2)`, `confidence`, `class_id`(선수 0~99 · 공 100~199 · 경기장 200~), `class_name`, `type`, 계산 속성 `center`, `area`, `is_ball`, `label`.

설정 파일 기반으로 만들려면:

```python
from pipeline.main_pipeline import Soccer3DPipeline

p = Soccer3DPipeline(config_path="configs/model_config.yaml")
p.initialize_detector()
profile = p.analysis_profile("precise")       # 모드별 설정
```

## 4. 성능 (CPU 4~5 코어 실측)

| 설정 | 시간 |
|---|---|
| 빠른 미리보기 (yolo11n 640 · 공 960, 3 프레임마다) | 약 1 초/프레임 (1080p) |
| 고정밀 (yolo11n 960 · 공 1280 + 640 타일, 매 프레임) | 약 0.9 초/프레임 (720p), 1080p 는 더 느림 |

GPU 가 있으면 `device: cuda`, 선수 모델 `yolo11s` 를 권장합니다. PyTorch CUDA 판 설치는 [PyTorch 안내](https://pytorch.org/get-started/locally/)를 따릅니다.

## 5. 가중치를 다시 받으려면

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"   # 현재 폴더에 다운로드
mv yolo11s.pt models/                                           # Windows: move yolo11s.pt models\
```
