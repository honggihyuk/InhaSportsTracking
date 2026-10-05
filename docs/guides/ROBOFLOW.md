# Roboflow 축구 전용 모델 가이드 (선택)

기본 COCO 모델 대신 축구 전용 데이터셋으로 학습한 모델을 쓰면 선수·골키퍼·심판 구분과 공 탐지가 좋아집니다. 이 단계는 **선택 사항**이며, 없어도 기본 YOLO11 모델로 전체 기능이 동작합니다.

## 1. 데이터셋

| 데이터셋 | 클래스 | 용도 |
|---|---|---|
| [football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) (v3) | player, goalkeeper, referee | 선수 탐지 |
| [football-ball-detection-rejhg](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) (v1) | ball | 공 탐지 |
| [football-field-detection-f07vi](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) (v1) | 경기장 키포인트 | 경기장 보정 (향후 자동 보정 가속용) |

## 2. API 키 설정

1. https://app.roboflow.com/settings/api 에서 API 키 발급
2. 저장소 루트에 `.env` 를 만들고 키 입력:
   ```bash
   cp .env.example .env          # Windows: copy .env.example .env
   ```
   ```env
   ROBOFLOW_API_KEY=발급받은_키
   ```

> `.env` 는 `.gitignore` 에 포함되어 저장소에 올라가지 않습니다. 키가 커밋된 적이 있다면 Roboflow 대시보드에서 **키를 재발급**하세요.

## 3. 데이터셋 다운로드

```bash
pip install roboflow
python scripts/setup_roboflow.py
```

`data/roboflow_datasets/` 아래에 데이터셋별 폴더(`train/`, `valid/`, `test/`, `data.yaml`)가 생깁니다. 경기장 키포인트 데이터셋은 YOLOv11 형식을 지원하지 않아 `yolov8` 형식으로 받습니다.

Python 에서 개별로 받으려면:

```python
from scripts.download_roboflow_datasets import RoboflowDatasetDownloader
RoboflowDatasetDownloader().download_all_datasets()
```

## 4. 학습 (GPU 권장)

Roboflow 에서 받는 것은 **데이터셋**입니다. 탐지에 쓸 가중치는 학습해서 만듭니다.

```bash
yolo detect train data=data/roboflow_datasets/football-players-detection-3zvbc/data.yaml \
    model=models/yolo11s.pt epochs=100 imgsz=1280
yolo detect train data=data/roboflow_datasets/football-ball-detection-rejhg/data.yaml \
    model=models/yolo11n.pt epochs=100 imgsz=1280
```

학습이 끝나면 `runs/detect/train*/weights/best.pt` 를 설정 파일이 가리키는 위치로 복사합니다:

```
data/roboflow_datasets/football-players-detection-3zvbc/weights/best.pt
data/roboflow_datasets/football-ball-detection-rejhg/weights/best.pt
```

## 5. 사용

`configs/model_config.yaml`:

```yaml
detection:
  use_roboflow_models: true
```

서버를 다시 시작하면 전용 모델을 씁니다. 전용 모델에는 COCO 클래스 필터를 적용하지 않고, 공은 클래스 ID 100~ 범위로 구분됩니다. 가중치 파일이 없으면 기본 YOLO11 모델로 대체됩니다.

연동 점검 스크립트 (API 키·다운로드·모델 로드·탐지, 결과 이미지 `data/test_detection_result.jpg`):

```bash
python scripts/test_roboflow_integration.py
```

## 6. 문제 해결

| 오류 | 해결 |
|---|---|
| `ROBOFLOW_API_KEY 가 설정되지 않았습니다` / `Invalid API Key` | 루트 `.env` 의 키 확인 (공백 없이), 저장 후 터미널 재시작 |
| `Project not found` | 데이터셋 공개 여부·버전 번호 확인 (`.env` 의 `FOOTBALL_*_VERSION`) |
| `Rate limit exceeded` | 무료 티어 다운로드 제한 — 잠시 후 재시도 |
| 모델 파일을 찾을 수 없음 | 4절의 `best.pt` 위치 확인. 없으면 기본 YOLO11 로 자동 대체 |
