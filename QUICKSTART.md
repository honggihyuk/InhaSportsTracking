# Soccer 3D Digital Twin - 빠른 시작 가이드

## 🚀 5 분 안에 시작하기

### 1 단계: 의존성 설치

```bash
pip install -r requirements.txt
pip install roboflow python-dotenv
```

### 2 단계: Roboflow API 키 설정 (선택사항)

Roboflow 의 축구 전문 모델을 사용하려면 API 키가 필요합니다.  
**기본 YOLO11 모델만 사용할 경우 이 단계는 건너뛰어도 됩니다.**

1. [Roboflow API 키 발급받기](https://app.roboflow.com/settings/api)
2. `.env` 파일 편집:
   ```bash
   ROBOFLOW_API_KEY=your_actual_api_key_here
   ```
3. 데이터셋 다운로드:
   ```bash
   python setup_roboflow.py
   ```

### 3 단계: 파이프라인 실행

#### 옵션 A: 기본 YOLO11 모델로 테스트 (추천 - 빠른 시작)

```bash
# configs/model_config.yaml 수정 (Roboflow 모델 경로 주석 처리)
# detector:
#   model_path: "models/yolo11s.pt"  # 이 줄 사용

python -m pipeline.main_pipeline --config configs/model_config.yaml
```

#### 옵션 B: Roboflow 전문 모델 사용

```bash
# 1. API 키 설정 후
python setup_roboflow.py

# 2. 파이프라인 실행
python -m pipeline.main_pipeline --config configs/model_config.yaml
```

#### 옵션 C: 실제 영상 처리

```bash
python -m pipeline.main_pipeline \
    --config configs/model_config.yaml \
    --video data/raw/your_soccer_video.mp4
```

### 4 단계: 결과 확인

처리된 데이터는 다음 위치에 저장됩니다:
- `data/processed/`: 처리된 프레임 데이터 (pickle)
- `trajectories.csv`: 선수/공 이동 경로 (CSV)
- `data/processed/visualization.mp4`: 시각화 비디오 (옵션)

---

## 🔧 문제 해결

### "모델 파일을 찾을 수 없습니다" 오류

**해결 방법 1: 기본 YOLO11 모델 사용**

`configs/model_config.yaml` 에서 모델 경로를 변경:

```yaml
detection:
  model_path: "models/yolo11s.pt"  # Roboflow 경로 대신 기본 모델 사용
  use_roboflow_models: false
```

**해결 방법 2: Roboflow 데이터셋 다운로드**

```bash
# 1. .env 파일에 API 키 설정
echo "ROBOFLOW_API_KEY=your_key_here" > .env

# 2. 다운로드 스크립트 실행
python setup_roboflow.py
```

### "UnicodeDecodeError" 오류

Windows 환경에서 YAML 파일 읽기 시 발생하는 인코딩 문제입니다.

**해결:** `main_pipeline.py` 의 `_load_config` 메서드가 이미 UTF-8 으로 설정되어 있습니다:

```python
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
```

### "Detection 클래스를 찾을 수 없습니다" 오류

`detector.py` 에는 `Detection` 데이터클래스가 포함되어 있지 않습니다.  
대신 Dict 형식을 사용합니다:

```python
# 올바른 사용법
from tracking.detector import RoboflowSoccerDetector

detector = RoboflowSoccerDetector()
detections = detector.detect(frame)

for det in detections:
    print(det['bbox'], det['class_name'], det['confidence'])
```

---

## 📊 사용된 Roboflow 데이터셋

프로젝트에서는 다음 3 개의 축구 전문 데이터셋을 사용합니다:

1. **[Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)**
   - 선수, 골키퍼, 심판 탐지
   - 버전: v3

2. **[Football Ball Detection](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)**
   - 축구공 탐지
   - 버전: v1

3. **[Football Field Detection](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)**
   - 경기장 라인, 영역 탐지 (호모그래피용)
   - 버전: v1

---

## 🎯 다음 단계

1. **실제 축구 영상 준비**: `data/raw/` 폴더에 영상 추가
2. **캘리브레이션**: 경기장 라인을 이용한 호모그래피 행렬 계산
3. **3DGS 렌더링**: Deformation MLP 와 Gaussian Splatting 연동
4. **웹 비주얼라이저**: Three.js 기반 Free-viewpoint 렌더링 구현

---

## 📞 도움이 필요하신가요?

- [Roboflow 문서](https://docs.roboflow.com/)
- [Ultralytics YOLO 문서](https://docs.ultralytics.com/)
- 프로젝트 이슈 트래커
