# 설치 · 서버 실행 가이드

> 이 문서의 모든 명령은 **저장소 루트(`InhaSportsTracking/`)** 에서 실행합니다. 프론트엔드 명령만 `frontend/` 에서 실행합니다.
> 설정 파일·모델의 상대 경로(`configs/…`, `models/…`, `data/…`)는 실행 위치에 없으면 프로젝트 루트 기준으로 찾으므로, 다른 폴더에서 실행해도 동작합니다.

## 1. 요구 사항

| 항목 | 버전 |
|---|---|
| Python | 3.11 ~ 3.12 |
| Node.js | 20 이상 (npm 포함) |
| ffmpeg | 선택 — 브라우저가 재생하지 못하는 영상 변환용 |
| GPU | 선택 — 없으면 CPU 로 동작 (`configs/model_config.yaml` 의 `detection.device: cpu`) |

## 2. 설치

```bash
# 가상환경 (권장)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Python 의존성
pip install -r requirements.txt

# 프론트엔드 의존성
cd frontend
npm install
cd ..
```

- GPU 가 없는 PC 에서 PyTorch 가 오류(Bus error 등)를 내면 CPU 전용판을 설치합니다:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- YOLO 가중치 `models/yolo11n.pt`, `models/yolo11s.pt` 는 저장소에 포함되어 있습니다 ([모델 가이드](./MODELS.md)).
- Roboflow 축구 전용 모델은 선택 사항입니다 ([Roboflow 가이드](./ROBOFLOW.md)).

## 3. 서버 실행

터미널 두 개를 엽니다.

**백엔드 (FastAPI, 포트 8000)** — 저장소 루트에서:

```bash
python -m uvicorn backend.server:app --port 8000
# 개발 중 코드 변경 시 자동 재시작: --reload 추가
```

**프론트엔드 (Vite, 포트 5173)**:

```bash
cd frontend
npm run dev
```

브라우저에서 `http://localhost:5173` 을 엽니다. 프론트엔드는 백엔드를 `http://localhost:8000` 으로 찾습니다 (`frontend/src/components/api.js` 의 `API_BASE_URL`).

| 확인 | 주소 |
|---|---|
| 백엔드 상태 | `http://localhost:8000/health` |
| API 문서 (Swagger) | `http://localhost:8000/docs` |
| 웹 화면 | `http://localhost:5173` |

### 실행 중 생기는 폴더

| 경로 | 내용 | git |
|---|---|---|
| `uploads/` | 업로드한 경기 영상 | 제외 |
| `uploads/analysis/{영상}.json` | 분석 결과 (박스·카메라 움직임·팀·보정·경기 지표) | 제외 |
| `uploads/analysis/{영상}.lines.npz` | 프레임별 경기장 라인 마스크 (고정밀 분석·자동 보정) | 제외 |

## 4. 사용 흐름

1. 오른쪽 **경기 영상 라이브러리**에서 영상을 업로드하고 선택
2. **영상 분석** 카드에서 모드(**고정밀** / 빠른 미리보기) 선택 → **분석 시작**
3. 고정밀 분석은 끝나면 **자동으로 경기장을 보정**하고 경기 지표를 만듭니다. 빠른 미리보기 결과는 **자동 보정** 버튼, 자동 보정이 안 되는 구간은 **기준점 직접 지정**
4. 영상·탐지 박스·3D 트윈·경기 지표가 같은 타임라인으로 재생

브라우저가 재생할 수 있는 **H.264(MP4)** 또는 **VP9/AV1(WebM)** 영상을 사용하세요:

```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -movflags +faststart output.mp4
```

## 5. 그 밖의 실행

```bash
# 테스트 (YOLO·GPU 불필요)
python -m pytest tests -q

# 실시간 vs 고정밀 vs 자동 보정 정량 비교 (정답이 있는 합성 중계 영상)
python scripts/benchmark_precise.py --frames 300

# 파이프라인 CLI (테스트용 — 앞 50 프레임만 처리, 결과는 data/processed/)
python -m pipeline.main_pipeline --config configs/model_config.yaml --video data/raw/sample_video.mp4
```

자세한 내용은 [테스트 가이드](./TESTING.md).

## 6. 문제 해결

| 증상 | 원인 · 해결 |
|---|---|
| 웹 화면에 "트래킹 서버에 연결할 수 없습니다" | 백엔드가 꺼져 있거나 8000 이 아닌 포트로 실행됨 → 3절 명령으로 실행 |
| 영상이 재생되지 않음 | 브라우저가 지원하지 않는 코덱 → 4절 ffmpeg 변환 |
| 분석 중 브라우저가 느림 | CPU 추론과 3D 렌더링 경합. 백엔드는 코어 2 개를 남기도록 추론 스레드를 제한함 |
| `UnicodeDecodeError` (Windows, 설정 파일) | 설정 파일은 UTF-8 로 읽음 — 편집기에서 UTF-8 로 저장 |
| Windows 콘솔에서 한글·이모지 출력 오류 | 서버는 출력 불가 문자를 대체하도록 설정됨. 그래도 깨지면 `set PYTHONUTF8=1` |
| `ModuleNotFoundError: tracking` 등 | 저장소 루트에서 `python -m …` 형식으로 실행 (스크립트는 `python scripts/…py`) |
