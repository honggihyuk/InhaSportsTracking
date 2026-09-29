# Soccer 3D Digital Twin — 기술 및 기능 명세 (3부)

## 컴퓨터 비전 고도화 · VLM 분석 층 조사 및 설계

> **이전 문서**: [1부](./PROJECT_OVERVIEW.md) — 전체 구성·기본 기능 · [2부](./PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md) — 영상 분석·경기장 보정·3D 트윈 연동
>
> 이 문서는 객체 인식과 움직임 분석을 고도화하고, 그 결과 위에 **VLM(Vision-Language Model) 분석 기능**을 올리기 위한 기술 조사와 설계를 정리합니다. 아직 구현 전 단계의 **조사·설계 문서**이며, 절 번호는 2부(7~18절)에 이어 19절부터 시작합니다.
>
> 작성 기준: 2026-09-29
>
> **원칙**: 유료 AI API(OpenAI·Anthropic·Google 등)는 사용하지 않고 **Hugging Face 오픈 모델 + 로컬 추론**으로 구현합니다. 유료 API 연동은 [25절](#25-향후-가능성--유료-api-연동)에 **미래 가능성**으로만 명시합니다.

---

## 목차

19. [조사 요약과 결론](#19-조사-요약과-결론)
20. [현재 구현의 한계](#20-현재-구현의-한계)
21. [제안 기술 스택](#21-제안-기술-스택)
22. [VLM 분석 층 설계](#22-vlm-분석-층-설계)
23. [하드웨어 요구와 실행 모드](#23-하드웨어-요구와-실행-모드)
24. [구현 로드맵](#24-구현-로드맵)
25. [향후 가능성 — 유료 API 연동](#25-향후-가능성--유료-api-연동)
26. [라이선스 정리](#26-라이선스-정리)
27. [참고 자료](#27-참고-자료)

---

## 19. 조사 요약과 결론

**결론: 고도화 효과가 있습니다.** 다만 효과의 대부분은 탐지·추적·팀 분류·등번호 인식이 만드는 **데이터 품질**에서 나오고, VLM 은 그 데이터를 해석하고 질의에 답하는 **상위 층**입니다.

| 판단 근거 | 내용 |
|---|---|
| 범용 VLM 단독은 약함 | SoccerBench(약 1 만 문항, 13 개 과제)에서 GPT-4o·Gemini 급 범용 모델도 축구 영상 QA 정확도 **30~62 %** |
| 도구 결합형 에이전트가 강함 | 탐지·등번호 인식·OCR 등 18 개 도구를 호출하는 SoccerAgent 가 텍스트 QA **82~86 %** (상용 API 단독 56~68 %) |
| 트윈의 정확도 = 데이터 정확도 | 3D 트윈·통계·VLM 답변 모두 좌표·팀·신원 데이터에 의존 → 하위 CV 품질 개선이 선행되어야 함 |

따라서 **CV 파이프라인을 "도구"로 만들고, 로컬 VLM/LLM 이 그 도구를 호출해 답하는 구조**를 채택합니다. VLM 이 좌표나 속도를 직접 추정하지 않고, 수치는 CV 데이터에서 가져와 해석·설명만 맡기므로 환각이 줄고 3D 트윈과 결과가 일치합니다.

**가장 큰 제약은 하드웨어**입니다. 현재 환경(CPU 5 코어, GPU 없음, YOLO11n 약 1 초/프레임)에서는 SAM3·VLM 을 실시간으로 돌릴 수 없으므로, **실시간 모드와 정밀 분석 모드를 분리**합니다([23절](#23-하드웨어-요구와-실행-모드)).

---

## 20. 현재 구현의 한계

2부 실측 결과([14절](./PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md#14-실측-결과), [18절](./PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md#18-남은-제약과-다음-단계))와 코드 기준입니다.

| 모듈 | 현재 구현 | 한계 |
|---|---|---|
| 탐지 `tracking/detector.py` | YOLO11 COCO — `person`, `sports ball` | 공은 19 % 프레임에서만 검출, 심판·골키퍼 구분 불가 |
| 추적 `tracking/tracker.py` | BoT-SORT + 카메라 움직임 보정, `BallTracker` | 선수끼리 겹칠 때 ID 스위치 |
| 팀 분류 `pipeline/video_analysis.py` `cluster_teams()` | 상체 평균 Lab 색 + `cv2.kmeans(k=2)` | 조명·그림자·비슷한 유니폼 색에 약함, 골키퍼·심판은 `other` |
| 신원 | 없음 (트랙 ID 만 있음) | 선수 이름·등번호와 연결 불가, 트랙이 끊기면 다른 선수로 취급 |
| 경기장 보정 | 수동 기준점 4 개 이상 | 키프레임에서 멀수록 오차 누적 |
| 분석 인터페이스 | 통계 패널 | 자연어 질의·해설·요약 없음 |

> `cluster_teams()` 주석에도 "정확도가 필요하면 SigLIP 임베딩 + 위치 기반 골키퍼 보정으로 확장" 이 이미 향후 방향으로 적혀 있습니다.

---

## 21. 제안 기술 스택

| 역할 | 모델 | 대체 대상 | 기대 효과 | 비용 |
|---|---|---|---|---|
| 탐지 | **RF-DETR** | YOLO11 COCO | 공·심판·골키퍼·등번호 영역 탐지 | 낮음 (실시간급) |
| 추적·분할 | **SAM3** | BoT-SORT (정밀 모드만) | 가림에 강한 추적, 마스크 기반 지면 좌표 | **매우 높음** |
| 팀 분류 | **SigLIP2 + UMAP + K-Means** | 평균색 + k-means | 조명·색 유사성에 강한 팀 분류 | 낮음 |
| 등번호 인식 | **GLM-OCR** | (신규) | 등번호 + 팀 → 선수 신원 | 중간 |
| 추론·질의 | **Qwen3-VL** + LangChain/LangGraph | (신규) | 자연어 질의·해설·요약 | 중간~높음 |

### 21.1 RF-DETR — 탐지

- **패키지**: `pip install rfdetr` (1.11.0, 2026-09-24), Python ≥ 3.10, `supervision` 연동
- **라이선스**: Nano~Large 는 **Apache 2.0**, XL/2XL 은 PML 1.0 (`rfdetr_plus`)
- **구조**: DINOv2 백본 기반 실시간 DETR. 탐지·인스턴스 분할·키포인트(preview)를 같은 API 로 지원

| 크기 | COCO AP50:95 | 지연 (ms) | 파라미터 |
|---|---|---|---|
| Nano | 48.4 | 2.3 | 30.5 M |
| Small | 53.0 | 3.5 | 32.1 M |
| Medium | 54.7 | 4.4 | 33.7 M |
| Large | 56.5 | 6.8 | 33.9 M |

**적용 방법**

1. 이미 받아 둔 Roboflow `football-players-detection`, `football-ball-detection` 데이터셋으로 **파인튜닝** (GPU 필요 — Colab T4 로도 가능)
2. 클래스 구성: `player`, `goalkeeper`, `referee`, `ball`, `number`
   - 참고한 Roboflow 예제는 농구용이라 `basket` 클래스가 있지만 축구에는 불필요 → `goalkeeper`, `referee` 로 대체
3. `detector.py` 의 `SoccerDetector` 에 RF-DETR 백엔드를 추가하고 `configs/model_config.yaml` 에서 선택
4. 키포인트 모드로 **경기장 키포인트**를 학습하면 수동 보정을 **자동 보정**으로 대체 가능 (2부 18.2 의 1 순위 과제와 연결)

```python
from rfdetr import RFDETRSmall
model = RFDETRSmall()
model.train(dataset_dir="data/roboflow_datasets/football-players-detection-3zvbc",
            epochs=50, batch_size=8)
detections = model.predict(frame, threshold=0.4)   # supervision.Detections
```

### 21.2 SAM3 — 추적·분할

- **모델**: `facebook/sam3` (Hugging Face 접근 승인 필요), Transformers `Sam3VideoModel` / `Sam3VideoProcessor`
- **기능**: 텍스트 프롬프트(`"soccer player"`, `"ball"`)로 영상 내 객체를 탐지·분할하고 ID 를 유지. 스트리밍(프레임 단위) 추론 지원
- **속도**: SAM3.1 이 H100 에서 약 32 FPS (SAM3 약 16 FPS). 더 가벼운 SAM2 를 쓴 Roboflow 농구 파이프라인도 T4 에서 **1~2 FPS** 였고, 병목이 SAM 이었음
- **라이선스**: **SAM License** (Apache 아님) — 상업화 시 조건 확인 필요

```python
from transformers import Sam3VideoModel, Sam3VideoProcessor
model = Sam3VideoModel.from_pretrained("facebook/sam3", device_map="auto")
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
session = processor.init_video_session(video=frames, inference_device="cuda")
session = processor.add_text_prompt(session, ["soccer player", "ball"])
for out in model.propagate_in_video_iterator(inference_session=session):
    res = processor.postprocess_outputs(session, out)   # object_ids, boxes, masks
```

**적용 방침**

- CPU 환경에서는 사실상 불가 → **메인 추적은 BoT-SORT 유지**
- GPU 가 있을 때 **정밀 재분석 모드**로 사용: RF-DETR 박스를 프롬프트로 넣어 추적
- 마스크의 최하단 픽셀을 발 위치로 쓰면 현재의 박스 하단 중앙(`ground_point()`)보다 지면 좌표가 정확해짐

### 21.3 SigLIP2 + UMAP + K-Means — 팀 분류

`cluster_teams()` 를 거의 그대로 교체할 수 있어 **효과 대비 비용이 가장 작습니다.**

```
트랙별 선수 크롭 샘플링 (예: 트랙당 5~10 장, 상체 중심)
   → SigLIP2 이미지 임베딩 (google/siglip2-base-patch16-224)
   → UMAP 3차원 축소
   → K-Means (k=2)
   → 트랙 단위 다수결 → home / away
```

- 매 프레임이 아니라 **트랙별 소수 샘플**만 임베딩하므로 CPU 에서도 가능
- 색뿐 아니라 무늬·질감·의미 정보까지 반영해 조명 변화에 강함
- **골키퍼**: 유니폼이 달라 군집에서 빠지므로, 경기장 좌표에서 가까운 팀 중심에 배정 (roboflow/sports 방식)
- **심판**: RF-DETR 의 `referee` 클래스로 먼저 분리해 군집에서 제외
- 결과 형식(`teams`, `team_colors`)은 그대로 유지 → 프론트엔드 수정 불필요

### 21.4 GLM-OCR — 등번호 인식

- **모델**: `zai-org/GLM-OCR` (0.9 B, CogViT 인코더 + GLM-0.5B 디코더), Transformers `GlmOcrForConditionalGeneration`
- **특성**: 문서 OCR 에 최적화된 모델 → 작고 흐린 등번호 크롭에서는 **제로샷 정확도가 제한적일 가능성이 높음**
- **참고 수치 (Roboflow 농구 실험)**: 파인튜닝한 SmolVLM2 86 %, ResNet-32 분류기 93 %

**적용 방침**

1. RF-DETR 의 `number` 박스와 선수 박스(또는 SAM3 마스크)를 **IoS**(Intersection over Smaller area)로 매칭
2. GLM-OCR 로 번호를 읽고, **트랙 전체에서 투표**해 확정 (프레임 단위 결과는 신뢰하지 않음)
3. `(팀, 등번호)` → 경기 명단과 연결해 **영구 신원** 부여, 트랙이 끊겨도 같은 선수로 재연결
4. 정확도가 부족하면 GLM-OCR 결과를 라벨링 보조로 써서 **경량 분류기(0~99)** 를 학습해 교체
5. 와이드 중계 화면에서는 번호가 보이는 프레임 자체가 적다는 점을 전제로 설계

```python
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")
model = GlmOcrForConditionalGeneration.from_pretrained("zai-org/GLM-OCR", device_map="auto")
messages = [{"role": "user", "content": [
    {"type": "image", "image": number_crop},
    {"type": "text", "text": "Text Recognition:"}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
text = processor.decode(model.generate(**inputs, max_new_tokens=8)[0], skip_special_tokens=True)
```

---

## 22. VLM 분석 층 설계

### 22.1 전체 구조

```
[CV 파이프라인 = 도구]                                   [추론 층]
RF-DETR 탐지 → BoT-SORT(/SAM3) 추적 → SigLIP2 팀 → 등번호 신원
      → 호모그래피 경기장 좌표 → 이벤트 추출 (점유·패스·스프린트)
                        │
                        ▼  구조화 데이터 (분석 JSON → SQLite)
        ┌──────────────────────────────────────────┐
        │ LangGraph 에이전트 (로컬 Qwen3-VL)        │
        │   tools: query_tracks, get_possession,    │
        │          find_events, player_stats,       │
        │          get_frame, crop_player           │
        └──────────────────────────────────────────┘
                        │
                        ▼
   자연어 질의 · 자동 해설 · 하이라이트 요약 · 이벤트 검증
                        │
                        ▼
   Web: 답변 + 3D 트윈 타임라인 이동 (해당 프레임으로 이동)
```

### 22.2 모델 — Qwen3-VL

| 항목 | 내용 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-Instruct`, `-8B-Instruct` |
| 라이선스 | Apache 2.0 |
| 입력 | 텍스트 + 이미지 + 영상 |
| 용도 | 도구 호출·결과 해석, 필요할 때 특정 프레임·선수 크롭을 직접 "보고" 판단 |
| 권장 | 개발·CPU 테스트는 2B, GPU 운영은 4B~8B |

### 22.3 서빙과 LangChain 연결

- **서빙**: vLLM 또는 Ollama 로 **OpenAI 호환 서버**를 로컬에 띄움
- **연결**: LangChain `ChatOpenAI(base_url="http://localhost:8001/v1")` + LangGraph 에이전트
- **주의**: `ChatHuggingFace` + 로컬 `HuggingFacePipeline` 조합은 **tool call 을 파싱하지 않는** 알려진 문제가 있어 사용하지 않음
- 이 구조 덕분에 [25절](#25-향후-가능성--유료-api-연동)의 유료 API 전환은 `base_url` 과 모델명만 바꾸면 됨

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(base_url="http://localhost:8001/v1", api_key="local",
                 model="Qwen/Qwen3-VL-4B-Instruct")

@tool
def player_stats(team: str, number: int, start_s: float, end_s: float) -> dict:
    """구간 내 선수의 이동 거리·최고 속도·스프린트 횟수를 반환"""
    ...

agent = create_react_agent(llm, tools=[player_stats, ...])
agent.invoke({"messages": [("user", "후반 10분 동안 home 7번의 스프린트 구간은?")]})
```

### 22.4 도구 목록 (초안)

| 도구 | 입력 | 출력 | 데이터 출처 |
|---|---|---|---|
| `query_tracks` | 시간 구간, 팀, 등번호 | 경기장 좌표 궤적 | `world` 필드 |
| `player_stats` | 선수, 구간 | 거리·평균/최고 속도·스프린트 | 궤적 계산 |
| `get_possession` | 구간 | 팀별 점유율, 점유 선수 | 공-선수 거리 |
| `find_events` | 이벤트 종류, 구간 | 패스·슈팅·스프린트 시점 목록 | 이벤트 추출 |
| `team_shape` | 팀, 시점 | 수비 라인 높이, 폭, 대형 | 선수 좌표 |
| `get_frame` | 프레임 번호 | 이미지 (VLM 입력) | 원본 영상 |
| `crop_player` | 트랙 ID, 프레임 | 선수 크롭 이미지 | 탐지 박스 |

### 22.5 설계 원칙

1. **수치는 도구가, 해석은 VLM 이** — VLM 에게 좌표·속도를 추정시키지 않음
2. **답변에 근거 프레임 포함** — 프론트엔드가 해당 프레임으로 영상·3D 트윈을 이동
3. **영상 이해는 선택적으로** — 전체 영상을 VLM 에 넣지 않고, 도구가 고른 프레임만 전달
4. **데이터 저장소** — 분석 JSON(`uploads/analysis/*.json`)을 SQLite 로 적재해 도구가 SQL 로 조회

### 22.6 활용 예

- "후반 A 팀 수비 라인 평균 높이는?"
- "7 번 선수의 스프린트 구간을 보여줘" → 3D 트윈 타임라인 이동
- "이 장면에서 누가 공을 잃었어?" → `get_possession` + `get_frame` 으로 확인
- 경기·구간 자동 요약, 하이라이트 후보 추출

---

## 23. 하드웨어 요구와 실행 모드

### 23.1 모델별 실행 가능성 (추정)

| 모델 | CPU (현재 환경) | GPU 8 GB | GPU 12 GB 이상 |
|---|---|---|---|
| RF-DETR 추론 | 가능 (느림) | 실시간급 | 실시간급 |
| RF-DETR 파인튜닝 | 비현실적 | 가능 | 가능 |
| SigLIP2 팀 분류 | **가능** (트랙 샘플만) | 가능 | 가능 |
| GLM-OCR 0.9B | 가능 (샘플 크롭만, 느림) | 가능 | 가능 |
| SAM3 | **불가** | 제한적 | 오프라인 분석 가능 |
| Qwen3-VL-2B | 느리지만 가능 | 가능 | 가능 |
| Qwen3-VL-4B/8B | 비현실적 | 4B 양자화 | 가능 |

> 모델 크기 기준 추정이며, 실측 후 갱신합니다. **RTX 3060 12 GB 이상급**이면 21~22절 전체를 오프라인 분석으로 운영할 수 있을 것으로 봅니다.

### 23.2 실행 모드 분리

| 모드 | 구성 | 용도 |
|---|---|---|
| **실시간** | RF-DETR + BoT-SORT + (미리 계산한) 팀 분류 | 라이브 스트림, 3D 트윈 실시간 표시 |
| **정밀 분석** | RF-DETR + SAM3 + SigLIP2 + GLM-OCR | 업로드 영상 백그라운드 분석 (현재 분석 작업 흐름에 통합) |
| **질의** | LangGraph + Qwen3-VL | 분석 완료된 경기에 대한 질의·요약 |

`configs/model_config.yaml` 에 `analysis.mode: realtime | precise` 를 추가하는 방식을 제안합니다.

---

## 24. 구현 로드맵

2부 [18.2](./PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md#182-다음-단계-우선순위순)의 다음 단계와 합친 우선순위입니다.

| 순서 | 작업 | 효과 | 필요 자원 | 변경 파일 |
|---|---|---|---|---|
| 1 | **SigLIP2 팀 분류** | 팀 분류 정확도 향상, `other` 감소 | CPU 가능 | `pipeline/video_analysis.py`, `requirements.txt` |
| 2 | **RF-DETR 파인튜닝·교체** | 공·심판·골키퍼 탐지 | GPU (Colab 가능) | `tracking/detector.py`, `configs/model_config.yaml` |
| 3 | **경기장 키포인트 → 자동 보정** | 수동 보정 제거, 누적 오차 해결 | GPU (학습) | `pipeline/video_analysis.py` (`frame_homographies()` 입력) |
| 4 | **등번호 인식·신원 부여** | 선수 이름·번호 연결, 트랙 재연결 | CPU 가능 (느림) | 신규 `tracking/identity.py` |
| 5 | **분석 데이터 SQLite 적재 + 이벤트 추출** | VLM 도구의 기반 | CPU | 신규 `analysis/store.py`, `analysis/events.py` |
| 6 | **LangGraph + 로컬 Qwen3-VL 질의** | 자연어 질의·요약 | GPU 권장 (2B 는 CPU 가능) | 신규 `agent/`, `backend/server.py` (질의 API), 프론트엔드 질의 패널 |
| 7 | **SAM3 정밀 모드** | 가림 구간 추적, 지면 좌표 정밀화 | GPU 12 GB 이상 | `tracking/tracker.py` |

**추가 의존성 (예정)**

```
rfdetr
supervision
transformers>=5        # Sam3VideoModel, GlmOcrForConditionalGeneration, SigLIP2
umap-learn
scikit-learn
langchain
langchain-openai
langgraph
# 서빙: vllm (Linux/GPU) 또는 ollama
```

> 현재 `requirements.txt` 는 NumPy 1.26.4·CPU 전용 PyTorch 에 맞춰져 있습니다. Transformers 최신판과 버전이 충돌할 수 있으므로, VLM·SAM3 는 **별도 가상환경 또는 별도 서비스 프로세스**로 분리하는 것을 권장합니다.

---

## 25. 향후 가능성 — 유료 API 연동

> 이 절은 **현재 작업 범위가 아닙니다.** 설계상 열어 둔 확장 지점만 기록합니다.

- LangChain 의 모델 객체만 교체하면 동일한 도구·에이전트를 유료 모델(Claude, GPT, Gemini 등)로 실행 가능
- 예상 용도: 로컬 모델로 부족한 긴 경기 요약, 다국어 해설, 복잡한 전술 추론
- 전환 시 고려사항: 비용, 영상 프레임 외부 전송에 따른 저작권·개인정보, 응답 지연
- 권장 방식: 로컬 모델을 기본으로 두고, 설정(`agent.provider: local | api`)으로 선택

---

## 26. 라이선스 정리

| 구성 | 라이선스 | 비고 |
|---|---|---|
| RF-DETR (N~L) | Apache 2.0 | XL/2XL 은 PML 1.0 |
| SAM3 / SAM3.1 | SAM License | HF 접근 승인 필요, 상업 이용 조건 확인 |
| SigLIP2 | Apache 2.0 | |
| GLM-OCR | 모델 카드 확인 필요 | |
| Qwen3-VL | Apache 2.0 | |
| LangChain / LangGraph | MIT | |
| Roboflow 데이터셋 | 데이터셋별 확인 | Universe 공개 데이터셋 |

> 라이선스는 변경될 수 있으므로 도입 시점에 각 모델 카드를 다시 확인합니다.

---

## 27. 참고 자료

- [How to Detect, Track, and Identify Basketball Players (Roboflow)](https://blog.roboflow.com/identify-basketball-players/) — RF-DETR + SAM2 + SigLIP/UMAP/K-Means + 등번호 인식 파이프라인
- [Jersey Number Recognition (Roboflow)](https://blog.roboflow.com/jersey-number-recognition-for-sports/)
- [roboflow/sports](https://github.com/roboflow/sports) — 축구 팀 분류·레이더 뷰 참고 구현
- [rfdetr · PyPI](https://pypi.org/project/rfdetr/)
- [SAM3 Video — Transformers docs](https://huggingface.co/docs/transformers/model_doc/sam3_video)
- [SAM 3.1 vs SAM 3 (Labellerr)](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/)
- [GLM-OCR — Transformers docs](https://huggingface.co/docs/transformers/main/en/model_doc/glm_ocr)
- [Multi-Agent System for Comprehensive Soccer Understanding (SoccerBench / SoccerAgent)](https://arxiv.org/pdf/2505.03735)
- [MSUE: Multi-Modal Soccer Understanding Expert](https://arxiv.org/pdf/2606.12106)
- [Qwen3-VL collection (Hugging Face)](https://huggingface.co/collections/Qwen/qwen3-vl)
- [LangChain — Hugging Face integration](https://docs.langchain.com/oss/python/integrations/chat/huggingface)
- [ChatHuggingFace tool-call parsing issue (LangChain forum)](https://forum.langchain.com/t/chathuggingface-huggingfacepipeline-code-never-parses-tool-call-code/2117)
