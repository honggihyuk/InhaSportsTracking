# Soccer 3D Digital Twin — 기술 및 기능 명세 (6부)

## GPU 정밀 모델 — RF-DETR · SAM3 · SigLIP2 · Qwen3-VL 질의 에이전트

> **이전 문서**: [1부](./PROJECT_OVERVIEW.md) · [2부](./PROJECT_OVERVIEW_2_VIDEO_ANALYSIS.md) · [3부 — CV 고도화 · VLM 설계](./PROJECT_OVERVIEW_3_VLM.md) · [4부](./PROJECT_OVERVIEW_4_PRECISE_ANALYSIS.md) · [5부](./PROJECT_OVERVIEW_5_AUTO_CALIBRATION.md)
>
> 3부에서 조사·설계만 해 둔 구성 요소(21~22절)를 GPU 환경(RTX 5070 12 GB)에서 구현하고 실제 중계 영상으로 실측한 내용입니다. 절 번호는 5부(37~43절)에 이어 44절부터 시작합니다.
>
> 작성 기준: 2026-10-15

---

## 목차

44. [구성과 실행 방식](#44-구성과-실행-방식)
45. [RF-DETR 탐지 백엔드](#45-rf-detr-탐지-백엔드)
46. [SAM3 마스크 — 발 위치 정밀화](#46-sam3-마스크--발-위치-정밀화)
47. [SigLIP2 팀 분류](#47-siglip2-팀-분류)
48. [Qwen3-VL 경기 질의 에이전트](#48-qwen3-vl-경기-질의-에이전트)
49. [실측 — 실제 중계 영상](#49-실측--실제-중계-영상)
50. [설치 · 설정](#50-설치--설정)
51. [한계와 다음 단계](#51-한계와-다음-단계)

---

## 44. 구성과 실행 방식

| 역할 | 모델 | 파일 | 켜지는 조건 |
|---|---|---|---|
| 탐지 | RF-DETR Medium (COCO 사전학습, Apache 2.0) | `tracking/rfdetr_detector.py` | `detection.backend: auto` + GPU + `rfdetr` 설치 |
| 마스크 | SAM3 (`facebook/sam3`), 못 받으면 SAM2.1 Large | `tracking/sam_refiner.py` | 정밀 모드 `sam_refine: auto` + GPU |
| 팀 분류 | SigLIP2 base + UMAP + K-Means | `pipeline/team_embed.py` | 정밀 모드 `team_classifier: auto` + GPU |
| 질의 | Qwen3-VL-4B-Instruct (로컬) 또는 OpenAI 호환 서버 | `agent/` | `POST /analysis/{name}/ask` |

- **`auto` 원칙**: GPU 가 없으면 전부 예전 방식(YOLO11, 평균색 팀 분류, 박스 하단 발 위치)으로 돌아간다. CPU 환경·테스트는 그대로다. `detection.device` 도 `auto` 로 바꿔 GPU 가 있으면 `cuda` 를 쓴다.
- **결과 형식 유지**: 프레임 객체 `[id, kind, x1, y1, x2, y2, conf]` 끝에 SAM 발 위치 `[foot_x, foot_y]` 가 붙을 뿐이고, `teams`·`team_colors` 형식은 그대로라 프론트엔드·경기 지표 코드는 바뀌지 않았다. 결과에 `team_method`(분류 방식·이상치 수), `sam`(모델·발 위치 수)이 추가된다.
- **VRAM 관리**: SigLIP2·SAM 은 분석이 끝나면 내리고(`unload()`), 분석을 시작할 때 질의용 VLM 도 내린다. 12 GB 카드에서 분석(최대 2.2 GB)과 VLM(최대 11.9 GB)을 번갈아 쓸 수 있다.

## 45. RF-DETR 탐지 백엔드

`RFDETRDetector` 는 YOLO 탐지기와 같은 `detect(frame) -> List[Detection]` 인터페이스라 추적·후처리 코드는 그대로다.

- 한 모델이 선수와 공을 함께 내므로 전체 화면은 **한 번만** 추론하고, 정밀 모드에서는 공만 640 px **타일 배치 추론**(1080p 기준 8 장을 한 번에)을 더한다.
- 추론 해상도(`img_size` 960)는 패치 크기 × 윈도 수(32)의 배수로 맞춘다.
- COCO 사전학습 가중치는 `class_names` 가 80 개 목록이지만 예측 `class_id` 는 **COCO 원래 번호**(1 = person, 37 = sports ball)라서 별도로 매핑한다.
- 파인튜닝 가중치를 쓰면 클래스 이름 `goalkeeper`·`referee` 를 라벨로 그대로 쓰고, 심판 라벨이 다수인 트랙은 팀 군집에서 미리 뺀다. 학습은 `scripts/train_rfdetr.py`(Roboflow 축구 데이터셋 필요).

## 46. SAM3 마스크 — 발 위치 정밀화

추적 박스를 프롬프트로 SAM 에 넣어 프레임의 모든 선수 마스크를 한 번에 얻는다(이미지 모드, 박스 프롬프트).

- **발 위치**: 마스크 최하단 4 % 행의 x 평균과 최하단 y. 박스 하단 중앙은 다리를 벌리거나 박스가 헐거우면 어긋난다(실측 중앙값 3.2 px, 먼 선수에게는 1 m 이상).
- **이상 마스크 거르기**: 마스크 면적이 박스의 15 % 미만이거나 박스 밖으로 크게 번지면 버리고 박스 하단 중앙을 쓴다.
- **경기장 좌표**: `calibrate()` 가 발 위치가 있으면 그것을, 없으면(보간 박스 등) 박스 하단 중앙을 호모그래피에 넣는다.
- **팀 분류 크롭**: 팀 샘플을 뽑는 프레임에서는 마스크로 배경(잔디·겹친 선수)을 회색으로 지운 크롭을 SigLIP2 에 넘긴다.
- **SAM3 ↔ SAM2.1**: Transformers 의 `Sam3TrackerModel` 은 `Sam2Model` 과 같은 박스 프롬프트 API 라 모델 이름만 바꾼다. SAM3 는 Hugging Face 접근 승인 + `hf auth login` 이 필요하고, 받을 수 없으면 `gpu_models.sam_fallback`(SAM2.1)로 자동 대체한다.

> 3부 설계의 "텍스트 프롬프트 영상 추적(Sam3VideoModel)" 은 아직 쓰지 않았다. 추적은 BoT-SORT 를 유지하고 SAM 은 프레임별 정밀화에만 쓴다([51절](#51-한계와-다음-단계)).

## 47. SigLIP2 팀 분류

```
트랙별 유니폼 크롭(3 프레임마다, 트랙당 최대 16 장, SAM 마스크로 배경 제거)
  → SigLIP2 이미지 임베딩 → UMAP 3차원 → K-Means → 크롭 단위 팀 → 트랙 다수결
```

**k=3 보완**: 처음 구현(k=2)은 RF-DETR 결과에서 **"선수 전체 vs 심판·골키퍼"** 로 갈라져 43 대 5 로 실패했다. 심판·골키퍼 트랙이 5 개로 많고 유니폼이 매우 달라서다. 그래서

1. k=3 을 먼저 하고, 가장 작은 군집이 트랙 기준 25 % 이하이면 그 군집을 `other`(심판·골키퍼)로 둔다.
2. 단, 작은 군집이 한 팀 중심에 가까우면(원래 임베딩 공간 거리 < 두 팀 거리의 절반, 그늘진 같은 팀 등) 그 팀으로 합친다. UMAP 은 군집 사이의 먼 거리를 보존하지 않으므로 이 판정은 UMAP 이 아니라 원래 임베딩에서 한다.
3. 투표는 군집 번호가 아니라 **팀 이름**으로 센다(합쳐진 두 군집에 나뉜 트랙도 같은 팀).
4. 크롭이 3 장 미만이거나 다수결이 60 % 미만, 군집 중심에서 중앙값 × 3 이상 먼 트랙은 `other`.

모델 로드 실패 등 예외가 나면 평균색 분류로 대체하고 `team_method.fallback` 에 이유를 남긴다.

## 48. Qwen3-VL 경기 질의 에이전트

```
질문 → Qwen3-VL (도구 정의 포함 채팅 템플릿) ⇄ 도구 실행 → 답변 + 근거 프레임
```

- **도구 8 개** (`agent/tools.py`): `match_overview`, `player_stats`, `get_possession`, `find_events`, `team_shape`, `query_tracks`, `get_frame`(탐지 박스를 그린 프레임 이미지), `crop_player`(선수 확대 이미지). 모두 근거 프레임 번호(`frames`)를 돌려준다.
- **대화 루프** (`agent/runner.py`): LangGraph ReAct 와 같은 구조를 직접 구현했다. Qwen 의 `<tool_call>{…}</tool_call>` 을 직접 파싱한다(LangChain `ChatHuggingFace` 는 로컬 모델의 tool call 을 파싱하지 못하는 문제가 있어 3부에서 제외). 이미지 도구 결과는 다음 사용자 메시지에 이미지로 붙여 VLM 이 직접 본다.
- **환각 방지**: 시스템 프롬프트는 수치를 도구에서만 가져오고, 사용자가 말하지 않은 조건(팀·구간)은 인자에 넣지 않게 한다. 실측에서 4B 모델이 도구 없이 이전 대화만 보고 선수 번호와 속도를 지어낸 경우가 있어서, **도구 호출 없이 바로 답하면 한 번 되돌려** 도구로 확인하게 했다.
- **모델 전환**: `agent.provider: openai` 로 바꾸면 vLLM·Ollama 같은 OpenAI 호환 서버나 유료 API 를 같은 도구로 쓴다(`base_url`, `model`, `api_key_env`).
- **API**: `POST /analysis/{name}/ask` `{question, history}` → `{answer, steps, frames}`. 분석 중에는 409(GPU 사용 중).
- **화면**: 오른쪽 패널의 **경기 질의** 카드. 예시 질문 버튼, 답변의 근거 시점 버튼(누르면 영상·3D 트윈이 그 프레임으로 이동), 사용한 도구 표시.

## 49. 실측 — 실제 중계 영상

`sports_video_1.mp4` (1920×1080, 30 fps, 35 초, 1,054 프레임, 레알 마드리드–바르셀로나 중계), RTX 5070 12 GB, 고정밀 모드.

| 구성 | 분석 시간 | 선수 트랙 ID | 공 실제 검출 프레임 | 팀 분류 오류 (눈 확인) | 최대 VRAM |
|---|---|---|---|---|---|
| YOLO11s + 평균색 (기존) | 141 s | 54 | 358 (34 %) | 약 6 (심판이 home, 바르셀로나 선수 3 명이 away·other) | — |
| YOLO11s + SAM2.1 + SigLIP2 | 272 s | 54 | 358 (34 %) | 약 3 (겹쳐 가려진 트랙) | 1.6 GB |
| RF-DETR-M + SAM2.1 + SigLIP2 | 371 s | 48 | 819 (78 %) | k=2: 43 대 5 실패 → k=3 보완 후 약 2 | 1.7 GB |
| **RF-DETR-M + SAM3 + SigLIP2 (기본값)** | **458 s** | **48** | **819 (78 %)** | **약 2** | **2.2 GB** |

- 팀 분류 오류는 트랙마다 가장 큰 크롭을 모은 접촉 시트를 눈으로 확인해 셌다(정답 라벨 없음). 기본값 구성의 남은 오류는 노란 심판 1 명이 한 팀에 들어간 것과, 두 선수가 겹친 트랙 1 개다. 골키퍼는 이후 위치 기반 판정(`roles`)이 따로 표시한다.
- 공: RF-DETR 은 실제 검출(보간 제외) 프레임이 2.3 배다. 보간까지 포함하면 1,054 프레임 전부에 공 위치가 있다.
- SAM3 발 위치: 선수 박스 12,908 개 중 12,588 개(97.5 %)에 마스크 발 위치가 붙었다.
- 프레임당 속도(선수 21 명 기준): RF-DETR 0.15 s(공 타일 포함), SAM3 0.18 s, SAM2.1 0.11 s.
- 질의: Qwen3-VL-4B(bf16) 첫 질문 20~45 s, VRAM 최대 11.9 GB. 지표 질문(가장 많이 뛴 선수, 가장 빠른 스프린트)은 도구 결과와 일치했고, `crop_player` 로 등번호를 맞게 읽었다(#33 → 3 번). 장면 설명은 일부 틀린 서술이 섞인다(4B 모델 한계).

## 50. 설치 · 설정

```bash
# GPU 용 PyTorch (RTX 50 시리즈는 CUDA 12.8 빌드 필요)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# SAM3 (선택): https://huggingface.co/facebook/sam3 접근 승인 후
hf auth login
```

모델은 처음 쓸 때 자동으로 받는다(RF-DETR Medium 405 MB, SigLIP2 1.5 GB, SAM3 3.4 GB, SAM2.1 Large 0.9 GB, Qwen3-VL-4B 8.9 GB).

> Hugging Face 다운로드가 0 바이트에서 멈추면 `HF_HUB_DISABLE_XET=1` 을 설정하고 다시 받는다(Xet 전송이 막히는 네트워크).

`configs/model_config.yaml` 주요 항목:

```yaml
detection:
  backend: auto        # yolo | rfdetr | auto
  device: auto         # auto | cuda | cpu
  rfdetr: {size: medium, checkpoint: null, confidence_threshold: 0.4, ball_threshold: 0.25}
analysis:
  profiles:
    precise:
      team_classifier: auto   # siglip | color | auto
      sam_refine: auto        # true | false | auto
gpu_models:
  siglip: google/siglip2-base-patch16-224
  sam: facebook/sam3
  sam_fallback: facebook/sam2.1-hiera-large
agent:
  provider: local             # local | openai
  model: Qwen/Qwen3-VL-4B-Instruct
```

테스트: `python -m pytest tests -q` (89 개, 신규 21 개는 `tests/test_gpu_components.py`). 실제 모델은 받지 않고 같은 인터페이스의 가짜 모델로 변환·판정·대화 루프를 검증한다.

## 51. 한계와 다음 단계

1. **경기장 보정이 가장 큰 오차원**: 이 영상은 자동 보정 키프레임이 1 개뿐이고 라인 투영이 실제와 어긋나 있다. 그래서 최고 속력 9.5 m/s 초과 선수가 43 명 중 14 명이고, 점유율·거리 지표와 그 위에서 답하는 질의 결과도 믿기 어렵다. 탐지·팀 분류가 좋아져도 미터 단위 지표는 보정이 고쳐져야 맞는다(수동 기준점 지정 또는 3부 24절 3 순위의 경기장 키포인트 모델). → [7부](./PROJECT_OVERVIEW_7_PITCH_CALIBRATION.md)에서 앵커 추적으로 개선 (정상 프레임 11 % → 97 %, 최고 속력 12.5 → 9.6 m/s).
2. **RF-DETR 파인튜닝**: COCO 모델은 심판·골키퍼를 구분하지 못한다. Roboflow 축구 데이터셋으로 학습하면 심판을 군집 전에 빼서 남은 팀 오류가 줄어든다(`scripts/train_rfdetr.py`, Roboflow API 키 필요).
3. **SAM3 영상 추적**: 지금은 프레임별 마스크만 쓴다. 가려짐에 강한 ID 유지를 위해서는 `Sam3VideoModel` 스트리밍 추적으로 BoT-SORT 를 대체하는 실험이 필요하다.
4. **등번호 신원(GLM-OCR)**: 3부 21.4절은 아직 구현하지 않았다. 대신 `crop_player` 로 VLM 이 등번호를 읽을 수는 있다.
5. **VLM 크기**: 12 GB 카드는 4B 가 한계(최대 11.9 GB). 장면 해석 품질을 높이려면 8B(24 GB 이상) 또는 `provider: openai` 로 외부 모델.
