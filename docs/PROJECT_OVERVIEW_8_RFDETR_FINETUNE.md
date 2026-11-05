# 8부 — RF-DETR 축구 데이터 파인튜닝

COCO 사전학습 RF-DETR 은 공·심판·골키퍼를 구분하지 못한다. 축구 전용 4 클래스(ball, goalkeeper, player, referee)로 파인튜닝했다.

## 데이터셋
- Hugging Face `martinjolif/football-player-detection` (Roboflow `football-players-detection` 사본), **CC BY 4.0**
- 372 장 (학습 298 / 검증 49 / 테스트 25), YOLO 형식. Roboflow 원본 최신 v20 도 같은 372 장이라 HF 사본을 사용
- 클래스별 주석: player 12,063 · referee 937 · ball 396 · goalkeeper 311
- PC 저장 위치: `D:\TermProject\datasets\football-player-detection-hf` (커밋하지 않음)

## 학습
```bash
pip install "rfdetr[train]==1.11.2"
python scripts/train_rfdetr.py D:/TermProject/datasets/football-player-detection-hf/data \
    --size medium --resolution 960 --epochs 60 --batch 4 --grad-accum 4 --patience 15 \
    --output models/rfdetr_football
```
- RF-DETR Medium, 해상도 960 (작은 공 때문에 기본 576 보다 높게), 실효 배치 16, lr 1e-4, EMA
- RTX 5070 12 GB: **24 분** (60 에폭, 에폭당 약 22 초), 최대 VRAM 7.3 GB (할당 기준)
- 가중치: `models/rfdetr_football/checkpoint_best_total.pth` (PC: `D:\TermProject\InhaSportsTracking\models\rfdetr_football`, 커밋하지 않음, .gitignore)
- 5 에폭마다 체크포인트, `train_summary.json`·`metrics.csv` 에 설정·지표

## 평가 (최고 EMA 가중치)

| 클래스 | 검증 AP50:95 | 테스트 AP50:95 | 테스트 F1 | 정밀도 | 재현율 |
|---|---|---|---|---|---|
| 전체 | 0.668 (mAP50 0.916) | **0.668 (mAP50 0.934)** | 0.932 | 0.966 | 0.907 |
| ball | 0.39 | 0.344 | 0.810 | 0.944 | 0.708 |
| goalkeeper | 0.75 | 0.789 | 0.947 | 0.947 | 0.947 |
| player | 0.82 | 0.826 | 0.990 | 0.990 | 0.990 |
| referee | 0.71 | 0.715 | 0.982 | 0.982 | 0.982 |

클래스별 mAP50 은 rfdetr 로그에 따로 나오지 않는다(전체 mAP50 만). 공은 수 픽셀 크기라 AP 가 낮다 — 분석에서는 공 타일 추론(640 px)을 함께 쓴다.

## 연결
- `configs/model_config.yaml`: `detection.rfdetr.checkpoint`, `resolution`. 가중치 파일이 없으면 경고 후 COCO 사전학습으로 대체
- 골키퍼·심판 라벨 트랙은 팀 군집에서 빼고, 골키퍼는 경기장 쪽으로 팀 배정, 심판은 역할만 기록 (`track_labels`, `roles`)
- 화면: 박스 칩에 GK/REF, 심판은 노란색, 점유 판정에서 심판 제외
- 테스트: `tests/test_football_classes.py`

## sports_video_1 비교 (기존 COCO vs 파인튜닝)

`scripts/compare_detectors.py` — 같은 영상(1,054 프레임), 같은 추론 설정(정밀 모드: 선수 960, 공 1280 + 640 px 타일), 같은 추적기(BoT-SORT)로 탐지·추적만 돌렸다 (SAM·SigLIP2 없음). RTX 5070.

| 지표 | COCO (기존) | 파인튜닝 |
|---|---|---|
| 공 검출 프레임 (보간 없음) | 976 (92.6 %) | **1,012 (96.0 %)** |
| 선수 박스 수 | 12,767 (모두 person) | 13,364 |
| 라벨별 박스 | person 12,767 | player 11,929 · **referee 1,116 · goalkeeper 319** |
| 트랙 ID 수 (후처리 트랙 잇기 전) | 52 | 80 |
| 처리 속도 (탐지+추적) | 5.6 FPS | 5.7 FPS |
| 최대 VRAM (torch 할당 기준) | 0.68 GB | 0.68 GB |

- 심판·골키퍼를 탐지 단계에서 바로 구분한다. 기존 모델은 모두 person 이라 팀 분류·위치 규칙으로만 추정했다.
- 트랙 ID 가 늘었다. 라벨을 모두 같은 클래스로 바꿔 추적해도 80 개로 같아, 라벨이 바뀌며 트랙이 쪼개진 것은 아니다. 새 모델이 화면 가장자리·겹친 선수 박스를 더 많이 내는 영향으로 보이며, 정밀 모드는 이후 트랙 잇기(stitch)로 줄인다.
- 속도·메모리는 같다 (같은 Medium 구조, 추론 해상도는 프로필이 정함).

비교 이미지 (`docs/report_screenshots/rfdetr/`, 왼쪽 COCO · 오른쪽 파인튜닝, GK = 초록, REF 표시, 공 = 원):
`frame_0120_04.0s.png`, `frame_0400_13.3s.png`, `frame_0600_20.0s.png` (골키퍼 GK, COCO 의 골키퍼 발 공 오탐 사라짐), `frame_0880_29.3s.png` (심판 REF), `frame_1000_33.3s.png`

## 환경 메모
- 학습에는 `pip install "rfdetr[train]==1.11.2"` 가 필요하다.
- `roboflow` 패키지를 설치하면 `opencv-python-headless 5.x` 가 함께 깔려 `opencv-python 4.14` 의 cv2 를 덮어쓰고, 광류 결과가 달라져 보정 테스트가 실패한다. 설치했다면 `pip uninstall opencv-python-headless` 후 `pip install --force-reinstall --no-deps opencv-python==4.14.0.94`.
