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

## 남은 일
- sports_video_1 에서 기존(COCO) vs 새 모델 비교(공 검출 프레임, 심판·골키퍼 수, 트랙 ID, FPS, VRAM)와 비교 이미지:
  `python scripts/compare_detectors.py data/samples/sports_video_1.mp4 --b models/rfdetr_football/checkpoint_best_total.pth --b-resolution 960 --out <폴더> --images docs/report_screenshots/rfdetr` (아직 실행하지 않음)
