"""
RF-DETR 축구 탐지 모델 파인튜닝 (3부 21.1절)

Roboflow 축구 데이터셋(선수·골키퍼·심판·공)을 COCO 또는 YOLO 형식으로 받아 학습한다.
학습이 끝나면 출력 폴더의 checkpoint_best_total.pth 를 configs/model_config.yaml 의
detection.rfdetr.checkpoint 에 적으면 분석에 쓰인다 (클래스 이름 goalkeeper·referee 를 그대로 라벨로 사용).

    python scripts/setup_roboflow.py                       # 데이터셋 다운로드 (.env 에 ROBOFLOW_API_KEY)
    python scripts/train_rfdetr.py data/roboflow_datasets/football-players-detection-3zvbc --epochs 50

GPU 메모리가 부족하면 --batch 를 줄이고 --grad-accum 을 늘린다 (실효 배치 = batch × grad-accum).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.rfdetr_detector import SIZES


def main():
    ap = argparse.ArgumentParser(description="RF-DETR 축구 탐지 파인튜닝")
    ap.add_argument("dataset", help="Roboflow 데이터셋 폴더 (train/valid/test, COCO 또는 YOLO 형식)")
    ap.add_argument("--size", default="medium", choices=SIZES)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output", default="models/rfdetr_football")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"데이터셋 폴더가 없습니다: {dataset}  (python scripts/setup_roboflow.py 로 먼저 받으세요)")

    import rfdetr
    model = getattr(rfdetr, f"RFDETR{args.size.capitalize()}")()
    model.train(dataset_dir=str(dataset), epochs=args.epochs, batch_size=args.batch,
                grad_accum_steps=args.grad_accum, lr=args.lr, output_dir=args.output)
    best = Path(args.output) / "checkpoint_best_total.pth"
    print(f"\n학습 완료. configs/model_config.yaml 에 다음을 적으세요:\n"
          f"  detection:\n    backend: rfdetr\n    rfdetr:\n      size: {args.size}\n      checkpoint: {best.as_posix()}")


if __name__ == "__main__":
    main()
