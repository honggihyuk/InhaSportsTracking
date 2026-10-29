"""
RF-DETR 축구 탐지 모델 파인튜닝 (3부 21.1절 · 8부)

축구 전용 4 클래스(ball, goalkeeper, player, referee) 데이터셋으로 COCO 사전학습 RF-DETR 을 파인튜닝한다.
데이터셋은 COCO 또는 YOLO(data.yaml + train/valid/test) 형식 폴더.

    # 키 없이: Hugging Face 의 Roboflow football-players-detection 사본(CC BY 4.0)을 받아 학습
    python scripts/train_rfdetr.py --hf-dataset martinjolif/football-player-detection \
        --data-root D:/TermProject/datasets --size medium --resolution 960 --epochs 60

    # Roboflow 에서 받은 폴더 (python scripts/setup_roboflow.py, .env 에 ROBOFLOW_API_KEY)
    python scripts/train_rfdetr.py data/roboflow_datasets/football-players-detection-3zvbc

학습이 끝나면 출력 폴더의 checkpoint_best_total.pth 를 configs/model_config.yaml 의
detection.rfdetr.checkpoint 에 적는다 (클래스 이름 goalkeeper·referee 를 그대로 라벨로 사용).
같은 폴더의 train_summary.json 에 설정·소요 시간·검증/테스트 지표가 남는다.
중간 체크포인트(checkpoint_interval 에폭마다, checkpoint.pth 는 매 에폭)는 scripts/eval_rfdetr.py 로 바로 평가할 수 있다.

GPU 메모리가 부족하면 --batch 를 줄이고 --grad-accum 을 늘린다 (실효 배치 = batch × grad-accum).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.rfdetr_detector import SIZES  # noqa: E402

CLASSES = ['ball', 'goalkeeper', 'player', 'referee']


def fetch_hf(repo: str, root: Path) -> Path:
    """Hugging Face 데이터셋 저장소를 root/<이름> 에 받고 data.yaml 이 있는 폴더를 돌려준다"""
    import os
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')        # Xet 전송이 막히는 네트워크 대비
    from huggingface_hub import snapshot_download
    target = root / repo.split('/')[-1]
    snapshot_download(repo, repo_type='dataset', local_dir=str(target))
    found = sorted(target.rglob('data.yaml')) or sorted(target.rglob('_annotations.coco.json'))
    if not found:
        sys.exit(f"{target} 에서 data.yaml / COCO 주석을 찾지 못했습니다")
    return found[0].parent if found[0].name == 'data.yaml' else found[0].parent.parent


def main():
    ap = argparse.ArgumentParser(description="RF-DETR 축구 탐지 파인튜닝")
    ap.add_argument("dataset", nargs='?', help="데이터셋 폴더 (train/valid/test, COCO 또는 YOLO 형식)")
    ap.add_argument("--hf-dataset", help="Hugging Face 데이터셋 저장소 (예: martinjolif/football-player-detection)")
    ap.add_argument("--data-root", default="data/datasets", help="--hf-dataset 을 받을 폴더 (저장소 밖 권장)")
    ap.add_argument("--size", default="medium", choices=SIZES)
    ap.add_argument("--resolution", type=int, default=None,
                    help="학습·추론 해상도 (32·패치 배수). 작은 공을 위해 기본값(576 등)보다 높게")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15, help="조기 종료 (검증 mAP 가 이 에폭 동안 안 오르면). 0 이면 끔")
    ap.add_argument("--checkpoint-interval", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", default=None, help="이어서 학습할 checkpoint.pth")
    ap.add_argument("--output", default="models/rfdetr_football")
    args = ap.parse_args()

    if args.hf_dataset:
        dataset = fetch_hf(args.hf_dataset, Path(args.data_root))
    elif args.dataset:
        dataset = Path(args.dataset)
    else:
        sys.exit("데이터셋 폴더 또는 --hf-dataset 을 지정하세요")
    if not dataset.is_dir():
        sys.exit(f"데이터셋 폴더가 없습니다: {dataset}  (python scripts/setup_roboflow.py 로 먼저 받으세요)")

    import rfdetr
    import torch
    kwargs = {'resolution': args.resolution} if args.resolution else {}
    model = getattr(rfdetr, f"RFDETR{args.size.capitalize()}")(**kwargs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train(dataset_dir=str(dataset), epochs=args.epochs, batch_size=args.batch,
                grad_accum_steps=args.grad_accum, lr=args.lr, output_dir=str(out),
                early_stopping=args.patience > 0, early_stopping_patience=max(args.patience, 1),
                checkpoint_interval=args.checkpoint_interval, num_workers=args.workers,
                log_per_class_metrics=True, run_test=True, tensorboard=False, resume=args.resume)
    hours = (time.time() - t0) / 3600
    best = out / "checkpoint_best_total.pth"
    summary = {
        'dataset': str(dataset), 'hf_dataset': args.hf_dataset, 'size': args.size,
        'resolution': args.resolution or getattr(model.model_config, 'resolution', None),
        'epochs': args.epochs, 'batch': args.batch, 'grad_accum': args.grad_accum, 'lr': args.lr,
        'patience': args.patience, 'hours': round(hours, 2), 'checkpoint': best.as_posix(),
        'max_vram_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
    }
    (out / 'train_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n학습 완료. configs/model_config.yaml 에 다음을 적으세요:\n"
          f"  detection:\n    backend: rfdetr\n    rfdetr:\n      size: {args.size}\n"
          f"      checkpoint: {best.as_posix()}\n      resolution: {summary['resolution']}")


if __name__ == "__main__":
    main()
