"""
탐지 모델 비교 — 같은 영상·같은 추론 설정·같은 추적기로 두 RF-DETR 가중치를 돌려 비교 (8부)

  python scripts/compare_detectors.py data/samples/sports_video_1.mp4 \
      --a coco --b models/rfdetr_football/checkpoint_best_total.pth --b-resolution 960 \
      --out C:/tmp/compare --images docs/report_screenshots/rfdetr --frames 120,400,600,880,1000

지표 (모델마다):
  - 공 검출 프레임 수 (보간 없이 탐지기가 실제로 공을 낸 프레임, 잔디 밖·길쭉한 오탐 제외)
  - 선수 박스 수 · 라벨별 박스 수 (player / goalkeeper / referee, COCO 모델은 person)
  - 트랙 ID 수 (분석과 같은 추적기 설정)
  - 처리 속도 (탐지+추적, 프레임/초), 최대 VRAM (torch.cuda.max_memory_allocated)
분석 모드 프로필(기본 precise: 선수 960, 공 1280 + 640 타일)과 같은 추론 설정을 쓴다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import video_analysis as va  # noqa: E402
from pipeline.main_pipeline import Soccer3DPipeline as Pipeline  # noqa: E402
from tracking.rfdetr_detector import RFDETRDetector  # noqa: E402
from tracking.tracker import BallTracker  # noqa: E402

COLORS = {'player': (80, 200, 255), 'person': (80, 200, 255), 'goalkeeper': (60, 230, 60),
          'referee': (0, 215, 255), 'ball': (60, 255, 215)}


def run(video: Path, weights: str, resolution, profile: dict, frames_to_keep, max_frames=None, size='medium'):
    import torch
    p = Pipeline(config_path=str(ROOT / 'configs' / 'model_config.yaml'))
    rf = (p.config.get('detection', {}) or {}).get('rfdetr', {}) or {}
    det = RFDETRDetector(size=size, checkpoint=None if weights == 'coco' else weights, resolution=resolution,
                         device='cuda' if torch.cuda.is_available() else 'cpu',
                         confidence_threshold=rf.get('confidence_threshold', 0.4),
                         ball_threshold=rf.get('ball_threshold', 0.25),
                         imgsz=profile.get('img_size'), ball_imgsz=profile.get('ball_imgsz'),
                         ball_tile=profile.get('ball_tile'))
    tracker = p.make_tracker()
    tracker.ball_tracker = BallTracker(max_jump=60)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ball_frames, labels, boxes, ids = 0, {}, 0, set()
    kept = {}
    t_det = 0.0
    try:
        for t in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=va.MOTION_SCALE, fy=va.MOTION_SCALE, interpolation=cv2.INTER_AREA)
            grass = va.grass_mask(small)
            t0 = time.perf_counter()
            dets = [d for d in det.detect(frame)
                    if (va.plausible_ball(d.bbox) if d.is_ball else va.on_grass(grass, d.bbox, va.MOTION_SCALE))]
            tracked = tracker.update(dets, frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_det += time.perf_counter() - t0
            ball_frames += any(d.is_ball for d in dets)
            for d in dets:
                if not d.is_ball:
                    boxes += 1
                    labels[d.label] = labels.get(d.label, 0) + 1
            ids.update(o.track_id for o in tracked if o.track_id != tracker.BALL_TRACK_ID)
            if t in frames_to_keep:
                kept[t] = [(list(map(int, d.bbox)), 'ball' if d.is_ball else d.label, round(float(d.confidence), 2))
                           for d in dets]
    finally:
        cap.release()
    n = t + 1
    return {
        'weights': weights, 'resolution': resolution, 'frames': n,
        'ball_frames': ball_frames, 'ball_frame_pct': round(100 * ball_frames / n, 1),
        'player_boxes': boxes, 'labels': labels, 'track_ids': len(ids),
        'fps': round(n / t_det, 2), 'max_vram_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        if torch.cuda.is_available() else None,
        'profile': profile,
    }, kept


def draw(frame, dets, title):
    img = frame.copy()
    for (x1, y1, x2, y2), label, conf in dets:
        col = COLORS.get(label, (200, 200, 200))
        if label == 'ball':
            cx, cy, r = (x1 + x2) // 2, (y1 + y2) // 2, max(x2 - x1, y2 - y1) // 2 + 10
            cv2.circle(img, (cx, cy), r, col, 4, cv2.LINE_AA)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        if label in ('goalkeeper', 'referee'):
            tag = 'GK' if label == 'goalkeeper' else 'REF'
            cv2.putText(img, tag, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 110), (w, h), (0, 0, 0), -1)
    cv2.putText(img, title, (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 4, cv2.LINE_AA)
    return cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=Path)
    ap.add_argument('--a', default='coco', help="기준 가중치 ('coco' = 사전학습)")
    ap.add_argument('--b', required=True, help='비교 가중치 (.pth)')
    ap.add_argument('--a-resolution', type=int, default=None)
    ap.add_argument('--b-resolution', type=int, default=None)
    ap.add_argument('--size', default='medium')
    ap.add_argument('--mode', default='precise', help='추론 설정을 가져올 분석 모드 프로필')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--frames', default='120,400,600,880,1000', help='비교 이미지로 남길 프레임')
    ap.add_argument('--images', type=Path, default=None, help='비교 이미지 저장 폴더')
    ap.add_argument('--out', type=Path, required=True, help='결과 JSON 폴더')
    a = ap.parse_args()

    profile = Pipeline(config_path=str(ROOT / 'configs' / 'model_config.yaml')).analysis_profile(a.mode)
    profile = {k: profile.get(k) for k in ('img_size', 'ball_imgsz', 'ball_tile')}
    keep = {int(x) for x in a.frames.split(',') if x}
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, w, res in (('A', a.a, a.a_resolution), ('B', a.b, a.b_resolution)):
        summary, kept = run(a.video, w, res, profile, keep, a.max_frames, a.size)
        results[key] = summary
        (a.out / f'detections_{key}.json').write_text(json.dumps({str(k): v for k, v in kept.items()}), encoding='utf-8')
        print(key, json.dumps(summary, ensure_ascii=False))
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    (a.out / 'compare.json').write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding='utf-8')

    if a.images:
        a.images.mkdir(parents=True, exist_ok=True)
        dA = json.loads((a.out / 'detections_A.json').read_text(encoding='utf-8'))
        dB = json.loads((a.out / 'detections_B.json').read_text(encoding='utf-8'))
        cap = cv2.VideoCapture(str(a.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t in sorted(keep):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if not ok or str(t) not in dA:
                continue
            panel = np.hstack([draw(frame, dA[str(t)], f'BEFORE  COCO  t={t / fps:.1f}s'),
                               draw(frame, dB[str(t)], f'AFTER  fine-tuned  t={t / fps:.1f}s')])
            path = a.images / f'frame_{t:04d}_{t / fps:04.1f}s.png'
            cv2.imwrite(str(path), panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(path)
        cap.release()


if __name__ == '__main__':
    main()
