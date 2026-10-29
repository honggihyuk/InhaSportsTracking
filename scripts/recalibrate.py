"""
분석 결과를 탐지 없이 다시 보정 — 영상을 다시 읽어 카메라 움직임(정지 오버레이 제외)·라인 마스크를 새로 만들고 자동 보정

탐지·추적 결과(frames, teams 등)는 그대로 두므로 GPU 없이 몇 분 안에 끝난다. 보정 방식이 바뀐 뒤 기존 분석 파일을
갱신하거나, 같은 탐지 결과로 보정 전후를 비교할 때 쓴다.

  python scripts/recalibrate.py data/samples/sports_video_1.mp4 data/samples/analysis/sports_video_1.mp4.json \
      --out uploads/analysis/sports_video_1.mp4.json [--every 0.5] [--keep-motion] [--keep-lines]
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
from pipeline.pitch import LineMaskStore, line_mask  # noqa: E402


def recompute(video: Path, result: dict, motion: bool = True, lines: bool = True):
    """영상을 한 번 읽어 카메라 움직임과 라인 마스크를 analyze_video 와 같은 방식으로 다시 만든다"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f'영상을 열 수 없습니다: {video}')
    n = result['frame_count']
    est = va.CameraMotion()
    store = LineMaskStore((0, 0)) if lines else None
    out_motion = []
    try:
        for t in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=va.MOTION_SCALE, fy=va.MOTION_SCALE, interpolation=cv2.INTER_AREA)
            boxes = [o[2:6] for o in result['frames'][t]] if t < len(result['frames']) else []
            if store is not None:
                store.append(*line_mask(small, boxes, box_scale=va.MOTION_SCALE))
            if motion:
                M = est(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), boxes)
                out_motion.append(None if M is None else np.round(M, 6).ravel().tolist())
    finally:
        cap.release()
    if motion:
        out_motion += [None] * (n - len(out_motion))
        result['motion'] = out_motion
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=Path)
    ap.add_argument('result', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--every', type=float, default=0.5, help='자동 키프레임 탐색 간격 (초)')
    ap.add_argument('--keep-motion', action='store_true', help='저장된 카메라 움직임을 그대로 사용')
    ap.add_argument('--keep-lines', action='store_true', help='저장된 .lines.npz 를 그대로 사용')
    a = ap.parse_args()

    result = va.load(a.result)
    lines_path = a.result.with_name(a.result.name[:-len('.json')] + '.lines.npz')
    t0 = time.perf_counter()
    store = recompute(a.video, result, motion=not a.keep_motion, lines=not a.keep_lines)
    if a.keep_lines:
        store = LineMaskStore.load(lines_path)
    t1 = time.perf_counter()
    manual = [kf for kf in (result.get('calibration') or {}).get('keyframes', []) if kf.get('source') != 'auto']
    result['calibration'] = {'keyframes': manual} if manual else None
    info = va.auto_calibrate(result, store, every_s=a.every)
    result['auto_calibration'] = {'status': 'ok', **info}
    va.save(result, a.out)
    store.save(a.out.with_name(a.out.name[:-len('.json')] + '.lines.npz'))
    print(json.dumps({'reread_s': round(t1 - t0, 1), **info,
                      **{k: v for k, v in result['calibration'].items() if k not in ('keyframes', 'confidence')}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
