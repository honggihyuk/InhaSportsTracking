"""
보정 전후 라인 투영 비교 이미지 — 같은 프레임에 두 분석 결과의 경기장 라인을 그려 나란히 저장

  python scripts/calibration_compare_images.py data/samples/sports_video_1.mp4 \
      before.json after.json docs/report_screenshots/calibration --frames 0,300,455,600,750,900,1040

각 결과 옆의 .lines.npz 로 투영 오차(px, 20 에서 자름)와 재현율을 계산해 이미지 아래에 적는다.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_calibration import frame_error  # noqa: E402
from pipeline.autocalib import AutoCalibrator, _densify  # noqa: E402
from pipeline.calib_track import orient  # noqa: E402
from pipeline.pitch import LineMaskStore, pitch_polylines, sample_pitch_lines  # noqa: E402


def draw(img, H, col, w, h):
    G = np.linalg.inv(orient(np.array(H, float).reshape(3, 3), w, h))
    for poly in pitch_polylines():
        d = _densify(poly, 0.3)
        q = d @ G[:, :2].T + G[:, 2]
        front = q[:, 2] > 1e-6
        p = np.clip(q[:, :2] / np.where(front, q[:, 2], 1)[:, None], -5000, 5000)
        for a, b, o1, o2 in zip(p[:-1], p[1:], front[:-1], front[1:]):
            if o1 and o2:
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), col, 4, cv2.LINE_AA)


def label(img, text, sub, col):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.putText(img, text, (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 5, cv2.LINE_AA)
    cv2.putText(img, sub, (int(w * 0.57), h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=Path)
    ap.add_argument('before', type=Path)
    ap.add_argument('after', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--frames', default='0,300,455,600,750,900,1040')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, path, col in (('BEFORE', a.before, (0, 0, 255)), ('AFTER', a.after, (0, 230, 0))):
        res = json.loads(path.read_text(encoding='utf-8'))
        runs.append((name, res, LineMaskStore.load(path.with_name(path.name[:-5] + '.lines.npz')), col))
    w, h, fps = runs[0][1]['width'], runs[0][1]['height'], runs[0][1].get('fps') or 30.0
    P, _ = sample_pitch_lines(0.25)
    renderer = AutoCalibrator(scale=0.5)
    cap = cv2.VideoCapture(str(a.video))
    for t in [int(x) for x in a.frames.split(',')]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        if not ok:
            continue
        panels = []
        for name, res, masks, col in runs:
            im = img.copy()
            H = res['homographies'][t]
            if H is not None:
                draw(im, H, col, w, h)
            m = masks(t)
            scale = m[0].shape[1] / w
            e = frame_error(H, m[0], m[1], scale, P, renderer) if H is not None else None
            sub = 'no calibration' if e is None else \
                f"line err {'>=' if e['err'] >= 20 else ''}{e['err']:.0f}px  recall {e['recall']:.2f}"
            label(im, f'{name}  t={t / fps:.1f}s', sub, col)
            panels.append(cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA))
        path = a.out / f'frame_{t:04d}_{t / fps:04.1f}s.png'
        cv2.imwrite(str(path), np.hstack(panels), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(path)
    cap.release()


if __name__ == '__main__':
    main()
