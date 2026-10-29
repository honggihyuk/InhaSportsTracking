"""
경기장 보정 품질 평가 — 분석 결과(JSON)와 라인 마스크(.lines.npz)로 프레임별 라인 투영 오차와 속력 분포를 잰다

  python scripts/eval_calibration.py data/samples/analysis/sports_video_1.mp4.json [--every 1.0] [--json out.json]

지표 (일정 간격 프레임마다):
  - 투영 오차: 호모그래피로 화면에 투영한 경기장 라인 모델 점(보이는 점) → 가장 가까운 검출 라인 픽셀까지
    거리의 중앙값 (원본 해상도 px, 20 px 에서 자름)
  - 정밀도: 투영 모델 점 중 검출 라인과 4 px(원본) 이내인 비율
  - 재현율: 검출 라인 픽셀 중 투영 모델과 6 px(원본) 이내인 비율
  - 정상 프레임: 투영 오차 ≤ 5 px 이고 재현율 ≥ 0.5 (라인이 거의 안 보이는 프레임은 판정 제외)
속력: world[t] = [id, x, y, speed] 의 선수(공 제외) 최고 속력·상위 분포
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.autocalib import AutoCalibrator  # noqa: E402
from pipeline.pitch import LineMaskStore, sample_pitch_lines, scale_matrix  # noqa: E402

CLIP_PX = 20.0
OK_ERR_PX = 5.0
OK_RECALL = 0.5
MIN_LINE_PX = 150        # 라인 픽셀이 이보다 적은 프레임은 판정 제외 (작업 해상도)


def oriented(H, width, height):
    """호모그래피 부호 맞춤: 화면 아래 가운데(경기장 위)의 동차 좌표가 양수가 되도록.
    H / H[2,2] 로 정규화하면 화면 왼쪽 위(관중석, 지평선 너머)가 기준이 되어 부호가 뒤집힐 수 있다."""
    H = np.array(H, float).reshape(3, 3)
    return H if (H @ [width / 2, height * 0.9, 1.0])[2] > 0 else -H


def frame_error(H, mask, valid, scale, P, renderer):
    """H: 원본 이미지 → 경기장. 반환 dict 또는 None (보정 없음)"""
    if H is None:
        return None
    S = scale_matrix(scale)
    h, w = mask.shape
    G = S @ np.linalg.inv(oriented(H, w / scale, h / scale))
    q = P @ G[:, :2].T + G[:, 2]
    z = q[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u, v = q[:, 0] / z, q[:, 1] / z
    inside = (z > 0) & (u >= 0) & (u < w - 0.5) & (v >= 0) & (v < h - 0.5)
    ui, vi = np.round(u[inside]).astype(int), np.round(v[inside]).astype(int)
    keep = valid[vi, ui] > 0
    ui, vi = ui[keep], vi[keep]
    dist = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    line_px = mask > 0
    if len(ui) < 20:
        return {'err': CLIP_PX, 'precision': 0.0, 'recall': 0.0, 'visible': int(len(ui)),
                'line_px': int(line_px.sum())}
    d = np.minimum(dist[vi, ui] / scale, CLIP_PX)
    near = cv2.dilate(renderer._render(G, mask.shape),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(6 * scale)) + 1,) * 2))
    recall = float((near[line_px] > 0).mean()) if line_px.any() else 0.0
    return {'err': round(float(np.median(d)), 2), 'precision': round(float((d <= 4).mean()), 3),
            'recall': round(recall, 3), 'visible': int(len(ui)), 'line_px': int(line_px.sum())}


def evaluate(result, masks, every_s=1.0):
    fps = result.get('fps') or 30.0
    n = result['frame_count']
    scale = masks.shape[1] / result['width']
    P, _ = sample_pitch_lines(0.25)
    renderer = AutoCalibrator(scale=scale)
    Hs = result.get('homographies') or [None] * n
    step = max(1, int(round(every_s * fps)))
    rows = []
    for t in range(0, n, step):
        data = masks(t)
        if data is None:
            continue
        m = frame_error(Hs[t], data[0], data[1], scale, P, renderer)
        row = {'frame': t, 'time': round(t / fps, 2)}
        if m is None:
            row.update(calibrated=False, ok=False, judged=True)
        else:
            judged = m['line_px'] >= MIN_LINE_PX
            row.update(m, calibrated=True, judged=judged,
                       ok=bool(judged and m['err'] <= OK_ERR_PX and m['recall'] >= OK_RECALL))
        rows.append(row)
    judged = [r for r in rows if r['judged']]
    errs = [r['err'] for r in judged if r['calibrated']]
    summary = {
        'frames': len(rows), 'judged': len(judged),
        'ok_ratio': round(sum(r['ok'] for r in judged) / max(1, len(judged)), 3),
        'median_err_px': round(float(np.median(errs)), 2) if errs else None,
        'mean_err_px': round(float(np.mean(errs)), 2) if errs else None,
        'mean_recall': round(float(np.mean([r['recall'] for r in judged if r['calibrated']])), 3) if errs else None,
    }
    summary.update(speed_summary(result))
    conf = (result.get('calibration') or {}).get('confidence')
    if conf:
        summary['low_conf_frames'] = int(sum(c is not None and c < 0.5 for c in conf))
    return summary, rows


def speed_summary(result, ball_id=0):
    world = result.get('world') or []
    per_player, all_sp = {}, []
    for entries in world:
        for e in entries or []:
            if int(e[0]) == ball_id or len(e) < 4:
                continue
            per_player[int(e[0])] = max(per_player.get(int(e[0]), 0.0), float(e[3]))
            all_sp.append(float(e[3]))
    if not all_sp:
        return {}
    tops = sorted(per_player.values(), reverse=True)
    return {
        'max_speed': round(tops[0], 2),
        'top5_max_speeds': [round(s, 2) for s in tops[:5]],
        'players': len(tops),
        'players_over_10': sum(s > 10 for s in tops),
        'players_over_9_5': sum(s > 9.5 for s in tops),
        'speed_p99': round(float(np.percentile(all_sp, 99)), 2),
        'speed_p999': round(float(np.percentile(all_sp, 99.9)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result', type=Path)
    ap.add_argument('--lines', type=Path, default=None, help='기본: <result 에서 .json 을 .lines.npz 로>')
    ap.add_argument('--every', type=float, default=1.0)
    ap.add_argument('--json', type=Path, default=None, help='프레임별 결과 저장')
    ap.add_argument('--rows', action='store_true', help='프레임별 결과 출력')
    a = ap.parse_args()
    result = json.loads(a.result.read_text(encoding='utf-8'))
    lines = a.lines or a.result.with_name(a.result.name[:-len('.json')] + '.lines.npz')
    masks = LineMaskStore.load(lines)
    summary, rows = evaluate(result, masks, a.every)
    if a.rows:
        for r in rows:
            print(r)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if a.json:
        a.json.write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
