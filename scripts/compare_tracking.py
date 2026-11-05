"""
두 분석 결과의 추적 품질 비교 (10부) — 트랙 수, 평균 트랙 길이, 끊김 수, 등번호 확정 비율, 처리 정보

  python scripts/compare_tracking.py before.json after.json [--labels 기존 새방식]

끊김 = 화면 가장자리가 아닌 곳에서 끝난 트랙 중, 1 초 안에 근처(박스 높이 3 배 이내)에서 다른 트랙이 시작한 경우
       (tracking/sam_video.break_events 와 같은 규칙 — 가림·겹침으로 ID 가 끊긴 것으로 보이는 곳)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracking.sam_video import break_events, overlap_events, player_tracks  # noqa: E402


def metrics(result: dict) -> dict:
    tracks = player_tracks(result['frames'])
    labels = result.get('track_labels') or {}
    players = {t: fr for t, fr in tracks.items() if labels.get(str(t)) != 'referee'}
    lengths = [len(fr) for fr in players.values()]
    fps = result.get('fps') or 30.0
    size = (result.get('width') or 1920, result.get('height') or 1080)
    ids = result.get('identities') or {}
    confirmed = [t for t in players if (ids.get(str(t)) or {}).get('jersey_number') is not None]
    boxes = sum(lengths)
    return {
        'tracks': len(tracks), 'player_tracks': len(players),
        'mean_track_s': round(float(np.mean(lengths)) / fps, 2) if lengths else 0.0,
        'median_track_s': round(float(np.median(lengths)) / fps, 2) if lengths else 0.0,
        'long_tracks_10s': sum(n >= 10 * fps for n in lengths),
        'breaks': len(break_events(tracks, len(result['frames']), size, int(fps))),
        'overlap_scenes': len(overlap_events(tracks)),
        'jersey_confirmed': len(confirmed),
        'jersey_confirmed_pct': round(100 * len(confirmed) / max(1, len(players)), 1),
        'jersey_box_coverage_pct': round(100 * sum(len(players[t]) for t in confirmed) / max(1, boxes), 1),
        'sam_video': {k: v for k, v in (result.get('sam_video') or {}).items() if k not in ('swap_evidence', 'link_iou')},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('before', type=Path)
    ap.add_argument('after', type=Path)
    ap.add_argument('--labels', nargs=2, default=['before', 'after'])
    a = ap.parse_args()
    out = {}
    for name, path in zip(a.labels, (a.before, a.after)):
        out[name] = metrics(json.loads(path.read_text(encoding='utf-8')))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
