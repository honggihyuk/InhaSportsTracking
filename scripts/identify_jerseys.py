"""
분석 결과에 등번호를 붙이기 (탐지 없이) — 영상을 다시 읽어 트랙별 크롭을 모으고 GLM-OCR → 투표 → 재연결 (9부)

  python scripts/identify_jerseys.py data/samples/sports_video_1.mp4 data/samples/analysis/sports_video_1.mp4.json \
      --out uploads/analysis/sports_video_1.mp4.json [--sheet-dir 폴더 --sheet-tracks 25]

analyze_video(jersey_reader=...) 와 같은 크롭 규칙(JERSEY_SAMPLE_EVERY 프레임마다, jersey_crop)을 쓴다.
--sheet-dir 를 주면 무작위 트랙의 크롭·판독 결과를 PNG 로 남겨 사람이 번호를 확인할 수 있게 한다.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import video_analysis as va  # noqa: E402
from tracking.identity import GlmOcrReader, JerseyCropBank, identify, jersey_crop, parse_number  # noqa: E402


def collect(video: Path, result: dict, every: int) -> JerseyCropBank:
    bank = JerseyCropBank()
    cap = cv2.VideoCapture(str(video))
    try:
        for t in range(result['frame_count']):
            ok, frame = cap.read()
            if not ok:
                break
            if t % every:
                continue
            objs = result['frames'][t]
            boxes = [o[2:6] for o in objs if o[1] == va.KIND_PLAYER]
            for o in objs:
                if o[1] == va.KIND_PLAYER and o[6] > 0:          # 보간 박스(신뢰도 0)는 제외
                    got = jersey_crop(frame, o[2:6], boxes)
                    if got is not None:
                        bank.add(int(o[0]), t, *got)
    finally:
        cap.release()
    return bank


def sheet(bank, details, ids, tids, path):
    """트랙마다 한 줄: 크롭 최대 8 장 + 판독 텍스트, 맨 앞에 트랙 ID 와 확정 번호"""
    rows = []
    for tid in tids:
        items = sorted(bank[tid], key=lambda x: x[1])[:8]
        reads = {t: (txt, c) for t, txt, c in details['texts'].get(tid, [])}
        info = ids.get(tid, {})
        head = np.zeros((130, 150, 3), np.uint8)
        num = info.get('jersey_number')
        cv2.putText(head, f'#{tid}', (8, 40), 0, 1.0, (255, 255, 255), 2)
        cv2.putText(head, f'-> {num if num is not None else "?"}', (8, 85), 0, 1.0,
                    (80, 230, 80) if num is not None else (120, 120, 255), 2)
        cv2.putText(head, f"conf {info.get('confidence', 0):.2f}", (8, 118), 0, 0.5, (200, 200, 200), 1)
        cells = [head]
        for q, t, c in items:
            im = cv2.resize(c, (90, 100), interpolation=cv2.INTER_CUBIC)
            cell = np.zeros((130, 90, 3), np.uint8)
            cell[:100] = im
            txt, cf = reads.get(t, ('', 0))
            ok = parse_number(txt) is not None
            cv2.putText(cell, (txt or '-')[:6], (3, 118), 0, 0.5, (80, 230, 80) if ok else (150, 150, 150), 1)
            cells.append(cell)
        while len(cells) < 9:
            cells.append(np.zeros((130, 90, 3), np.uint8))
        rows.append(np.hstack(cells))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=Path)
    ap.add_argument('result', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--every', type=int, default=va.JERSEY_SAMPLE_EVERY)
    ap.add_argument('--model', default='zai-org/GLM-OCR')
    ap.add_argument('--sheet-dir', type=Path, default=None)
    ap.add_argument('--sheet-tracks', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    import torch
    result = va.load(a.result)
    t0 = time.perf_counter()
    bank = collect(a.video, result, a.every)
    t1 = time.perf_counter()
    snapshot = {tid: list(items) for tid, items in bank.items.items()}      # 재연결 전 크롭 (확인용)
    reader = GlmOcrReader(a.model, device='cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    teams = {int(k): v for k, v in (result.get('teams') or {}).items()}
    labels = {int(k): v for k, v in (result.get('track_labels') or {}).items()}
    referees = [t for t, lab in labels.items() if lab == 'referee']
    keepers = [t for t, lab in labels.items() if lab == 'goalkeeper'] +         [int(k) for k, role in (result.get('roles') or {}).items() if role == 'goalkeeper']
    ids, parent, details = identify(bank, reader, result['frames'], teams, exclude=referees, goalkeepers=keepers)
    summary = {**details['summary'], 'collect_s': round(t1 - t0, 1),
               'max_vram_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
               'model': a.model}
    result['identities'] = {str(k): v for k, v in ids.items()}
    result['identity'] = summary
    for child in parent:
        result['teams'].pop(str(child), None)
    va.save(result, a.out)
    (a.out.with_suffix('.identity.json')).write_text(json.dumps(
        {'summary': summary, 'parent': {str(k): v for k, v in parent.items()},
         'identities': result['identities'], 'votes': {str(k): v for k, v in details['votes'].items()},
         'texts': {str(k): v for k, v in details['texts'].items()}},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))

    if a.sheet_dir:
        a.sheet_dir.mkdir(parents=True, exist_ok=True)
        pre_ids = details['votes']
        cand = sorted(t for t in snapshot if len(snapshot[t]) >= 2)
        random.Random(a.seed).shuffle(cand)
        pick = cand[:a.sheet_tracks]
        for i in range(0, len(pick), 13):
            sheet(snapshot, details, pre_ids, pick[i:i + 13], a.sheet_dir / f'tracks_{i // 13 + 1}.png')
        (a.sheet_dir / 'picked_tracks.json').write_text(json.dumps(pick), encoding='utf-8')


if __name__ == '__main__':
    main()
