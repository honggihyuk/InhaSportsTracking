"""
선수 속력 이상값 진단 (11부)

정밀 분석(파인튜닝 RF-DETR + SAM 발 위치 + SAM3 영상 추적 + 등번호 재연결 + 자동 보정)을 다시 돌리면서
트랙 ID 가 어디서 합쳐졌는지(stitch_tracks / SAM3 repair_tracks / 등번호 identify)를 함께 기록하고,
보정 결과에서 속력이 튀는 구간마다 원인 후보를 표로 보여 준다.

  # 1) GPU 분석 (원본 결과는 건드리지 않고 --out 폴더에 결과·라인 마스크·후처리 전 스냅숏 저장)
  python scripts/diagnose_speed.py run D:/TermProject/InhaSportsTracking/uploads/sports_video_1.mp4 \
      --checkpoint D:/TermProject/InhaSportsTracking/models/rfdetr_football/checkpoint_best_total.pth \
      --out uploads/diag

  # 2) 진단 표 (빠름, GPU 불필요) — 속력 상위 트랙, 이상 구간의 원인 후보
  python scripts/diagnose_speed.py report uploads/diag/sports_video_1.mp4.json [--around 22.4]

  # 3) 평활화만 다시 계산해 지표 비교 (speed_filter 끈 것 vs 켠 것)
  python scripts/diagnose_speed.py compare uploads/diag/sports_video_1.mp4.json
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import postprocess as pp  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 분석 실행 (+ 병합 기록)
# ---------------------------------------------------------------------------
def run(a):
    import torch
    from pipeline import video_analysis as va
    from pipeline.main_pipeline import Soccer3DPipeline as Pipeline
    from pipeline.pitch import LineMaskStore
    from tracking import identity, sam_video
    from tracking.rfdetr_detector import RFDETRDetector
    from tracking.tracker import BallTracker

    p = Pipeline(config_path=str(ROOT / 'configs' / 'model_config.yaml'))
    profile = p.analysis_profile('precise')
    rf = (p.config.get('detection', {}) or {}).get('rfdetr', {}) or {}
    det = RFDETRDetector(size=rf.get('size', 'medium'), checkpoint=a.checkpoint,
                         resolution=rf.get('resolution') if a.checkpoint else None,
                         device='cuda' if torch.cuda.is_available() else 'cpu',
                         confidence_threshold=rf.get('confidence_threshold', 0.4),
                         ball_threshold=rf.get('ball_threshold', 0.25),
                         imgsz=profile.get('img_size'))
    for attr, key in (('imgsz', 'img_size'), ('ball_imgsz', 'ball_imgsz'), ('ball_tile', 'ball_tile')):
        if key in profile and hasattr(det, attr):
            setattr(det, attr, profile[key])
    tracker = p.make_tracker()
    tracker.ball_tracker = BallTracker(max_jump=60)
    lines = LineMaskStore((0, 0))
    embedder = p.make_team_embedder() if profile.get('team_classifier') == 'siglip' else None
    refiner = p.make_sam_refiner() if profile.get('sam_refine') else None
    jersey = p.make_jersey_reader() if profile.get('jersey_ocr') and not a.no_ocr else None
    vtrack = p.make_video_tracker() if profile.get('sam_video') and not a.no_sam_video else None

    # 병합 기록: 원래 함수를 감싸 입력 스냅숏과 반환 매핑을 남긴다
    log = {}
    orig_stitch, orig_repair, orig_identify = pp.stitch_tracks, sam_video.repair_tracks, identity.identify

    def stitch(frames, *args, **kw):
        log['raw_frames'] = copy.deepcopy(frames)
        colors = args[2] if len(args) > 2 else kw.get('track_colors')
        log['raw_colors'] = {str(k): np.asarray(v).round(1).tolist() for k, v in (colors or {}).items()}
        out = orig_stitch(frames, *args, **kw)
        log['stitch'] = {str(k): v for k, v in out.items()}
        return out

    def repair(video_path, frames, *args, **kw):
        out = orig_repair(video_path, frames, *args, **kw)
        log['sam_links'] = {str(k): v for k, v in out[0].items()}
        log['sam_swaps'] = [list(s) for s in out[1]]
        return out

    def ident(bank, reader, frames, *args, **kw):
        out = orig_identify(bank, reader, frames, *args, **kw)
        log['jersey_relink'] = {str(k): v for k, v in out[1].items()}
        return out

    pp.stitch_tracks, sam_video.repair_tracks, identity.identify = stitch, repair, ident
    video = Path(a.video)
    try:
        result = va.analyze_video(video, det, tracker, stride=1, mode='precise', line_store=lines,
                                  postprocess=True, team_embedder=embedder, refiner=refiner,
                                  jersey_reader=jersey, video_tracker=vtrack, max_frames=a.max_frames,
                                  progress=lambda d, t: print(f'\r{d}/{t}', end='', flush=True) if d % 50 == 0 else None)
    finally:
        pp.stitch_tracks, sam_video.repair_tracks, identity.identify = orig_stitch, orig_repair, orig_identify
        for m in (embedder, refiner, jersey, vtrack):
            if m is not None:
                m.unload()
    print()
    a.out.mkdir(parents=True, exist_ok=True)
    lines.save(a.out / f'{video.name}.lines.npz')
    info = va.auto_calibrate(result, lines)
    result['auto_calibration'] = {'status': 'ok', **info}
    result['diagnose'] = {'checkpoint': a.checkpoint, 'sam_video': not a.no_sam_video, 'ocr': not a.no_ocr,
                          **{k: v for k, v in log.items() if not k.startswith('raw_')}}
    va.save(result, a.out / f'{video.name}.json')
    (a.out / f'{video.name}.raw_frames.json').write_text(json.dumps(log.get('raw_frames')), encoding='utf-8')
    (a.out / f'{video.name}.raw_colors.json').write_text(json.dumps(log.get('raw_colors')), encoding='utf-8')
    print('저장:', a.out / f'{video.name}.json', json.dumps(info, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 2. 진단 표
# ---------------------------------------------------------------------------
def raw_world(result):
    """보정만 한(평활화 전) 경기장 좌표: 저장된 호모그래피로 다시 투영 (가장자리 잘림 제외 규칙 동일)"""
    import cv2
    from pipeline import video_analysis as va
    out = []
    for objs, H in zip(result['frames'], result.get('homographies') or []):
        if H is None:
            out.append(None)
            continue
        H = np.array(H, float).reshape(3, 3)
        objs = [o for o in objs or [] if o[1] == pp.KIND_BALL
                or not va.clipped_by_border(o, result.get('width'), result.get('height'))]
        ent = []
        if objs:
            pts = np.float32([va.object_ground_point(o) for o in objs]).reshape(-1, 1, 2)
            for o, (x, y) in zip(objs, cv2.perspectiveTransform(pts, H).reshape(-1, 2)):
                if abs(x) <= va.HALF_LENGTH + va.PITCH_MARGIN and abs(y) <= va.HALF_WIDTH + va.PITCH_MARGIN:
                    ent.append([o[0], round(float(x), 2), round(float(y), 2)])
        out.append(ent)
    return out


def merge_origin(result):
    """{대표 ID: [(출처, 합쳐진 ID)]} — stitch / sam / jersey"""
    d = result.get('diagnose') or {}
    out = {}
    for src in ('stitch', 'sam_links', 'jersey_relink'):
        for child, root in (d.get(src) or {}).items():
            out.setdefault(int(root), []).append((src, int(child)))
    return out


def merge_frames(result):
    """병합 지점: 대표 ID → [(합쳐진 조각의 첫 프레임, 출처, 조각 ID)] (raw_frames 스냅숏이 있을 때만 정확)"""
    raw_path = Path(result['_path']).with_name(Path(result['_path']).name.replace('.json', '.raw_frames.json'))
    starts = {}
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        for t, objs in enumerate(raw or []):
            for o in objs:
                if o[1] == pp.KIND_PLAYER:
                    starts.setdefault(o[0], t)
    out = {}
    for root, items in merge_origin(result).items():
        for src, child in items:
            out.setdefault(root, []).append((starts.get(child), src, child))
    return out


def speed_events(world, fps, thresh=10.0):
    """평활 속력이 thresh 를 넘는 구간 [(tid, 시작, 끝, 최고)]"""
    ev = []
    series = {}
    for t, ent in enumerate(world):
        for e in ent or []:
            if int(e[0]) != 0:
                series.setdefault(int(e[0]), []).append((t, e[3]))
    for tid, s in series.items():
        s = np.array(s)
        hot = s[:, 1] > thresh
        i = 0
        while i < len(s):
            if hot[i]:
                j = i
                while j + 1 < len(s) and hot[j + 1] and s[j + 1, 0] - s[j, 0] <= 2:
                    j += 1
                ev.append((tid, int(s[i, 0]), int(s[j, 0]), float(s[i:j + 1, 1].max())))
                i = j + 1
            else:
                i += 1
    return sorted(ev, key=lambda e: -e[3])


def report(a):
    result = json.loads(Path(a.result).read_text(encoding='utf-8'))
    result['_path'] = str(a.result)
    fps = result['fps']
    world = result['world']
    conf = (result.get('calibration') or {}).get('confidence') or []
    rw = raw_world(result)
    print('후처리:', result.get('postprocess'))
    print('병합:', {k: len(v) for k, v in (result.get('diagnose') or {}).items() if isinstance(v, (dict, list))})
    print_summary('현재 결과', result)
    merges = merge_frames(result)
    boxes = {}
    for t, objs in enumerate(result['frames']):
        for o in objs:
            if o[1] == pp.KIND_PLAYER:
                boxes[(o[0], t)] = o
    raw_by = {}
    for t, ent in enumerate(rw):
        for e in ent or []:
            raw_by[(int(e[0]), t)] = (e[1], e[2])
    evs = speed_events(world, fps, a.thresh)
    if a.around is not None:
        f0 = int(a.around * fps)
        evs = [e for e in evs if e[1] - fps <= f0 <= e[2] + fps]
    print(f'\n속력 {a.thresh} m/s 초과 구간 {len(evs)} 개')
    for tid, s, e, vmax in evs[:a.top]:
        print(f'\n== 트랙 {tid}  {s / fps:.2f}~{e / fps:.2f}s (프레임 {s}~{e})  최고 {vmax:.1f} m/s  '
              f'팀 {result["teams"].get(str(tid))}  병합 {merges.get(tid, [])}')
        print('  t     프레임 conf  raw(x,y)        raw속력 평활속력  박스(x1,y1,x2,y2)  h   conf_det  비고')
        prev = None
        for t in range(max(0, s - a.pad), min(len(world), e + a.pad + 1)):
            sm = next((x for x in world[t] or [] if int(x[0]) == tid), None)
            r = raw_by.get((tid, t))
            o = boxes.get((tid, t))
            rs = ''
            if r is not None and prev is not None:
                rs = f'{np.hypot(r[0] - prev[1][0], r[1] - prev[1][1]) / ((t - prev[0]) / fps):6.1f}'
            note = []
            if o is not None and o[6] == 0.0:
                note.append('보간')
            if o is not None and r is None:
                note.append('잘림/밖')
            for f, src, child in merges.get(tid, []):
                if f == t:
                    note.append(f'←{src}:{child}')
            c = conf[t] if t < len(conf) else None
            print(f'  {t / fps:5.2f} {t:5d} {c if c is None else round(c, 2)!s:5} '
                  f'{str(r):16} {rs:>6} {sm[3] if sm else "-":>7}  '
                  f'{str(o[2:6]) if o else "-":24} {o[5] - o[3] if o else "-":>3} '
                  f'{o[6] if o else "-":>5}  {" ".join(note)}')
            if r is not None:
                prev = (t, r)


def print_summary(title, result):
    s = summary(result)
    print(f'[{title}] 최고 속력(지표) {s["max_speed"]} m/s, 평활 원시 최고 {s["world_max"]}, '
          f'10 m/s 초과 선수 {s["over10"]}, 스프린트 {s["sprints"]}, 총 거리 {s["distance"]} m, 상위 {s["top"]}')


def summary(result):
    from analysis import stats as match_stats
    st = match_stats.compute(result)['players']
    world_max = 0.0
    for ent in result['world']:
        for e in ent or []:
            if int(e[0]) != 0:
                world_max = max(world_max, e[3])
    return {
        'top': sorted(((s['max_speed'], tid) for tid, s in st.items()), reverse=True)[:5],
        'max_speed': max((s['max_speed'] for s in st.values()), default=0.0),
        'world_max': round(world_max, 2),
        'over10': sum(s['max_speed'] > 10 for s in st.values()),
        'over10_raw': len({int(e[0]) for ent in result['world'] for e in ent or [] if int(e[0]) != 0 and e[3] > 10}),
        'sprints': sum(len(s['sprints']) for s in st.values()),
        'sprint_list': sorted((tid, sp['start'], sp['end'], sp['max_speed'])
                              for tid, s in st.items() for sp in s['sprints']),
        'distance': round(sum(s['distance'] for s in st.values()), 1),
    }


# ---------------------------------------------------------------------------
# 3. 평활화 전후 비교 (보정·트랙은 그대로, smooth_world 만 다시)
# ---------------------------------------------------------------------------
def resmooth(result, **kw):
    """저장된 호모그래피로 다시 투영한 좌표를 video_analysis.calibrate 와 같은 인자로 다시 평활화"""
    r = dict(result)
    r['world'] = pp.smooth_world(raw_world(result), result['fps'],
                                 frame_conf=(result.get('calibration') or {}).get('confidence'), **kw)
    return r


def compare(a):
    result = json.loads(Path(a.result).read_text(encoding='utf-8'))
    before = summary(resmooth(result, speed_filter=False))
    rep = {}
    after = summary(resmooth(result, speed_filter=pp.load_speed_filter(), report=rep))
    keys = ['max_speed', 'world_max', 'over10', 'over10_raw', 'sprints', 'distance']
    print(f'{"지표":14} {"처리 전":>10} {"처리 후":>10}')
    for k in keys:
        print(f'{k:14} {before[k]!s:>10} {after[k]!s:>10}')
    b, c = set(before['sprint_list']), set(after['sprint_list'])
    print('\n스프린트 (트랙, 시작, 끝, 최고)')
    print('  처리 전에만:', sorted(b - c))
    print('  처리 후에만:', sorted(c - b))
    print('  공통 수:', len(b & c))
    print('\n점프 처리:', {k: v for k, v in rep.items() if k != 'tracks'})
    for tr in rep['tracks']:
        f = tr['frames']
        print(f"  트랙 {tr['id']}: 이상값 {len(f)} 프레임 ({f[0]}~{f[-1]}), 끊음 {tr['cuts']}")
    if a.json:
        Path(a.json).write_text(json.dumps({'before': before, 'after': after, 'filter': rep}, ensure_ascii=False,
                                           indent=1), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('video')
    r.add_argument('--checkpoint', default=None, help='파인튜닝 RF-DETR 가중치 (.pth). 없으면 COCO')
    r.add_argument('--out', type=Path, default=ROOT / 'uploads' / 'diag')
    r.add_argument('--max-frames', type=int, default=None)
    r.add_argument('--no-ocr', action='store_true')
    r.add_argument('--no-sam-video', action='store_true')
    g = sub.add_parser('report')
    g.add_argument('result')
    g.add_argument('--around', type=float, default=None, help='이 시각(초) 부근 구간만')
    g.add_argument('--thresh', type=float, default=10.0)
    g.add_argument('--top', type=int, default=8)
    g.add_argument('--pad', type=int, default=10, help='구간 앞뒤로 보여 줄 프레임 수')
    c = sub.add_parser('compare')
    c.add_argument('result')
    c.add_argument('--json', default=None)
    a = ap.parse_args()
    {'run': run, 'report': report, 'compare': compare}[a.cmd](a)


if __name__ == '__main__':
    main()
