"""
실시간(realtime) vs 정밀(precise) 분석 모드 정량 비교 — 정답이 있는 합성 중계 영상 사용

실제 중계 영상에는 프레임별 정답 좌표가 없어 정확도를 수치로 비교하기 어렵다. 이 스크립트는
pipeline/synthetic.py 로 팬·줌하는 중계 화면을 만들고, 정답 박스에 잡음·누락·가려짐(연속 누락)·공 오탐을
섞은 "모의 탐지기"로 두 모드를 똑같이 돌려 다음을 비교한다.

  - 보정 오차: 키프레임 1 개(클릭 오차 ±1.5 px)에서 전파한 호모그래피의 경기장 위치 오차 (m)
  - 선수 위치 오차: 경기장 좌표 vs 정답 (m)
  - 속력 오차: 0.2 초 차분(현재 화면 방식) / 평활 속력 vs 정답 (m/s)
  - ID 안정성: 정답 선수 1 명당 트랙 ID 수 (1 이 이상적)
  - 공 검출 프레임 비율

실행: python scripts/benchmark_precise.py [--frames 240] [--tracker botsort|simple]
"""

import argparse
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import synthetic as syn, video_analysis as va                         # noqa: E402
from pipeline.pitch import LineMaskStore                                           # noqa: E402
from tracking.detector import BALL_OFFSET, Detection                               # noqa: E402
from tracking.tracker import BallTracker, SoccerTracker                            # noqa: E402

LANDMARKS = [(0, 34), (0, -34), (0, 9.15), (0, -9.15), (-36, 20.16), (36, 20.16), (-52.5, 34), (52.5, 34)]


class MockDetector:
    """정답 박스 + 잡음 + 무작위 누락 + 가려짐(연속 누락) + 공 누락·오탐"""

    def __init__(self, clip: syn.SyntheticClip, frame_index, seed=1, miss=0.06, ball_rate=0.45, ball_fp=0.08):
        self.clip, self.index = clip, frame_index
        self.rng = np.random.default_rng(seed)
        self.miss, self.ball_rate, self.ball_fp = miss, ball_rate, ball_fp
        n = len(clip.frames)
        # 선수마다 1~2 번, 0.8~1.6 초 가려짐
        self.occluded = defaultdict(set)
        for pid in {p[0] for p in clip.players[0]}:
            for _ in range(self.rng.integers(1, 3)):
                start = int(self.rng.integers(0, n))
                self.occluded[pid].update(range(start, start + int(self.rng.integers(24, 48))))

    def detect(self, frame):
        t = self.index(frame)
        out = []
        for pid, team, (x1, y1, x2, y2) in self.clip.boxes[t]:
            if t in self.occluded[pid] or self.rng.random() < self.miss:
                continue
            j = self.rng.normal(0, 1.0, 4)
            out.append(Detection((int(x1 + j[0]), int(y1 + j[1]), int(x2 + j[2]), int(y2 + j[3])),
                                 0.9, 0, 'player_person', 'player'))
        G = self.clip.G[t]
        if self.rng.random() < self.ball_rate:
            bx, by = self.clip.ball[t]
            q = G @ [bx, by, 1]
            u, v = q[0] / q[2], q[1] / q[2]
            out.append(Detection((int(u - 3), int(v - 3), int(u + 3), int(v + 3)), 0.6, BALL_OFFSET, 'ball_ball', 'ball'))
        if self.rng.random() < self.ball_fp:
            h, w = frame.shape[:2]
            u, v = self.rng.uniform(0, w), self.rng.uniform(h * 0.3, h)
            out.append(Detection((int(u - 3), int(v - 3), int(u + 3), int(v + 3)), 0.5, BALL_OFFSET, 'ball_ball', 'ball'))
        return out


def frame_indexer(video: Path):
    cap = cv2.VideoCapture(str(video))
    keys = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        keys[f[::37, ::41].tobytes()] = i
        i += 1
    cap.release()
    return lambda frame: keys[frame[::37, ::41].tobytes()]


def keyframe_clicks(clip, t, rng, noise=1.5):
    G = clip.G[t]
    pts = []
    for x, y in LANDMARKS:
        q = G @ [x, y, 1]
        u, v = q[0] / q[2], q[1] / q[2]
        if 0 <= u < clip.frames[0].shape[1] and 0 <= v < clip.frames[0].shape[0]:
            pts.append({'image': [float(u + rng.normal(0, noise)), float(v + rng.normal(0, noise))],
                        'pitch': [x, y]})
    return pts


def gt_speed(clip, fps):
    """정답 선수 속력 (중앙 차분)"""
    pos = defaultdict(dict)
    for t, ps in enumerate(clip.players):
        for pid, _, x, y in ps:
            pos[pid][t] = np.array([x, y])
    sp = defaultdict(dict)
    for pid, d in pos.items():
        for t in d:
            if t - 1 in d and t + 1 in d:
                sp[pid][t] = np.linalg.norm(d[t + 1] - d[t - 1]) * fps / 2
    return sp


def evaluate(name, clip, res, fps):
    W, H = clip.frames[0].shape[1], clip.frames[0].shape[0]
    Hs = [None if h is None else np.array(h).reshape(3, 3) for h in res.get('homographies', [])]
    calib = [syn.pitch_error(h, clip.G[t], W, H) for t, h in enumerate(Hs) if h is not None]

    # 트랙 ↔ 정답 선수: 박스 하단 중앙이 가장 가까운 정답 선수 (다수결)
    votes = defaultdict(Counter)
    for t, objs in enumerate(res['frames']):
        gts = clip.boxes[t]
        for o in objs:
            if o[1] != va.KIND_PLAYER or not gts:
                continue
            foot = np.array([(o[2] + o[4]) / 2, o[5]])
            d = [np.linalg.norm(foot - [(b[0] + b[2]) / 2, b[3]]) for _, _, b in gts]
            i = int(np.argmin(d))
            if d[i] < 15:
                votes[o[0]][gts[i][0]] += 1
    track_gt = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    ids_per_player = Counter(track_gt.values())

    gt_pos = [{pid: np.array([x, y]) for pid, _, x, y in ps} for ps in clip.players]
    sp_true = gt_speed(clip, fps)
    pos_err, sp_err_raw, sp_err_smooth = [], [], []
    world = res.get('world') or []
    window = 6
    prev_map = lambda t: {e[0]: e for e in (world[t] if 0 <= t < len(world) and world[t] else [])}
    for t, entries in enumerate(world):
        if not entries:
            continue
        prev = prev_map(t - window)
        for e in entries:
            pid = track_gt.get(e[0])
            if e[0] == 0 or pid is None or pid not in gt_pos[t]:
                continue
            pos_err.append(np.linalg.norm(np.array(e[1:3]) - gt_pos[t][pid]))
            if t in sp_true[pid]:
                if len(e) > 3:
                    sp_err_smooth.append(abs(e[3] - sp_true[pid][t]))
                p = prev.get(e[0])
                if p is not None:
                    raw = np.hypot(e[1] - p[1], e[2] - p[2]) * fps / window
                    sp_err_raw.append(abs(raw - sp_true[pid][t]))
    ball_frames = sum(any(o[1] == va.KIND_BALL for o in f) for f in res['frames'])
    ball_err = []
    for t, entries in enumerate(world):
        for e in entries or []:
            if e[0] == 0:
                ball_err.append(np.linalg.norm(np.array(e[1:3]) - clip.ball[t]))

    stat = lambda v: f"{np.mean(v):.3f} (p95 {np.percentile(v, 95):.3f})" if len(v) else "-"
    print(f"\n[{name}]")
    print(f"  보정 경기장 오차 (m)     : {stat(calib)}  최대 {max(calib):.3f}" if calib else "  보정: -")
    if res.get('calibration', {}).get('refined_frames') is not None:
        print(f"  라인 정렬 채택 프레임    : {res['calibration']['refined_frames']} / {len(Hs)}")
    print(f"  선수 위치 오차 (m)       : {stat(pos_err)}")
    print(f"  속력 오차 0.2s 차분 (m/s): {stat(sp_err_raw)}")
    if sp_err_smooth:
        print(f"  속력 오차 평활 (m/s)     : {stat(sp_err_smooth)}")
    print(f"  정답 선수당 트랙 ID 수    : {np.mean(list(ids_per_player.values())):.2f} "
          f"(전체 트랙 {len(votes)}, 정답 선수 {len(ids_per_player)})")
    print(f"  공 박스 프레임 비율      : {ball_frames / len(res['frames']):.2%}")
    print(f"  공 위치 오차 (m)         : {stat(ball_err)}")
    if res.get('postprocess'):
        print(f"  후처리                   : {res['postprocess']}")
    if res.get('stats'):
        s = res['stats']
        print(f"  지표: 점유율 {s['possession']['share']}, 이벤트 {len(s['events'])} 개, "
              f"골키퍼/역할 {res.get('roles')}")
    return {'calib': np.mean(calib) if calib else None, 'pos': np.mean(pos_err) if pos_err else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--tracker', default='botsort')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    clip = syn.render_clip(n=args.frames, seed=args.seed)
    fps = 30.0
    with tempfile.TemporaryDirectory() as d:
        video = Path(d) / 'synthetic.avi'
        syn.write_video(clip, video, fps)
        index = frame_indexer(video)
        print(f"합성 영상 {args.frames} 프레임 생성 {time.time() - t0:.1f}s")
        rng = np.random.default_rng(args.seed + 7)
        kf = [{'frame': 0, 'points': keyframe_clicks(clip, 0, rng)}]

        for mode, stride in (('realtime', 3), ('precise', 1)):
            t0 = time.time()
            tracker = SoccerTracker(tracker_type=args.tracker, max_age=30)
            tracker.ball_tracker = BallTracker(max_jump=60 * stride)
            lines = LineMaskStore((0, 0)) if mode == 'precise' else None
            res = va.analyze_video(video, MockDetector(clip, index, seed=args.seed + 1), tracker,
                                   stride=stride, mode=mode, line_store=lines)
            res = va.calibrate(res, kf, line_masks=lines)
            print(f"\n{mode}: 분석+보정 {time.time() - t0:.1f}s")
            evaluate(mode, clip, res, fps)


if __name__ == '__main__':
    main()
