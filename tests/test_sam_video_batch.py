"""SAM3 영상 추적 묶음 처리 테스트 (tracking/sam_video.py — merge_windows, track_many, repair_tracks(batch=True))

실제 SAM3 는 쓰지 않는다. '진짜 선수 위치'를 아는 가짜 추적기로, 위험 구간을 덩어리로 묶어 처리해도
하나씩 처리할 때와 결과(이은 끊김, 바로잡은 ID 교체, 고친 frames)가 같은지 확인한다.
실행: python -m pytest tests -q
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking import sam_video as sv

W, H = 1920, 1080


def box_at(x, y=500.0, w=60.0, h=160.0):
    return (x, y, x + w, y + h)


class OneByOneTracker:
    """
    선수별 실제 위치 truth[pid](t) 를 알고, 프롬프트 박스와 가장 가까운 선수를 계속 따라가는 가짜 SAM3.
    self.start = 마지막으로 읽은 구간의 첫 영상 프레임 (가짜 read_frames 가 설정)
    """

    def __init__(self, truth):
        self.truth = truth
        self.start = 0
        self.sessions = 0              # 세션(영상 전처리) 수
        self.propagations = 0          # 전파 횟수

    def _owner(self, t, b):
        return min(self.truth, key=lambda p: np.hypot(self.truth[p](t)[0] - b[0], self.truth[p](t)[1] - b[1]))

    def track(self, frames_rgb, prompts):
        self.sessions += 1
        self.propagations += 1
        owner = {oid: self._owner(self.start, b) for oid, b in prompts.items()}
        return [{oid: self.truth[p](self.start + k) for oid, p in owner.items()} for k in range(len(frames_rgb))]



class AbsTruthTracker(OneByOneTracker):
    """OneByOneTracker + track_many (덩어리 한 번에)"""

    def track_many(self, frames_rgb, jobs):
        self.sessions += 1
        self.propagations += len({s for s, _, _ in jobs})          # 같은 시작 프레임은 한 번에 전파
        out = []
        for s, e, prompts in jobs:
            t0 = self.start + s
            owner = {oid: self._owner(t0, b) for oid, b in prompts.items()}
            out.append([{oid: self.truth[p](t0 + k) for oid, p in owner.items()}
                        for k in range(min(e, len(frames_rgb) - 1) - s + 1)])
        return out


def patch_reader(monkeypatch, tracker, reads):
    def fake_read(path, start, end):
        tracker.start = start
        reads.append((start, end))
        return [np.zeros((4, 4, 3), np.uint8) for _ in range(start, end + 1)]
    monkeypatch.setattr(sv, 'read_frames', fake_read)


# ---------------------------------------------------------------------------
# 구간 묶기
# ---------------------------------------------------------------------------
def test_merge_windows_joins_overlapping_and_touching():
    groups = sv.merge_windows([(50, 80), (10, 40), (35, 60), (100, 120), (121, 130)], max_span=500)
    assert groups == [(10, 80, [1, 2, 0]), (100, 130, [3, 4])]


def test_merge_windows_respects_gap_and_span():
    assert sv.merge_windows([(0, 10), (15, 20)], max_span=100) == [(0, 10, [0]), (15, 20, [1])]
    assert sv.merge_windows([(0, 10), (15, 20)], max_span=100, max_gap=4) == [(0, 20, [0, 1])]
    # 덩어리가 max_span 을 넘으면 새 덩어리
    assert sv.merge_windows([(0, 40), (30, 70), (60, 100)], max_span=75) == [(0, 70, [0, 1]), (60, 100, [2])]
    # max_span 보다 긴 구간은 혼자
    assert sv.merge_windows([(0, 200), (10, 20)], max_span=50) == [(0, 200, [0]), (10, 20, [1])]
    assert sv.merge_windows([], max_span=50) == []


def test_track_jobs_skips_short_clips_and_slices_results():
    truth = {'P': lambda t: box_at(400 + t)}
    fake = AbsTruthTracker(truth)
    fake.start = 100
    clip = [np.zeros((4, 4, 3), np.uint8) for _ in range(10)]
    res = sv.track_jobs(fake, clip, [(2, 5, {7: box_at(402)}), (9, 20, {8: box_at(409)}), (4, 30, {9: box_at(404)})])
    assert len(res[0]) == 4 and res[0][3][7] == box_at(400 + 105)
    assert res[1] is None                      # 덩어리 끝이라 프레임 1 개뿐 → 하나씩 처리할 때처럼 건너뜀
    assert len(res[2]) == 6                    # 덩어리 끝에서 잘림


# ---------------------------------------------------------------------------
# 묶음 처리 = 하나씩 처리
# ---------------------------------------------------------------------------
def scenario():
    """
    끊김 2 곳(시간 겹침 → 한 덩어리), ID 교체 1 곳, 그리고 교체를 바로잡은 뒤 프롬프트가 바뀌는 겹침 1 곳
      P →, Q ← : t=30 에서 교차, 추적기는 교차 뒤 ID 1·2 를 바꿔 붙임
      R 정지(x=500): 바뀐 ID 1(실제 Q)이 t≈60 에 R 과 겹침 → 바로잡은 뒤에는 ID 1 = P 라 프롬프트가 달라짐
      S, T: 가려져 끊겼다가 새 ID 로 다시 잡힘 (끊긴 구간이 시간상 겹침)
    """
    n = 100
    truth = {'P': lambda t: box_at(500 + 10 * t), 'Q': lambda t: box_at(1100 - 10 * t),
             'R': lambda t: box_at(500, y=505.0),
             'S': lambda t: box_at(1500 + t, y=200.0), 'T': lambda t: box_at(1700 - t, y=800.0)}
    frames = [[] for _ in range(n)]
    for t in range(n):
        p, q = truth['P'](t), truth['Q'](t)
        a, b = (p, q) if t <= 30 else (q, p)
        frames[t] += [[1, 0, *a, 0.9], [2, 0, *b, 0.9], [3, 0, *truth['R'](t), 0.9]]
        if t < 20:
            frames[t].append([4, 0, *truth['S'](t), 0.9])
        elif t >= 28:
            frames[t].append([5, 0, *truth['S'](t), 0.9])
        if t < 25:
            frames[t].append([6, 0, *truth['T'](t), 0.9])
        elif t >= 33:
            frames[t].append([7, 0, *truth['T'](t), 0.9])
    return truth, frames


def run(monkeypatch, batched, **kw):
    truth, frames = scenario()
    fake = (AbsTruthTracker if batched else OneByOneTracker)(truth)
    reads = []
    patch_reader(monkeypatch, fake, reads)
    calls = []
    parent, swaps, info = sv.repair_tracks('clip', frames, (W, H), fake, fps=30.0,
                                           progress=lambda d, t: calls.append((d, t)), **kw)
    return frames, parent, swaps, info, fake, reads, calls


def test_batched_repair_matches_one_by_one(monkeypatch):
    f1, p1, s1, i1, fake1, reads1, _ = run(monkeypatch, batched=False)
    f2, p2, s2, i2, fake2, reads2, _ = run(monkeypatch, batched=True)
    assert p1 == p2 == {5: 4, 7: 6}
    assert s1 == s2 and len(s1) == 1 and s1[0][:2] == (1, 2)
    assert f1 == f2
    strip = lambda i: {k: v for k, v in i.items() if k not in ('seconds', 'batched', 'batches', 'recomputed')}
    assert strip(i1) == strip(i2)
    assert i2['batched'] and not i1['batched']
    # 묶음 처리: 끊김 2 곳은 한 덩어리 (영상 한 번 읽기), 교체를 바로잡아 프롬프트가 바뀐 겹침은 다시 전파
    assert i2['recomputed'] >= 1
    assert i2['batches'] < i2['break_events'] + i2['overlap_events']
    assert reads1[:2] == [(19, 49), (24, 54)] and reads2[0] == (19, 54)     # 겹친 두 끊김 → 한 번 읽기


def test_batch_flag_off_uses_one_by_one(monkeypatch):
    _, _, _, info, fake, reads, _ = run(monkeypatch, batched=True, batch=False)
    assert not info['batched'] and info['batches'] == 0
    assert fake.sessions == len(reads) == fake.propagations


def test_small_max_span_still_matches(monkeypatch):
    f1, p1, s1, _, _, _, _ = run(monkeypatch, batched=False)
    f2, p2, s2, info, _, _, _ = run(monkeypatch, batched=True, max_span=20)
    assert (f1, p1, s1) == (f2, p2, s2)
    assert info['batches'] >= info['break_events']        # 구간(31 프레임)이 max_span 보다 길어 묶이지 않음


def test_batched_progress_reaches_total(monkeypatch):
    *_, info, _, _, calls = run(monkeypatch, batched=True)
    total = calls[-1][1]
    assert calls[-1] == (total, total)
    assert all(a[0] <= b[0] for a, b in zip(calls, calls[1:]))       # 줄어들지 않음


# ---------------------------------------------------------------------------
# 영상 특징 캐시
# ---------------------------------------------------------------------------
def test_feature_cache_shares_position_embeddings():
    torch = pytest.importorskip('torch')
    cache = sv._FrameFeatureCache('cpu')
    pos = [torch.ones(4, 1, 2), torch.zeros(2, 1, 2)]
    for t in range(3):
        cache.cache_vision_features(t, {'vision_feats': [torch.full((4, 1, 2), float(t))],
                                        'vision_pos_embeds': [p.clone() for p in pos]})
    assert len(cache) == 3 and cache._pos == {}
    got = cache.get_vision_features(2)
    assert torch.equal(got['vision_feats'][0], torch.full((4, 1, 2), 2.0))
    assert got['vision_pos_embeds'][0] is cache.get_vision_features(0)['vision_pos_embeds'][0]
    # 위치 임베딩이 다른 프레임은 따로 저장
    cache.cache_vision_features(5, {'vision_feats': [torch.zeros(4, 1, 2)],
                                    'vision_pos_embeds': [torch.full((4, 1, 2), 3.0), pos[1]]})
    assert torch.equal(cache.get_vision_features(5)['vision_pos_embeds'][0], torch.full((4, 1, 2), 3.0))
    assert cache.get_vision_features(9) is None
    cache.clear_all()
    assert len(cache) == 0 and cache.get_vision_features(0) is None


def test_crowded_overlaps_share_one_session(monkeypatch):
    # 네 선수가 t≈30 근처에서 한데 모였다 흩어짐 → 겹침 장면 여러 곳이 시간상 겹쳐 한 세션에서 처리
    truth = {'A': lambda t: box_at(600 + 10 * t), 'B': lambda t: box_at(1200 - 10 * t),
             'C': lambda t: box_at(700 + 7 * t, y=510.0), 'D': lambda t: box_at(1110 - 7 * t, y=490.0)}
    ids = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    def make():
        return [[[ids[p], 0, *truth[p](t), 0.9] for p in truth] for t in range(80)]

    results = {}
    for name, cls in (('one', OneByOneTracker), ('many', AbsTruthTracker)):
        frames = make()
        fake = cls(truth)
        patch_reader(monkeypatch, fake, [])
        parent, swaps, info = sv.repair_tracks('clip', frames, (W, H), fake, fps=30.0)
        results[name] = (frames, parent, swaps, info, fake)
    one, many = results['one'], results['many']
    assert one[:3] == many[:3] and many[2] == []
    n_events = many[3]['overlap_events']
    assert n_events >= 3
    assert one[4].sessions == n_events and many[4].sessions == many[3]['batches'] == 1
    assert many[4].propagations <= n_events
