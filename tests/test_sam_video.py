"""SAM3 영상 추적 바로잡기 테스트 (tracking/sam_video.py, 10부)

실제 SAM3 는 내려받지 않는다. '진짜 선수 위치'를 알고 있는 가짜 영상 추적기로 끊김 잇기·ID 교체 바로잡기를 검증한다.
실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking import sam_video as sv

W, H = 1920, 1080


def box_at(x, y=500.0, w=60.0, h=160.0):
    return (x, y, x + w, y + h)


class TruthTracker:
    """
    선수별 실제 위치 함수 truth[pid](t) 를 알고, 프롬프트 박스와 가장 가까운 선수를 계속 따라가는 가짜 SAM3
    (첫 프레임 번호 = start 로 설정)
    """
    def __init__(self, truth):
        self.truth = truth
        self.start = 0
        self.calls = 0

    def track(self, frames_rgb, prompts):
        self.calls += 1
        owner = {}
        for oid, b in prompts.items():
            owner[oid] = min(self.truth, key=lambda p: abs(self.truth[p](self.start)[0] - b[0]))
        return [{oid: self.truth[p](self.start + k) for oid, p in owner.items()} for k in range(len(frames_rgb))]


def patch_reader(monkeypatch, tracker):
    def fake_read(path, start, end):
        tracker.start = start
        return [np.zeros((4, 4, 3), np.uint8) for _ in range(start, end + 1)]
    monkeypatch.setattr(sv, 'read_frames', fake_read)


def test_box_iou_and_border():
    assert sv.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert sv.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert sv.at_border((0, 100, 50, 200), W, H) and not sv.at_border((500, 100, 550, 200), W, H)


def test_break_event_is_linked_when_sam_keeps_the_player(monkeypatch):
    # 선수 P: 0~59 프레임 x=400+2t 로 이동. 추적기는 25~34 프레임(가림) 동안 놓치고 35 부터 새 ID 2 로 다시 잡음
    # 선수 Q: 멀리 있는 다른 선수 (ID 3)
    truth = {'P': lambda t: box_at(400 + 2 * t), 'Q': lambda t: box_at(1300)}
    frames = [[] for _ in range(60)]
    for t in range(60):
        if t < 25:
            frames[t].append([1, 0, *truth['P'](t), 0.9])
        elif t >= 35:
            frames[t].append([2, 0, *truth['P'](t), 0.9])
        frames[t].append([3, 0, *truth['Q'](t), 0.9])
    fake = TruthTracker(truth)
    patch_reader(monkeypatch, fake)
    parent, swaps, info = sv.repair_tracks('clip', frames, (W, H), fake, fps=30.0)
    assert parent == {2: 1} and not swaps
    assert {o[0] for objs in frames for o in objs} == {1, 3}
    assert info['tracks_before'] == 3 and info['tracks_after'] == 2 and info['linked'] == 1


def test_far_new_track_is_not_linked(monkeypatch):
    truth = {'P': lambda t: box_at(400), 'R': lambda t: box_at(1500)}
    frames = [[] for _ in range(50)]
    for t in range(50):
        if t < 20:
            frames[t].append([1, 0, *truth['P'](t), 0.9])         # P 는 사라짐 (가림 아님)
        if t >= 25:
            frames[t].append([2, 0, *truth['R'](t), 0.9])         # 멀리서 다른 선수가 새로 나타남
    fake = TruthTracker(truth)
    patch_reader(monkeypatch, fake)
    parent, swaps, info = sv.repair_tracks('clip', frames, (W, H), fake, fps=30.0)
    assert parent == {} and info['break_events'] == 0


def test_id_swap_after_crossing_is_fixed(monkeypatch):
    # 선수 P 는 오른쪽으로, Q 는 왼쪽으로 지나침 (t=30 에서 교차). 추적기는 교차 뒤 ID 를 바꿔 붙임
    truth = {'P': lambda t: box_at(500 + 10 * t), 'Q': lambda t: box_at(1100 - 10 * t)}
    frames = [[] for _ in range(70)]
    for t in range(70):
        p, q = truth['P'](t), truth['Q'](t)
        a, b = (p, q) if t <= 30 else (q, p)                       # t > 30: ID 1 이 Q 를, ID 2 가 P 를 따라감 (교체)
        frames[t] += [[1, 0, *a, 0.9], [2, 0, *b, 0.9]]
    fake = TruthTracker(truth)
    patch_reader(monkeypatch, fake)
    parent, swaps, info = sv.repair_tracks('clip', frames, (W, H), fake, fps=30.0)
    assert len(swaps) == 1 and swaps[0][:2] == (1, 2)
    at = swaps[0][2]
    # 바로잡은 뒤에는 ID 1 이 끝까지 P(오른쪽으로 가는 선수)
    for t in range(at, 70):
        box1 = next(o for o in frames[t] if o[0] == 1)[2:6]
        assert sv.box_iou(box1, truth['P'](t)) > 0.9, t
    ev = info['swap_evidence'][0]
    assert ev['crossed'] >= 5 and ev['mean_diff'] > 0


def test_players_who_really_cross_are_not_swapped(monkeypatch):
    # 실제로 교차하고 추적기도 맞게 따라간 경우 → 바꾸지 않음
    truth = {'P': lambda t: box_at(500 + 10 * t), 'Q': lambda t: box_at(1100 - 10 * t)}
    frames = [[[1, 0, *truth['P'](t), 0.9], [2, 0, *truth['Q'](t), 0.9]] for t in range(70)]
    fake = TruthTracker(truth)
    patch_reader(monkeypatch, fake)
    parent, swaps, info = sv.repair_tracks('clip', frames, (W, H), fake, fps=30.0)
    assert swaps == [] and info['overlap_events'] >= 1


def test_crop_bank_swap_moves_crops_after_the_swap():
    from tracking.identity import JerseyCropBank
    bank = JerseyCropBank()
    for t in (0, 10, 50, 60):
        bank.add(1, t, np.full((2, 2, 3), 1, np.uint8), 1.0)
        bank.add(2, t, np.full((2, 2, 3), 2, np.uint8), 1.0)
    bank.swap([(1, 2, 40)])
    own1 = {t: int(c[0, 0, 0]) for _, t, c in bank.items[1]}
    assert own1 == {0: 1, 10: 1, 50: 2, 60: 2}


def test_profile_enables_sam_video_only_with_postprocess():
    from pipeline.main_pipeline import Soccer3DPipeline
    p = Soccer3DPipeline.__new__(Soccer3DPipeline)
    p.device = 'cuda'
    p.config = {'analysis': {'profiles': {'precise': {'postprocess': True, 'sam_video': True},
                                          'realtime': {'postprocess': False, 'sam_video': True}}}}
    assert p.analysis_profile('precise')['sam_video'] is True
    assert p.analysis_profile('realtime')['sam_video'] is False
