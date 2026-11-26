"""등번호 인식·신원 부여 테스트 (tracking/identity.py, 9부)

실제 GLM-OCR 은 내려받지 않는다. 크롭에 그린 숫자를 '읽는' 가짜 판독기로 투표·중복 정리·재연결을 검증한다.
실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import video_analysis as va
from tracking import identity as idn
from tracking.tracker import SoccerTracker


def test_parse_number_accepts_only_1_to_99():
    assert idn.parse_number('14') == 14 and idn.parse_number(' #7 ') == 7
    for bad in ('', '0', '100', 'A1', '1 4', '```markdown', '22.', None):
        assert idn.parse_number(bad) is None


def test_vote_needs_repeated_agreement():
    assert idn.vote([(14, 0.9), (14, 0.8), (11, 0.6), (None, 0.5)])[0] == 14
    assert idn.vote([(14, 0.9)])[0] is None                                  # 한 번만 읽힘
    assert idn.vote([(14, 0.9), (14, 0.9), (22, 0.9), (22, 0.9), (22, 0.9)])[0] == 22
    assert idn.vote([(14, 0.3), (14, 0.3)])[0] is None                       # 신뢰도 낮음
    assert idn.vote([(2, 0.9), (2, 0.9)] + [(None, 0.5)] * 10)[0] is None    # 크롭 대부분이 안 읽힘
    assert idn.vote([])[0] is None


def frame_with_box(boxes, h=1080, w=1920):
    img = np.full((h, w, 3), (40, 140, 40), np.uint8)
    rng = np.random.default_rng(0)
    for x1, y1, x2, y2 in boxes:
        img[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 3))      # 선명한 질감
    return img


def test_jersey_crop_filters_small_blurry_and_overlapping():
    a, b = (100, 100, 160, 260), (500, 100, 560, 260)
    img = frame_with_box([a, b])
    crop, q = idn.jersey_crop(img, a, [a, b])
    assert crop.shape[0] == int(0.6 * 160) - int(0.12 * 160) and q > 0
    assert idn.jersey_crop(img, (100, 100, 120, 150), [a]) is None           # 작음
    assert idn.jersey_crop(img, a, [a, (150, 200, 200, 300)]) is None        # 다른 선수와 겹침
    flat = np.full_like(img, 120)
    assert idn.jersey_crop(flat, a, [a]) is None                              # 흐림(질감 없음)
    assert idn.jersey_crop(img, (0, 100, 60, 260), []) is None                # 화면 가장자리에 잘림


def test_crop_bank_keeps_best_and_merges():
    bank = idn.JerseyCropBank(per_track=2)
    for q in (1, 5, 3):
        bank.add(7, q, np.zeros((2, 2, 3)), q)
    assert sorted(q for q, _, _ in bank.items[7]) == [3, 5]
    bank.add(9, 0, np.zeros((2, 2, 3)), 9)
    bank.merge({9: 7})
    assert 9 not in bank.items and sorted(q for q, _, _ in bank.items[7]) == [5, 9]


class FakeReader:
    """크롭 왼쪽 위 픽셀 값에 번호를 숨겨 둔 가짜 OCR (0 이면 읽지 못함 '#')"""
    model_id = 'fake'

    def __call__(self, crops):
        out = []
        for c in crops:
            n = int(c[0, 0, 0])
            out.append((str(n) if n else '#', 0.9))
        return out


def crop_of(n):
    c = np.zeros((10, 10, 3), np.uint8)
    c[0, 0, 0] = n
    return c


def frames_for(spans):
    """spans: {track_id: (start, end)} → frames (선수 박스만)"""
    n = max(e for _, e in spans.values()) + 1
    frames = [[] for _ in range(n)]
    for tid, (s, e) in spans.items():
        for t in range(s, e + 1):
            frames[t].append([tid, 0, 10 * tid, 10, 10 * tid + 5, 40, 0.9])
    return frames


def test_identify_votes_dedups_and_relinks():
    # 1·2: 같은 팀 14 번, 시간상 안 겹침 → 같은 선수로 재연결 / 3: 같은 팀 14 번이 1 과 동시에 → 신뢰도 낮아 미확정
    # 4: 다른 팀 14 번 → 그대로 / 5: '1' 만 읽힘 (골키퍼 아님) → 미확정 / 6: 골키퍼 1 번 → 확정
    spans = {1: (0, 9), 2: (20, 29), 3: (5, 12), 4: (0, 29), 5: (0, 29), 6: (0, 29)}
    frames = frames_for(spans)
    teams = {1: 'home', 2: 'home', 3: 'home', 4: 'away', 5: 'home', 6: 'other'}
    bank = idn.JerseyCropBank()
    reads = {1: [14, 14, 14], 2: [14, 14], 3: [14, 14, 0], 4: [14, 14], 5: [1, 1, 1], 6: [1, 1]}
    for tid, nums in reads.items():
        for k, n in enumerate(nums):
            bank.add(tid, k, crop_of(n), 10 - k)
    ids, parent, info = idn.identify(bank, FakeReader(), frames, teams, goalkeepers=[6])
    assert parent == {2: 1}                                              # 2 → 1 로 재연결
    assert all(o[0] != 2 for objs in frames for o in objs)               # frames 의 ID 도 바뀜
    assert ids[1]['jersey_number'] == 14 and ids[1]['merged'] == [2] and 2 not in ids
    assert ids[3]['jersey_number'] is None and info['votes'][3].get('conflict')
    assert ids[4]['jersey_number'] == 14
    assert ids[5]['jersey_number'] is None and ids[6]['jersey_number'] == 1
    s = info['summary']
    assert s['relinked'] == 1 and s['tracks_before'] == 6 and s['tracks_after'] == 5 and s['conflicts'] == 1


def test_analyze_video_attaches_identities(tmp_path):
    from tests.test_analysis import write_clip

    class BigDetector:
        """번호를 읽을 수 있을 만큼 큰 선수 1 명"""
        def detect(self, frame):
            from tracking.detector import Detection
            return [Detection((100, 40, 160, 200), 0.9, 0, 'player_player', 'player')]

    class Reader:
        model_id = 'fake'

        def __call__(self, crops):
            return [('9', 0.95) for _ in crops]
    clip = tmp_path / 'clip.avi'
    write_clip(clip, n=12)
    res = va.analyze_video(clip, BigDetector(), SoccerTracker(tracker_type='simple'), jersey_reader=Reader())
    assert res['identity']['model'] == 'fake'
    tid = next(o[0] for o in res['frames'][0] if o[1] == va.KIND_PLAYER)
    assert res['identities'][str(tid)]['jersey_number'] == 9
    # 판독기가 실패해도 분석은 계속
    class Broken:
        model_id = 'broken'

        def __call__(self, crops):
            raise RuntimeError('no model')
    res2 = va.analyze_video(clip, BigDetector(), SoccerTracker(tracker_type='simple'), jersey_reader=Broken())
    assert res2['identities'] is None and 'error' in res2['identity']


def test_stats_and_tools_use_jersey_numbers():
    from agent.tools import MatchTools
    from analysis import stats as match_stats
    world = [[[1, 0.1 * t, 0.0, 3.0], [2, 5.0, 5.0, 0.0]] for t in range(60)]
    result = {'fps': 30.0, 'world': world, 'teams': {'1': 'home', '2': 'away'}, 'roles': {},
              'identities': {'1': {'jersey_number': 14, 'confidence': 0.9, 'evidence': 3, 'readings': 4},
                             '2': {'jersey_number': None, 'confidence': 0.2, 'evidence': 0, 'readings': 4}},
              'frame_count': 60, 'frames': [[] for _ in range(60)]}
    result['stats'] = match_stats.compute(result)
    assert result['stats']['players']['1']['jersey_number'] == 14
    assert 'jersey_number' not in result['stats']['players']['2']
    tools = MatchTools(result)
    out = tools.player_stats(jersey_number=14)
    assert out['player_id'] == 1 and out['jersey_number'] == 14
    assert 'error' in tools.player_stats(jersey_number=7)
