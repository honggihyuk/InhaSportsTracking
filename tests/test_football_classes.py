"""축구 전용 탐지 클래스(골키퍼·심판) 처리 테스트 (8부)

파인튜닝한 RF-DETR 이 내는 goalkeeper / referee 라벨이 팀 분류·역할·설정에서 자연스럽게 처리되는지 확인한다.
실제 모델은 내려받지 않는다.
실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import postprocess as pp, team_embed, video_analysis as va
from tracking.detector import BALL_OFFSET, Detection
from tracking.rfdetr_detector import RFDETRDetector
from tracking.tracker import SoccerTracker


def test_majority_labels_needs_enough_votes():
    labels = team_embed.majority_labels({1: {'player': 5, 'referee': 1}, 2: {'referee': 4}, 3: {'goalkeeper': 2}})
    assert labels == {1: 'player', 2: 'referee'}                       # 3 표 미만 트랙은 판단하지 않음
    assert team_embed.majority_labels({3: {'goalkeeper': 2}}, 1) == {3: 'goalkeeper'}


def test_goalkeeper_label_assigns_team_without_box_rule():
    world = []
    for t in range(30):
        # 9: 탐지 모델이 골키퍼로 본 트랙 — 박스 밖(스위퍼 키퍼)이어도 왼쪽 반이면 왼쪽 골문 팀
        world.append([[1, -20, 0], [2, -15, 5], [3, 20, 0], [4, 25, -5], [9, -30, 2], [7, 0, 10], [8, -48, 0]])
    teams = {'1': 'home', '2': 'home', '3': 'away', '4': 'away', '9': 'other', '7': 'other', '8': 'other'}
    labels = {'9': 'goalkeeper', '7': 'referee'}
    roles = pp.assign_goalkeepers(world, teams, 30.0, labels)
    assert roles['9'] == 'goalkeeper' and teams['9'] == 'home'
    assert roles['7'] == 'referee' and teams['7'] == 'other'           # 심판은 팀 없이 역할만
    assert roles['8'] == 'goalkeeper'                                  # 라벨 없는 기존 규칙(박스 안 1 초)도 그대로
    # 라벨 없이는 박스 밖 트랙을 골키퍼로 보지 않음
    teams2 = {'1': 'home', '2': 'home', '3': 'away', '4': 'away', '9': 'other'}
    assert '9' not in pp.assign_goalkeepers(world, teams2, 30.0)


class LabelledDetector:
    """선수 2 명(한 명은 심판 라벨) + 공을 내는 가짜 축구 전용 탐지기"""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        x = 100 + self.calls * 6
        self.calls += 1
        return [Detection((x, 150, x + 20, 200), 0.9, 0, 'player_player', 'player'),
                Detection((300, 150, 320, 200), 0.9, 2, 'player_referee', 'player'),
                Detection((250, 250, 258, 258), 0.8, BALL_OFFSET, 'ball_ball', 'ball')]


def test_analyze_video_records_labels_and_keeps_referee_out_of_teams(tmp_path):
    from tests.test_analysis import write_clip
    clip = tmp_path / "clip.avi"
    write_clip(clip)
    res = va.analyze_video(clip, LabelledDetector(), SoccerTracker(tracker_type='simple'))
    labels = res['track_labels']
    referee = [tid for tid, lab in labels.items() if lab == 'referee']
    assert referee and 'player' in labels.values()
    assert all(res['teams'][tid] == 'other' for tid in referee)


def test_missing_checkpoint_falls_back_to_coco(tmp_path, capsys):
    class Model:
        class_names = {1: 'person', 37: 'sports ball'}
        model_config = None

        def predict(self, *a, **k):
            return []
    det = RFDETRDetector(checkpoint=str(tmp_path / 'nope.pth'), model=Model())
    assert det.checkpoint is None and '파인튜닝 가중치가 없습니다' in capsys.readouterr().out
    real = tmp_path / 'best.pth'
    real.write_bytes(b'x')
    assert RFDETRDetector(checkpoint=str(real), model=Model()).checkpoint == str(real)


def test_pipeline_passes_resolution_and_checkpoint(monkeypatch):
    from pipeline.main_pipeline import Soccer3DPipeline
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr('tracking.rfdetr_detector.RFDETRDetector', Spy)
    p = Soccer3DPipeline.__new__(Soccer3DPipeline)
    p.config = {'detection': {'backend': 'rfdetr', 'rfdetr': {'size': 'medium', 'checkpoint': 'models/x.pth',
                                                             'resolution': 960}}}
    p.device, p.detector = 'cuda', None
    p.initialize_detector()
    assert seen['resolution'] == 960 and seen['checkpoint'] == 'models/x.pth'
