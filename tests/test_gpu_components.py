"""GPU 고도화 구성 요소 테스트 — RF-DETR 백엔드, SAM 발 위치, SigLIP2 팀 분류, VLM 에이전트

실제 모델(RF-DETR, SAM, SigLIP2, Qwen3-VL)은 내려받지 않는다. 모델 자리에 같은 인터페이스의 가짜를 넣어
변환·판정·도구·대화 루프 로직만 검증한다.
실행: python -m pytest tests -q
"""
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import parse_tool_calls
from agent.runner import MatchAgent
from agent.tools import MatchTools
from pipeline import team_embed, video_analysis as va
from pipeline.main_pipeline import Soccer3DPipeline, resolve_device
from tracking.detector import BALL_OFFSET, Detection
from tracking.rfdetr_detector import RFDETRDetector
from tracking.sam_refiner import MaskResult, mask_foot, plausible
from tracking.tracker import SoccerTracker

RNG = np.random.default_rng(0)


# --- RF-DETR 백엔드 -----------------------------------------------------------
class FakeDets:
    def __init__(self, rows):
        a = np.array(rows, float).reshape(-1, 6)
        self.xyxy, self.confidence, self.class_id = a[:, :4], a[:, 4], a[:, 5].astype(int)

    def __len__(self):
        return len(self.xyxy)


class FakeRFDETR:
    """COCO 클래스 이름(1: person, 37: sports ball, 3: car)을 가진 가짜 RF-DETR"""
    class_names = {1: 'person', 3: 'car', 37: 'sports ball'}
    model_config = types.SimpleNamespace(patch_size=16, num_windows=2)

    def __init__(self):
        self.calls = []

    def predict(self, images, threshold=0.5, shape=None, include_source_image=True):
        batch = images if isinstance(images, list) else [images]
        self.calls.append((len(batch), shape, threshold))
        out = []
        for im in batch:
            h, w = im.shape[:2]
            rows = [[10, 10, 30, 60, 0.9, 1],     # 선수
                    [40, 40, 46, 46, 0.3, 37],    # 공
                    [0, 0, 5, 5, 0.95, 3],        # 자동차 → 버림
                    [60, 10, 80, 60, 0.35, 1]]    # 점수 낮은 선수 → 버림
            out.append(FakeDets([r for r in rows if r[4] >= threshold]))
        return out if isinstance(images, list) else out[0]


def test_rfdetr_maps_coco_classes_and_thresholds():
    model = FakeRFDETR()
    det = RFDETRDetector(model=model, confidence_threshold=0.4, ball_threshold=0.25, imgsz=950)
    dets = det.detect(np.zeros((200, 300, 3), np.uint8))
    players = [d for d in dets if not d.is_ball]
    balls = [d for d in dets if d.is_ball]
    assert [d.bbox for d in players] == [(10, 10, 30, 60)] and players[0].class_name == 'player_person'
    assert [d.bbox for d in balls] == [(40, 40, 46, 46)] and balls[0].class_id == BALL_OFFSET
    assert model.calls[0][1] == (960, 960)          # 해상도는 패치(16×2) 배수로 맞춤


def test_rfdetr_ball_tiles_shift_to_frame_coords():
    model = FakeRFDETR()
    det = RFDETRDetector(model=model, ball_tile=100, tile_overlap=0.0)
    dets = det.detect(np.zeros((200, 200, 3), np.uint8))
    balls = sorted(d.bbox for d in dets if d.is_ball)
    assert model.calls[1][0] == 4                   # 타일 4 장을 한 번에 배치 추론
    assert (140, 140, 146, 146) in balls and (40, 40, 46, 46) in balls


def test_rfdetr_pretrained_uses_original_coco_ids():
    from rfdetr.assets.coco_classes import COCO_CLASSES

    class Pretrained(FakeRFDETR):
        class_names = list(COCO_CLASSES.values())   # 80 개 목록이지만 예측 ID 는 COCO 원래 번호
    det = RFDETRDetector(model=Pretrained())
    kinds = sorted((d.type, d.bbox) for d in det.detect(np.zeros((100, 100, 3), np.uint8)))
    assert kinds == [('ball', (40, 40, 46, 46)), ('player', (10, 10, 30, 60))]


def test_rfdetr_finetuned_labels():
    class Tuned(FakeRFDETR):
        class_names = ['ball', 'goalkeeper', 'player', 'referee']

        def predict(self, images, threshold=0.5, shape=None, include_source_image=True):
            return [FakeDets([[0, 0, 10, 30, 0.9, 1], [20, 0, 30, 30, 0.9, 3]])]
    labels = sorted(d.label for d in RFDETRDetector(model=Tuned()).detect(np.zeros((50, 50, 3), np.uint8)))
    assert labels == ['goalkeeper', 'referee']


def test_pipeline_device_and_backend_auto(monkeypatch):
    import pipeline.main_pipeline as mp
    monkeypatch.setattr(mp, 'cuda_available', lambda: False)
    assert resolve_device('auto') == 'cpu' and resolve_device('cuda') == 'cuda'
    p = Soccer3DPipeline(config_path='configs/model_config.yaml')
    assert p.device == 'cpu' and p.detector_backend() == 'yolo'       # CPU 면 YOLO
    prof = p.analysis_profile('precise')
    assert prof['team_classifier'] == 'color' and prof['sam_refine'] is False
    p.device = 'cuda'
    monkeypatch.setattr(mp, '_installed', lambda m: True)
    assert p.detector_backend() == 'rfdetr'
    prof = p.analysis_profile('precise')
    assert prof['team_classifier'] == 'siglip' and prof['sam_refine'] is True
    assert p.analysis_profile('realtime')['sam_refine'] is False


# --- SAM 발 위치 ---------------------------------------------------------------
def test_mask_foot_uses_lowest_rows():
    m = np.zeros((100, 60), bool)
    m[10:80, 20:40] = True       # 몸통
    m[80:95, 30:36] = True       # 한쪽 다리가 더 아래
    foot, box = mask_foot(m, x0=100, y0=200)
    assert foot == pytest.approx((132.5, 294.0)) and box == (120, 210, 139, 294)
    assert mask_foot(np.zeros((10, 10), bool)) is None


def test_mask_plausibility():
    assert plausible((0, 0, 20, 60), (2, 2, 18, 58), 0.6)
    assert not plausible((0, 0, 20, 60), (2, 2, 18, 20), 0.6)   # 상반신만 잡힘 → 발 위치 부정확
    assert not plausible((0, 0, 20, 60), (2, 2, 18, 58), 0.05)  # 면적이 너무 작음


def test_sam_falls_back_when_model_is_gated():
    from tracking.sam_refiner import SamRefiner
    r = SamRefiner('facebook/sam3', device='cpu', fallback='facebook/sam2.1-hiera-large')
    loaded = []

    def load(mid):
        if 'sam3' in mid:
            raise OSError('You are trying to access a gated repo.')
        loaded.append(mid)
    r._load_model = load
    r._load()
    assert loaded == ['facebook/sam2.1-hiera-large'] and r.model_id == 'facebook/sam2.1-hiera-large'


def test_team_clusters_put_referees_in_third_cluster():
    """심판·골키퍼가 여럿이면 k=2 가 '선수 전체 vs 심판' 으로 나뉠 수 있다 → k=3 으로 분리"""
    crops = {i: kit((230, 230, 230)) for i in range(1, 9)}
    crops.update({i: kit((60, 20, 120)) for i in range(10, 18)})
    crops.update({i: kit((20, 230, 230)) for i in range(20, 24)})       # 노랑 심판·골키퍼 4 트랙
    teams, _, _ = team_embed.classify_teams(crops, color_embedder)
    assert {teams[i] for i in range(1, 9)} | {teams[i] for i in range(10, 18)} == {'home', 'away'}
    assert len({teams[i] for i in range(1, 9)}) == 1 and len({teams[i] for i in range(10, 18)}) == 1
    assert {teams[i] for i in range(20, 24)} == {'other'}


def test_calibrate_prefers_sam_foot():
    from tests.test_analysis import RECT_POINTS, fake_result
    frames = [[[1, 0, 480, 300, 520, 340, 0.9, 500.0, 340.0],   # SAM 발 = 경기장 중앙
               [2, 0, 480, 300, 520, 340, 0.9]]]                # 발 위치 없음 → 박스 하단 중앙
    w = va.calibrate(fake_result(frames), [{'frame': 0, 'points': RECT_POINTS}])['world'][0]
    assert w[0][1] == pytest.approx(0, abs=0.01) and w[1][1] == pytest.approx(0, abs=0.01)
    frames = [[[1, 0, 480, 300, 520, 340, 0.9, 600.0, 340.0]]]
    w = va.calibrate(fake_result(frames), [{'frame': 0, 'points': RECT_POINTS}])['world'][0]
    assert w[0][1] == pytest.approx(10.5, abs=0.05)              # 박스가 아니라 SAM 발 위치를 씀


# --- SigLIP2 팀 분류 ------------------------------------------------------------
def color_embedder(crops):
    """크롭 평균색을 임베딩으로 쓰는 가짜 SigLIP"""
    e = np.array([c.reshape(-1, 3).mean(axis=0) for c in crops], np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def kit(bgr, n=6):
    return [np.clip(np.full((96, 96, 3), bgr, float) + RNG.normal(0, 6, (96, 96, 3)), 0, 255).astype(np.uint8)
            for _ in range(n)]


def test_classify_teams_with_embeddings():
    crops = {i: kit((230, 230, 230)) for i in range(1, 8)}               # 흰 유니폼 7 명
    crops.update({i: kit((60, 20, 120)) for i in range(10, 16)})        # 자주색 6 명
    crops[20] = kit((20, 230, 230))                                     # 노랑 심판(라벨로 제외)
    crops[30] = kit((230, 230, 230), 2)                                 # 크롭 부족
    labels = {20: {'referee': 9, 'person': 1}}
    teams, colors, info = team_embed.classify_teams(crops, color_embedder, track_labels=labels)
    assert {teams[i] for i in range(1, 8)} == {'home'} and {teams[i] for i in range(10, 16)} == {'away'}
    assert teams[20] == 'other' and teams[30] == 'other'
    assert info['method'] == 'siglip' and info['referee_tracks'] == 1
    json.dumps(info)                                                    # 결과 JSON 저장 가능
    assert set(colors) == {'home', 'away'} and colors['home'].startswith('#')


def test_jersey_crop_masks_background():
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[:, :] = (0, 255, 0)
    mask = np.zeros((100, 40), bool)
    mask[:, 10:30] = True
    crop = team_embed.jersey_crop(frame, (50, 50, 90, 150), mask)
    assert crop.shape == (96, 96, 3)
    assert (crop[48, 2] == 128).all()            # 마스크 밖 잔디는 회색으로 지움
    assert team_embed.jersey_crop(frame, (0, 0, 4, 4)) is None


class FakeRefiner:
    model_id = 'fake-sam'

    def refine(self, frame, boxes, return_masks=False):
        return [MaskResult((b[0] + 3.0, float(b[3])), tuple(b),
                           np.ones((b[3] - b[1], b[2] - b[0]), bool) if return_masks else None) for b in boxes]


def test_analyze_video_with_embedder_and_refiner(tmp_path):
    from tests.test_analysis import ScriptedDetector, write_clip
    clip = tmp_path / "clip.avi"
    write_clip(clip)
    seen = []
    emb = lambda crops: (seen.append(len(crops)), color_embedder(crops))[1]
    res = va.analyze_video(clip, ScriptedDetector(), SoccerTracker(tracker_type='simple'),
                           team_embedder=emb, refiner=FakeRefiner())
    player = next(o for o in res['frames'][0] if o[1] == va.KIND_PLAYER)
    assert len(player) == 9 and player[7] == player[2] + 3.0     # 발 위치가 객체 끝에 붙음
    assert res['sam']['feet'] >= 9 and res['team_method']['method'] == 'siglip'
    ball = next(o for o in res['frames'][0] if o[1] == va.KIND_BALL)
    assert len(ball) == 7                                          # 공은 SAM 대상 아님


def test_analyze_video_falls_back_to_color_when_embedder_fails(tmp_path, monkeypatch):
    from tests.test_analysis import ScriptedDetector, write_clip
    clip = tmp_path / "clip.avi"
    write_clip(clip)

    def broken(*a, **k):
        raise RuntimeError("no model")
    monkeypatch.setattr(team_embed, 'classify_teams', broken)
    res = va.analyze_video(clip, ScriptedDetector(), SoccerTracker(tracker_type='simple'),
                           team_embedder=color_embedder)
    assert res['team_method'] == {'method': 'color', 'fallback': 'no model'}
    assert set(res['teams'].values()) <= {'home', 'away', 'other'}


# --- VLM 에이전트 ----------------------------------------------------------------
def match_result(tmp_path=None):
    fps, n = 10.0, 100
    world = [[[0, float(t) * 0.1, 0.0, 1.0], [7, -10.0 + t * 0.2, 5.0, 2.0]] for t in range(n)]
    stats = {
        'players': {'7': {'team': 'home', 'role': 'player', 'frames': 100, 'seconds': 10.0, 'distance': 20.0,
                          'high_speed_distance': 0.0, 'max_speed': 7.5, 'avg_speed': 2.0,
                          'sprints': [{'start': 30, 'end': 45, 'max_speed': 7.5}], 'mean_position': [0, 5]},
                    '8': {'team': 'away', 'role': 'player', 'frames': 50, 'seconds': 5.0, 'distance': 35.0,
                          'high_speed_distance': 3.0, 'max_speed': 6.0, 'avg_speed': 3.0, 'sprints': [],
                          'mean_position': [5, 5]}},
        'teams': {'home': {'players': 1}, 'away': {'players': 1}},
        'possession': {'share': {'home': 0.6, 'away': 0.4},
                       'segments': [[0, 59, 'home', 7], [60, 99, 'away', 8]]},
        'events': [{'type': 'pass', 'frame': 20, 'end_frame': 25, 'team': 'home', 'from': 7, 'to': 9},
                   {'type': 'sprint', 'frame': 30, 'end_frame': 45, 'team': 'home', 'player': 7, 'max_speed': 7.5}],
        'shape': [{'frame': 0, 'home': {'centroid': [0, 0], 'width': 30, 'length': 20, 'defensive_line': -20,
                                        'players': 5}},
                  {'frame': 50, 'home': {'centroid': [2, 0], 'width': 40, 'length': 30, 'defensive_line': -10,
                                         'players': 5}}],
    }
    frames = [[[7, 0, 100, 100, 120, 160, 0.9], [0, 1, 200, 200, 206, 206, 0.8]] for _ in range(n)]
    return {'video': 'm.avi', 'fps': fps, 'frame_count': n, 'frames': frames, 'world': world,
            'teams': {'7': 'home', '8': 'away'}, 'team_colors': {'home': '#ff0000'}, 'stats': stats,
            'calibration': {'keyframes': []}}


def test_tools_answer_from_data():
    t = MatchTools(match_result())
    assert t.player_stats(player_id=7)['max_speed'] == 7.5
    assert t.player_stats(sort_by='distance', top=1)['players'][0]['player_id'] == 8
    poss = t.get_possession(start_s=3.0, end_s=8.0)
    assert poss['share'] == {'home': pytest.approx(30 / 51, abs=1e-3), 'away': pytest.approx(21 / 51, abs=1e-3)}
    ev = t.find_events(type='sprint')
    assert ev['count'] == 1 and ev['events'][0]['time_s'] == 3.0 and ev['frames'] == [30]
    assert t.team_shape('home', time_s=4.0)['width'] == 40
    assert t.team_shape('home')['average']['defensive_line'] == -15
    tr = t.query_tracks(7, start_s=0, end_s=2, step_s=1)
    assert [p['time_s'] for p in tr['points']] == [0.0, 1.0, 2.0]
    assert 'error' in t.call('player_stats', {'player_id': 99})
    assert 'error' in t.call('rm_rf', {})
    assert 'error' in t.call('team_shape', {'bogus': 1})


def test_image_tools_read_video(tmp_path):
    from tests.test_analysis import write_clip
    clip = tmp_path / "m.avi"
    write_clip(clip, n=12)
    res = match_result()
    res['frame_count'], res['frames'] = 12, res['frames'][:12]
    t = MatchTools(res, clip)
    fr = t.get_frame(time_s=0.5)
    assert fr['frame'] == 5 and fr['image'].shape[2] == 3
    crop = t.crop_player(7, time_s=0.2)
    assert crop['image'].shape[0] == 384 and crop['frame'] == 2
    assert 'error' in MatchTools(res).call('get_frame', {'frame': 1})   # 영상 없음


def test_parse_tool_calls():
    text = ('<think>hmm</think>확인할게요<tool_call>\n{"name": "player_stats", "arguments": {"player_id": 7}}\n'
            '</tool_call><tool_call>{"name": "find_events", "arguments": "{\\"type\\": \\"pass\\"}"}</tool_call>')
    content, calls = parse_tool_calls(text)
    assert content == '확인할게요'
    assert calls == [{'name': 'player_stats', 'arguments': {'player_id': 7}},
                     {'name': 'find_events', 'arguments': {'type': 'pass'}}]


class ScriptedLLM:
    """정해진 순서로 tool call → 최종 답을 내는 가짜 VLM"""
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None):
        self.seen.append(messages)
        return self.replies.pop(0)


def test_agent_loop_calls_tools_and_collects_frames(tmp_path):
    from tests.test_analysis import write_clip
    clip = tmp_path / "m.avi"
    write_clip(clip, n=50)
    res = match_result()
    res['frame_count'], res['frames'] = 50, res['frames'][:50]
    llm = ScriptedLLM([
        {'content': '', 'tool_calls': [{'name': 'find_events', 'arguments': {'type': 'sprint'}},
                                       {'name': 'get_frame', 'arguments': {'frame': 3}}]},
        {'content': '#7 이 3.0초에 스프린트했습니다.', 'tool_calls': []},
    ])
    out = MatchAgent(llm, MatchTools(res, clip)).ask('스프린트 언제?')
    assert out['answer'].startswith('#7') and [s['tool'] for s in out['steps']] == ['find_events', 'get_frame']
    assert out['frames'][:2] == [30, 3]
    last = llm.seen[1]
    assert last[-1]['role'] == 'user' and any(c['type'] == 'image' for c in last[-1]['content'])  # 이미지를 보여줌
    assert any(m['role'] == 'tool' and '"count": 1' in m['content'] for m in last)
    assert 'image' not in json.dumps(out['steps'][1]['result'])                                 # 응답에 이미지 없음


def test_agent_stops_after_max_steps():
    loop = {'content': '', 'tool_calls': [{'name': 'match_overview', 'arguments': {}}]}
    llm = ScriptedLLM([loop, loop, {'content': '요약입니다', 'tool_calls': []}])
    out = MatchAgent(llm, MatchTools(match_result()), max_steps=2).ask('요약')
    assert out['answer'] == '요약입니다' and len(out['steps']) == 2


def test_agent_nudges_answer_without_tools():
    llm = ScriptedLLM([
        {'content': '#10 이 제일 빨라요', 'tool_calls': []},                      # 도구 없이 지어낸 답
        {'content': '', 'tool_calls': [{'name': 'player_stats', 'arguments': {'sort_by': 'max_speed', 'top': 1}}]},
        {'content': '#7 이 7.5 m/s 로 가장 빠릅니다.', 'tool_calls': []},
    ])
    out = MatchAgent(llm, MatchTools(match_result())).ask('제일 빠른 선수?')
    assert out['answer'].startswith('#7') and [s['tool'] for s in out['steps']] == ['player_stats']
    greet = ScriptedLLM([{'content': '안녕하세요', 'tool_calls': []}, {'content': '안녕하세요', 'tool_calls': []}])
    assert MatchAgent(greet, MatchTools(match_result())).ask('안녕')['answer'] == '안녕하세요'


def test_ask_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import server
    from tests.test_analysis import write_clip
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(server, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(server, "analysis_jobs", {})
    write_clip(tmp_path / "m.avi")
    llm = ScriptedLLM([{'content': '', 'tool_calls': [{'name': 'get_possession', 'arguments': {}}]},
                       {'content': 'home 60%', 'tool_calls': []}])
    monkeypatch.setattr(server, "agent_llm", llm)
    with TestClient(server.app) as c:
        assert c.post("/analysis/m.avi/ask", json={"question": "점유율?"}).status_code == 404  # 분석 전
        va.save(match_result(), tmp_path / "analysis" / "m.avi.json")
        assert c.post("/analysis/m.avi/ask", json={"question": " "}).status_code == 400
        r = c.post("/analysis/m.avi/ask", json={"question": "점유율?"})
        assert r.status_code == 200 and r.json()['answer'] == 'home 60%' and r.json()['frames'] == [0, 60]
        server.analysis_jobs["m.avi"] = {"state": "running"}
        assert c.post("/analysis/m.avi/ask", json={"question": "점유율?"}).status_code == 409
