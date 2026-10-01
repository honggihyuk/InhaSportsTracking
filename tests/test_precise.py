"""정밀 분석 모드 테스트 — 라인 정렬 보정, 공 타일 추론, 오프라인 후처리, 경기 지표, API

합성 중계 영상(pipeline/synthetic.py)은 프레임별 정답 호모그래피를 함께 주므로 보정 정확도를 수치로 검증한다.
실행: python -m pytest tests -q
"""
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import stats as match_stats
from pipeline import pitch, postprocess as pp, synthetic as syn, video_analysis as va
from tracking import detector as detector_mod
from tracking.detector import BALL_OFFSET, Detection


@pytest.fixture(scope="module")
def clip():
    return syn.render_clip(n=40, seed=3)


def small(frame, s=0.5):
    return cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def masks_for(clip, t, s=0.5):
    return pitch.line_mask(small(clip.frames[t], s), [b for _, _, b in clip.boxes[t]], box_scale=s)


def motions(clip, s=0.5):
    out, prev = [None], None
    for i, f in enumerate(clip.frames):
        g = cv2.cvtColor(small(f, s), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            M = va.camera_motion(prev, g, [b for _, _, b in clip.boxes[i - 1]], s)
            out.append(None if M is None else M.ravel().tolist())
        prev = g
    return out


# --- 라인 모델 · 라인 검출 ----------------------------------------------------
def test_pitch_line_samples_lie_on_lines():
    P, T = pitch.sample_pitch_lines(1.0)
    assert len(P) > 500 and np.allclose(np.linalg.norm(T, axis=1), 1)
    assert np.all(np.abs(P[:, 0]) <= 52.5 + 1e-6) and np.all(np.abs(P[:, 1]) <= 34 + 1e-6)
    # 센터서클 점은 원점에서 9.15 m
    circle = P[(np.abs(P[:, 0]) < 9.2) & (np.abs(P[:, 1]) < 9.2) & (np.abs(P[:, 0]) > 0.1)]
    assert np.allclose(np.hypot(circle[:, 0], circle[:, 1]), 9.15, atol=0.05)


def test_line_mask_finds_lines_not_stands(clip):
    mask, valid = masks_for(clip, 0)
    G = pitch.scale_matrix(0.5) @ clip.G[0]
    P, _ = pitch.sample_pitch_lines(0.5)
    q = P @ G[:, :2].T + G[:, 2]
    uv = np.round(q[:, :2] / q[:, 2:3]).astype(int)
    h, w = mask.shape
    inside = (uv[:, 0] >= 2) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 2) & (uv[:, 1] < h - 2)
    near = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    hit = near[uv[inside, 1], uv[inside, 0]] > 0
    assert hit.mean() > 0.8                                 # 보이는 라인 대부분 검출
    stands_rows = valid.max(axis=1) == 0                    # 경기장 영역이 전혀 없는 줄(관중석)
    assert stands_rows.any() and mask[stands_rows].sum() == 0


# --- 라인 정렬 ---------------------------------------------------------------
def perturb(H, dx=6.0, rot=0.004, zoom=1.01):
    """이미지 쪽에서 약간 이동·회전·확대한 잘못된 보정"""
    c, s = np.cos(rot), np.sin(rot)
    A = np.array([[zoom * c, -zoom * s, dx], [zoom * s, zoom * c, -dx / 2], [0, 0, 1]])
    return H @ A


def test_refine_recovers_perturbed_homography(clip):
    H_true = clip.H[5]
    H_bad = perturb(H_true)
    before = syn.pitch_error(H_bad, clip.G[5], 960, 540)
    r = pitch.LineRefiner(scale=0.5).refine(H_bad, *masks_for(clip, 5))
    after = syn.pitch_error(r.H, clip.G[5], 960, 540)
    assert before > 0.4
    assert r.accepted and after < 0.1 and r.ratio > r.ratio_before


def test_refine_rejects_without_enough_lines(clip):
    mask, valid = masks_for(clip, 5)
    r = pitch.LineRefiner(scale=0.5).refine(clip.H[5], np.zeros_like(mask), valid)
    assert not r.accepted and np.allclose(r.H, clip.H[5])
    # 한 방향 라인만 남기면(하프라인 근처 세로 줄) 호모그래피가 정해지지 않으므로 거부
    only = np.zeros_like(mask)
    only[:, 230:250] = mask[:, 230:250]
    r = pitch.LineRefiner(scale=0.5).refine(perturb(clip.H[5]), only, valid)
    assert not r.accepted


def test_track_homographies_removes_drift(clip):
    motion = motions(clip)
    store = pitch.LineMaskStore((0, 0))
    for t in range(len(clip.frames)):
        store.append(*masks_for(clip, t))
    kf = {0: perturb(clip.H[0], dx=3)}                      # 클릭 오차가 있는 키프레임
    chain = va.frame_homographies(motion, kf)
    refined, quality = pitch.track_homographies(motion, kf, store, pitch.LineRefiner(scale=0.5))
    err = lambda Hs: np.mean([syn.pitch_error(H, clip.G[t], 960, 540) for t, H in enumerate(Hs)])
    assert err(refined) < 0.1 and err(refined) < err(chain) / 3
    assert sum(q is not None for q in quality) > len(clip.frames) // 2
    # masks 없이 호출하면 기존 전파와 같다
    plain, q2 = pitch.track_homographies(motion, kf)
    assert all(np.allclose(a, b) for a, b in zip(plain, chain)) and set(q2) == {None}


def test_line_mask_store_roundtrip(tmp_path, clip):
    store = pitch.LineMaskStore((0, 0))
    m, v = masks_for(clip, 0)
    store.append(m, v)
    store.append(None)
    store.save(tmp_path / "x.lines.npz")
    back = pitch.LineMaskStore.load(tmp_path / "x.lines.npz")
    assert len(back) == 2 and back(1) is None and back(9) is None
    m2, v2 = back(0)
    assert np.array_equal(m2 > 0, m > 0) and np.array_equal(v2 > 0, v > 0)


# --- 공 타일 추론 ---------------------------------------------------------------
def test_tiles_cover_frame():
    ts = detector_mod.tiles(1920, 1080, 640, 0.2)
    cover = np.zeros((1080, 1920), bool)
    for x, y, s in ts:
        assert x + s <= 1920 and y + s <= 1080
        cover[y:y + s, x:x + s] = True
    assert cover.all()


class _CropModel:
    """타일(crop) 좌표로 공 하나를 돌려주는 가짜 YOLO — 프레임 (1000, 500) 위치의 흰 점을 찾음"""
    names = {0: 'ball'}

    def predict(self, source, imgsz, **kw):
        ys, xs = np.nonzero(source[..., 0] > 200)
        boxes = [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.7, 0]] if len(xs) else []
        arr = np.array(boxes, float).reshape(-1, 6)
        b = types.SimpleNamespace(xyxy=arr[:, :4], conf=arr[:, 4], cls=arr[:, 5])
        return [types.SimpleNamespace(boxes=b)]


def test_tiled_ball_detection_maps_back_and_dedups():
    d = detector_mod.RoboflowSoccerDetector.__new__(detector_mod.RoboflowSoccerDetector)
    d.confidence_threshold, d.iou_threshold, d.device = 0.3, 0.45, "cpu"
    d.imgsz, d.ball_imgsz, d.player_classes, d.ball_classes = 640, 960, None, None
    d.players_model, d.ball_model, d.field_model = None, _CropModel(), None
    d.class_names = d._get_class_names()
    d.ball_tile = 640
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[500:506, 1000:1006] = 255
    balls = d.detect(frame, detect_players=False)
    assert len(balls) == 1 and balls[0].bbox == (1000, 500, 1006, 506) and balls[0].is_ball


# --- 후처리 -------------------------------------------------------------------
def tr(dx):
    return np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1]], float).ravel().tolist()


def box(tid, x, y=100, kind=0, h=40):
    return [tid, kind, x - 8, y - h, x + 8, y, 0.9]


def test_motion_chain_transfer_and_segments():
    motion = [None, tr(5), tr(5), None, tr(5)]
    c = pp.MotionChain(motion)
    assert np.allclose(c.transfer([[10, 0]], 0, 2), [[20, 0]])
    assert c.same_segment(0, 2) and not c.same_segment(2, 3)


def test_stitch_tracks_links_fragment_with_camera_motion():
    # 카메라가 프레임마다 +3 px 팬 → 화면상 선수는 +3 px/frame 처럼 보이지만 실제로는 정지
    n = 40
    motion = [None] + [tr(3)] * (n - 1)
    frames = [[] for _ in range(n)]
    for t in range(0, 15):
        frames[t].append(box(1, 100 + 3 * t))
    for t in range(30, n):                                # 15 프레임(0.5 s) 가려진 뒤 다른 ID 로 재등장
        frames[t].append(box(7, 100 + 3 * t))
    for t in range(0, n):                                 # 멀리 있는 다른 선수 (연결되면 안 됨)
        if t < 14 or t > 31:
            frames[t].append(box(3 if t < 14 else 9, 600 + 3 * t))
    colors = {1: [np.array([200, 128, 128])] * 3, 7: [np.array([201, 128, 128])] * 3,
              3: [np.array([60, 150, 110])] * 3, 9: [np.array([60, 150, 110])] * 3}
    merged = pp.stitch_tracks(frames, motion, 30.0, colors)
    assert merged == {7: 1, 9: 3}
    assert {o[0] for f in frames for o in f} == {1, 3}
    assert len(colors[1]) == 6 and 7 not in colors
    added = pp.fill_gaps(frames, motion, 30)
    assert added == 15 + 18                               # 프레임 15~29, 14~31
    mid = next(o for o in frames[22] if o[0] == 1)
    assert abs((mid[2] + mid[4]) / 2 - (100 + 3 * 22)) <= 1 and mid[6] == 0.0   # 카메라 움직임을 따라 보간


def test_stitch_respects_appearance_and_distance():
    motion = [None] + [tr(0)] * 29
    frames = [[] for _ in range(30)]
    for t in range(10):
        frames[t].append(box(1, 100))
    for t in range(15, 30):
        frames[t].append(box(2, 105))                      # 가깝지만 유니폼 색이 다름
    colors = {1: [np.array([240, 128, 128])] * 3, 2: [np.array([40, 160, 100])] * 3}
    assert pp.stitch_tracks(frames, motion, 30.0, colors) == {}
    frames2 = [[box(1, 100)] if t < 10 else [box(2, 700)] for t in range(30)]
    assert pp.stitch_tracks(frames2, motion, 30.0) == {}   # 너무 멀리서 시작


def test_clean_ball_removes_spike():
    motion = [None] + [tr(0)] * 9
    frames = [[box(0, 100 + 5 * t, kind=1, h=6)] for t in range(10)]
    frames[5][0] = box(0, 600, kind=1, h=6)
    assert pp.clean_ball(frames, motion, 30.0) == 1 and frames[5] == []


def test_rts_smooth_reduces_noise_and_estimates_speed():
    rng = np.random.default_rng(0)
    ts = np.arange(90)
    truth = np.column_stack([ts * (5.0 / 30), np.zeros(90)])    # 5 m/s 직선
    z = truth + rng.normal(0, 0.4, truth.shape)
    pos, vel = pp.rts_smooth(ts, z, 1 / 30, 5.0, 0.4)
    assert np.mean(np.linalg.norm(pos - truth, axis=1)) < 0.5 * np.mean(np.linalg.norm(z - truth, axis=1))
    assert abs(np.median(np.hypot(vel[:, 0], vel[:, 1])) - 5.0) < 0.5


def test_smooth_world_adds_speed_and_keeps_none():
    world = [None] + [[[3, t * 0.2, 1.0], [0, 0.0, 0.0]] for t in range(1, 40)]
    out = pp.smooth_world(world, 30.0)
    assert out[0] is None and len(out[10]) == 2
    e = next(e for e in out[20] if e[0] == 3)
    assert len(e) == 4 and abs(e[3] - 6.0) < 0.5


def test_assign_goalkeepers():
    world = []
    for t in range(60):
        world.append([[1, -20, 0], [2, -15, 5], [3, 20, 0], [4, 25, -5], [9, -49, 1], [10, 0, 30]])
    teams = {'1': 'home', '2': 'home', '3': 'away', '4': 'away', '9': 'other', '10': 'other'}
    roles = pp.assign_goalkeepers(world, teams, 30.0)
    assert roles == {'9': 'goalkeeper'} and teams['9'] == 'home' and teams['10'] == 'other'


# --- 경기 지표 ----------------------------------------------------------------
def test_stats_distance_sprint_possession_and_passes():
    fps = 30.0
    world = []
    for t in range(90):
        x1 = -10 + 8.0 * t / fps                               # 선수 1: 8 m/s 로 3 초 → 스프린트, 24 m
        ball_owner = (x1, 0.0) if t < 30 else ((5.0, 5.0) if t < 60 else (20.0, -5.0))
        world.append([[1, x1, 0.0, 8.0], [2, 5.0, 5.0, 0.0], [3, 20.0, -5.0, 0.0],
                      [0, ball_owner[0] + 0.3, ball_owner[1], 0.0]])
    result = {'fps': fps, 'world': world, 'teams': {'1': 'home', '2': 'home', '3': 'away'}, 'roles': {}}
    s = match_stats.compute(result)
    p1 = s['players']['1']
    assert abs(p1['distance'] - 23.7) < 0.5 and p1['max_speed'] == 8.0 and len(p1['sprints']) == 1
    kinds = [e['type'] for e in s['events'] if e['type'] != 'sprint']
    assert kinds == ['pass', 'turnover']
    assert s['possession']['share']['home'] == pytest.approx(60 / 90, abs=0.05)
    assert s['teams']['home']['passes'] == 1 and s['teams']['home']['turnovers'] == 1
    assert s['shape'] and 'home' in s['shape'][0] and len(s['heatmaps']['home']) == 14


# --- 설정 프로필 ----------------------------------------------------------------
def test_analysis_profiles_from_config():
    from pipeline.main_pipeline import Soccer3DPipeline
    p = Soccer3DPipeline(config_path=str(Path(__file__).resolve().parent.parent / "configs" / "model_config.yaml"),
                         detector=types.SimpleNamespace())
    precise, realtime = p.analysis_profile('precise'), p.analysis_profile('realtime')
    assert precise['stride'] == 1 and precise['ball_tile'] and precise['line_refine'] and precise['postprocess']
    assert realtime['stride'] == 3 and not realtime['ball_tile'] and not realtime['line_refine']
    assert p.analysis_profile()['mode'] == p.config['analysis']['mode']
    with pytest.raises(ValueError):
        p.analysis_profile('turbo')


# --- 합성 영상 전체 흐름 + API ------------------------------------------------------
class TruthDetector:
    """합성 영상 정답 박스를 그대로 돌려주는 탐지기 (호출 순서 = 프레임 순서, stride 1)"""

    def __init__(self, clip):
        self.clip, self.t = clip, 0

    def detect(self, frame):
        t = min(self.t, len(self.clip.boxes) - 1)
        self.t += 1
        out = [Detection(tuple(int(v) for v in b), 0.9, 0, 'player_person', 'player') for _, _, b in self.clip.boxes[t]]
        bx, by = self.clip.ball[t]
        q = self.clip.G[t] @ [bx, by, 1]
        u, v = q[0] / q[2], q[1] / q[2]
        out.append(Detection((int(u - 3), int(v - 3), int(u + 3), int(v + 3)), 0.6, BALL_OFFSET, 'ball_ball', 'ball'))
        return out


@pytest.fixture
def precise_client(tmp_path, monkeypatch, clip):
    from fastapi.testclient import TestClient
    from backend import server
    from tracking.tracker import SoccerTracker
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(server, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(server, "analysis_jobs", {})
    profiles = {'precise': {'mode': 'precise', 'stride': 1, 'postprocess': True, 'line_refine': True},
                'realtime': {'mode': 'realtime', 'stride': 2, 'postprocess': False, 'line_refine': False}}
    fake = types.SimpleNamespace(detector=TruthDetector(clip), config={},
                                 analysis_profile=lambda mode=None: dict(profiles[mode or 'precise']),
                                 make_tracker=lambda: SoccerTracker(tracker_type='simple'))
    monkeypatch.setattr(server, "_get_pipeline", lambda: fake)
    syn.write_video(clip, tmp_path / "broadcast.avi")
    with TestClient(server.app) as c:
        c.fake = fake
        yield c


def wait(client, name, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/analysis/{name}/status").json()
        if s["state"] in ("done", "error", "cancelled"):
            return s
        time.sleep(0.1)
    raise AssertionError("분석이 끝나지 않음")


def test_precise_api_flow_with_line_refined_calibration(precise_client, clip):
    c = precise_client
    assert c.post("/analysis/broadcast.avi?mode=turbo").status_code == 400
    assert c.post("/analysis/broadcast.avi?mode=precise").status_code == 200
    s = wait(c, "broadcast.avi")
    assert s["state"] == "done" and s["mode"] == "precise" and s["line_refine"] is True
    assert c.get("/analysis/broadcast.avi/stats").status_code == 404      # 보정 전

    pts = []
    for x, y in [(0, 34), (0, 9.15), (0, -9.15), (-36, 20.16), (36, 20.16), (-36, -20.16), (36, -20.16)]:
        q = clip.G[0] @ [x, y, 1]
        u, v = q[0] / q[2], q[1] / q[2]
        if 0 <= u < 960 and 0 <= v < 540:
            pts.append({"image": [u + 1.0, v - 1.0], "pitch": [x, y]})
    r = c.post("/analysis/broadcast.avi/calibration", json={"keyframes": [{"frame": 0, "points": pts}]}).json()
    assert r["refined_frames"] > len(clip.frames) // 2 and r["has_stats"] is True

    res = c.get("/analysis/broadcast.avi").json()
    assert res["mode"] == "precise" and res["version"] == 2 and res["postprocess"] is not None
    errs = [syn.pitch_error(np.array(h).reshape(3, 3), clip.G[t], 960, 540) for t, h in enumerate(res["homographies"])]
    assert np.mean(errs) < 0.15
    assert all(len(e) == 4 for w in res["world"] if w for e in w)          # [id, x, y, speed]
    stats = c.get("/analysis/broadcast.avi/stats").json()
    assert stats["players"] and "possession" in stats and stats["covered_frames"] == len(clip.frames)

    # refine=false 이면 라인 정렬 없이 전파
    r2 = c.post("/analysis/broadcast.avi/calibration",
                json={"keyframes": [{"frame": 0, "points": pts}], "refine": False}).json()
    assert r2["refined_frames"] is None


def test_cancel_analysis(precise_client):
    c = precise_client
    assert c.delete("/analysis/broadcast.avi").status_code == 409          # 진행 중 아님
    slow = c.fake.detector.detect
    c.fake.detector.detect = lambda f: (time.sleep(0.05), slow(f))[1]
    c.post("/analysis/broadcast.avi?mode=realtime")
    time.sleep(0.2)
    assert c.delete("/analysis/broadcast.avi").json()["state"] in ("cancelling", "cancelled")
    assert wait(c, "broadcast.avi")["state"] == "cancelled"
    assert c.get("/analysis/broadcast.avi").status_code == 404
