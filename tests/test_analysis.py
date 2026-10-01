"""영상 분석·경기장 보정 테스트 (합성 영상 + 가짜 탐지기, YOLO 불필요)

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

from pipeline import video_analysis as va
from tracking.detector import Detection, BALL_OFFSET
from tracking.tracker import SoccerTracker

RNG = np.random.default_rng(0)
TEXTURE = cv2.GaussianBlur(RNG.integers(0, 255, (400, 560), dtype=np.uint8), (0, 0), 1.2)


def textured(dx=0, dy=0):
    """카메라가 (dx, dy) 만큼 팬한 것처럼 보이는 배경 (원본 해상도 320x480)"""
    img = TEXTURE[40 + dy:40 + dy + 320, 40 + dx:40 + dx + 480]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# --- 카메라 움직임 ----------------------------------------------------------
def test_camera_motion_recovers_pan():
    a, b = textured(0, 0), textured(8, 4)  # 배경이 왼쪽·위로 (8, 4) px 이동
    g = lambda im: cv2.cvtColor(cv2.resize(im, None, fx=va.MOTION_SCALE, fy=va.MOTION_SCALE), cv2.COLOR_BGR2GRAY)
    M = va.camera_motion(g(a), g(b), prev_boxes=[(100, 100, 140, 180)])
    assert M is not None
    p = M @ np.array([200.0, 150.0, 1.0])
    assert np.allclose(p[:2] / p[2], (192, 146), atol=1.0)  # p_t = M · p_{t-1}


def test_camera_motion_fails_on_scene_cut():
    g = lambda im: cv2.cvtColor(cv2.resize(im, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    other = cv2.cvtColor(RNG.integers(0, 255, (320, 480), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    assert va.camera_motion(g(textured()), g(other), prev_boxes=[]) is None


# --- 보정 전파 ---------------------------------------------------------------
def translation(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)


def test_frame_homographies_propagate_both_ways_and_stop_at_cut():
    # 프레임마다 화면이 오른쪽으로 10px (p_t = p_{t-1} + 10), 프레임 4 에서 장면 전환
    motion = [None, translation(10, 0).ravel().tolist(), translation(10, 0).ravel().tolist(), None,
              translation(10, 0).ravel().tolist()]
    Hs = va.frame_homographies(motion, {1: np.eye(3)})  # 프레임 1: 픽셀 = 경기장 좌표
    apply = lambda H, u: (H @ [u, 0, 1])[0] / (H @ [u, 0, 1])[2]
    assert apply(Hs[1], 50) == pytest.approx(50)
    assert apply(Hs[2], 60) == pytest.approx(50)   # 같은 지점이 다음 프레임에선 +10px
    assert apply(Hs[0], 40) == pytest.approx(50)   # 이전 프레임에선 -10px
    assert Hs[3] is None and Hs[4] is None          # 움직임이 끊긴 구간 너머로는 전파 안 함


def test_nearest_keyframe_wins():
    motion = [None] + [translation(1, 0).ravel().tolist()] * 9
    H_far, H_near = np.eye(3), translation(100, 0)
    Hs = va.frame_homographies(motion, {0: H_far, 9: H_near})
    assert np.allclose(Hs[8], H_near @ translation(1, 0))  # 9 에 더 가까움


# --- 경기장 보정 -------------------------------------------------------------
def fake_result(frames, motion=None):
    return {'frames': frames, 'motion': motion or [None] + [np.eye(3).ravel().tolist()] * (len(frames) - 1),
            'frame_count': len(frames)}


RECT_POINTS = [  # 이미지 0~1000 x 0~680 → 경기장 전체 (y 는 화면 위쪽이 +)
    {'image': [0, 0], 'pitch': [-52.5, 34]}, {'image': [1000, 0], 'pitch': [52.5, 34]},
    {'image': [1000, 680], 'pitch': [52.5, -34]}, {'image': [0, 680], 'pitch': [-52.5, -34]},
]


def test_calibrate_maps_feet_and_filters_off_pitch():
    frames = [[[1, 0, 490, 300, 510, 340, 0.9],       # 발 = (500, 340) → 경기장 중앙
               [2, 0, 5000, 100, 5020, 140, 0.9]]] * 3  # 경기장 한참 밖 → 제외
    res = va.calibrate(fake_result(frames), [{'frame': 0, 'points': RECT_POINTS}])
    assert res['calibration']['keyframes'][0]['frame'] == 0
    for w in res['world']:
        assert len(w) == 1 and w[0][0] == 1
        assert w[0][1] == pytest.approx(0, abs=0.01) and w[0][2] == pytest.approx(0, abs=0.01)


def test_calibrate_rejects_bad_points():
    with pytest.raises(ValueError):
        va.calibrate(fake_result([[]]), [{'frame': 0, 'points': RECT_POINTS[:3]}])
    collinear = [{'image': [i * 10, i * 10], 'pitch': [i, i]} for i in range(4)]
    with pytest.raises(ValueError):
        va.calibrate(fake_result([[]]), [{'frame': 0, 'points': collinear}])


# --- 보조 함수 ---------------------------------------------------------------
def test_interpolate_objects_only_shared_tracks():
    a = [[1, 0, 0, 0, 10, 20, 0.9], [2, 0, 50, 50, 60, 70, 0.8]]
    b = [[1, 0, 10, 10, 20, 30, 0.7]]
    assert va.interpolate_objects(a, b, 0.5) == [[1, 0, 5, 5, 15, 25, 0.7]]


def test_plausible_ball_and_on_grass():
    assert va.plausible_ball((0, 0, 12, 12)) and not va.plausible_ball((0, 0, 14, 5))
    mask = np.zeros((100, 100), np.uint8)
    mask[50:, :] = 255  # 아래쪽 절반만 잔디
    assert va.on_grass(mask, (40, 20, 60, 70), scale=1.0)       # 발이 잔디 위
    assert not va.on_grass(mask, (40, 0, 60, 30), scale=1.0)    # 관중석
    assert not va.plausible_ball((0, 0, 0, 0))


def test_cluster_teams_two_kits_and_outlier():
    white, blaugrana, yellow = [245, 128, 128], [40, 150, 110], [230, 110, 200]  # Lab
    colors = {i: [np.array(white) + RNG.normal(0, 2, 3) for _ in range(5)] for i in range(1, 7)}
    colors.update({i: [np.array(blaugrana) + RNG.normal(0, 2, 3) for _ in range(5)] for i in range(7, 12)})
    colors[20] = [np.array(yellow, float)] * 5   # 심판
    colors[30] = [np.array(white, float)]        # 샘플 부족
    teams, kit = va.cluster_teams(colors)
    assert {teams[i] for i in range(1, 7)} == {'home'}      # 트랙이 많은 쪽이 home
    assert {teams[i] for i in range(7, 12)} == {'away'}
    assert teams[20] == 'other' and teams[30] == 'other'
    assert set(kit) == {'home', 'away'} and kit['home'].startswith('#')


# --- 영상 분석 (합성 영상) ---------------------------------------------------
class ScriptedDetector:
    """프레임 호출 순서대로 선수 1명(오른쪽으로 이동) + 공을 돌려주는 가짜 탐지기"""
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        x = 100 + self.calls * 12
        self.calls += 1
        return [Detection((x, 150, x + 20, 200), 0.9, 0, 'player_person', 'player'),
                Detection((300, 250, 310, 260), 0.8, BALL_OFFSET, 'ball_ball', 'ball')]


def write_clip(path, n=9):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (480, 320))
    for i in range(n):
        img = textured(i * 2, 0)
        img[200:, :] = (40, 140, 40)  # 아래쪽 잔디 (선수 발 위치)
        w.write(img)
    w.release()


def test_analyze_video_with_stride_interpolates(tmp_path):
    clip = tmp_path / "clip.avi"
    write_clip(clip)
    det = ScriptedDetector()
    seen = []
    res = va.analyze_video(clip, det, SoccerTracker(tracker_type='simple'), stride=4,
                           progress=lambda d, t: seen.append((d, t)))
    assert det.calls == 3                              # 프레임 0, 4, 8 에서만 탐지
    assert res['frame_count'] == 9 and len(res['motion']) == 9 and res['stride'] == 4
    assert seen[-1] == (9, 9)
    player = lambda f: next(o for o in res['frames'][f] if o[1] == va.KIND_PLAYER)
    assert player(0)[2] == 100 and player(4)[2] == 112
    assert player(2)[2] == 106                         # 0 과 4 사이 보간
    assert all(any(o[1] == va.KIND_BALL and o[0] == 0 for o in f) for f in res['frames'])  # 공은 항상 ID 0


# --- API ----------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import server
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(server, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(server, "analysis_jobs", {})
    fake = types.SimpleNamespace(detector=ScriptedDetector(), config={'analysis': {'stride': 2}},
                                 make_tracker=lambda: SoccerTracker(tracker_type='simple'))
    monkeypatch.setattr(server, "_get_pipeline", lambda: fake)
    write_clip(tmp_path / "match.avi")
    with TestClient(server.app) as c:
        yield c


def wait_done(client, name, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        status = client.get(f"/analysis/{name}/status").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.1)
    raise AssertionError("분석이 끝나지 않음")


def test_analysis_api_flow(client):
    assert client.get("/analysis/match.avi/status").json()["state"] == "none"
    assert client.get("/analysis/match.avi").status_code == 404
    assert client.post("/analysis/match.avi/calibration", json={"keyframes": []}).status_code == 404

    assert client.post("/analysis/match.avi").json()["state"] in ("queued", "running", "done")
    status = wait_done(client, "match.avi")
    assert status["state"] == "done" and status["calibrated"] is False

    result = client.get("/analysis/match.avi").json()
    assert result["frame_count"] == 9 and len(result["frames"]) == 9 and result["world"] is None

    body = {"keyframes": [{"frame": 0, "points": RECT_POINTS}]}
    r = client.post("/analysis/match.avi/calibration", json=body)
    assert r.status_code == 200 and r.json()["calibrated_frames"] >= 1
    assert client.get("/analysis/match.avi/status").json()["calibrated"] is True
    assert client.get("/analysis/match.avi").json()["world"][0] is not None

    bad = {"keyframes": [{"frame": 0, "points": RECT_POINTS[:3]}]}
    assert client.post("/analysis/match.avi/calibration", json=bad).status_code == 400


def test_analysis_rejects_unknown_and_traversal(client):
    assert client.post("/analysis/nope.mp4").status_code == 404
    assert client.get("/analysis/..%2F..%2Fsecret/status").status_code == 404


def test_calibrate_skips_boxes_clipped_by_frame_border():
    frames = [[[1, 0, 490, 300, 510, 340, 0.9],      # 화면 안
               [2, 0, 0, 300, 15, 340, 0.9],         # 왼쪽 가장자리에 잘림
               [3, 0, 600, 640, 620, 679, 0.9],      # 아래 가장자리에 잘림 (발이 화면 밖)
               [0, 1, 0, 100, 6, 106, 0.5]]]         # 공은 가장자리여도 유지
    res = fake_result(frames)
    res.update(width=1000, height=680)
    world = va.calibrate(res, [{'frame': 0, 'points': RECT_POINTS}])['world'][0]
    assert sorted(e[0] for e in world) == [0, 1]
