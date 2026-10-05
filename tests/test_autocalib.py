"""자동 초기 보정 테스트 — 기준점 없이 경기장 라인·센터서클로 호모그래피 찾기

합성 중계 영상(pipeline/synthetic.py)의 정답 호모그래피와 비교한다.
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

from pipeline import autocalib as ac, pitch, synthetic as syn, video_analysis as va


@pytest.fixture(scope="module")
def clip():
    return syn.render_clip(n=60, seed=2)


def masks_at(clip, t, s=0.5):
    small = cv2.resize(clip.frames[t], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return pitch.line_mask(small, [b for _, _, b in clip.boxes[t]], box_scale=s)


def err(H, clip, t):
    return syn.pitch_error(H, clip.G[t], 960, 540)


# --- 직선 검출 -----------------------------------------------------------------
def test_detect_lines_keeps_straight_lines_and_rejects_arcs():
    mask = np.zeros((270, 480), np.uint8)
    cv2.line(mask, (20, 60), (460, 80), 255, 1)                 # 긴 직선
    cv2.line(mask, (100, 260), (200, 100), 255, 1)              # 기울어진 직선
    cv2.ellipse(mask, (300, 180), (140, 35), 0, 0, 360, 255, 1)  # 납작한 타원(멀리 있는 센터서클)
    lines = ac.detect_lines(mask)
    assert len(lines) == 2
    angles = sorted(round(L.angle) for L in lines)
    assert angles[0] == pytest.approx(-58, abs=2) and angles[1] == pytest.approx(3, abs=1)


# --- 단일 프레임 ----------------------------------------------------------------
@pytest.mark.parametrize("t", [5, 20, 30, 45])
def test_auto_calibrate_frame_matches_ground_truth(clip, t):
    r = ac.AutoCalibrator().calibrate(*masks_at(clip, t))
    assert r.H is not None, r.reason
    assert err(r.H, clip, t) < 0.3
    assert r.f1 >= 0.6 and r.margin <= 0.85


def test_center_circle_view_uses_conic_solver():
    # 센터서클 정면: 하프라인 + 센터서클 + 먼 터치라인만 보임 (2×2 직선 가설이 불가능한 장면)
    cam = syn.BroadcastCamera()
    P, G = cam.matrices((0.0, 3.0), 960 * 2.2)
    T = np.array([[1 / syn.PPM, 0, syn.X0], [0, -1 / syn.PPM, syn.Y1], [0, 0, 1]])
    img = cv2.warpPerspective(syn._topdown_texture(0), G @ T, (960, 540))
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mask, valid = pitch.line_mask(small)
    lines = ac.detect_lines(mask)
    assert len(lines) <= 3
    hyps = ac.circle_hypotheses(mask, lines, 480, 270)
    assert hyps and min(syn.pitch_error(np.linalg.inv(np.linalg.inv(pitch.scale_matrix(0.5)) @ h), G, 960, 540)
                        for h in hyps) < 1.0
    r = ac.AutoCalibrator().calibrate(mask, valid)
    assert r.H is not None and syn.pitch_error(r.H, G, 960, 540) < 0.3


def test_refuses_noise_and_single_line():
    cal = ac.AutoCalibrator()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)
    assert cal.calibrate(*pitch.line_mask(noise)).H is None
    mask = np.zeros((270, 480), np.uint8)
    cv2.line(mask, (0, 100), (479, 120), 255, 1)               # 터치라인 하나 — 위치를 정할 수 없음
    r = cal.calibrate(mask, np.full_like(mask, 255))
    assert r.H is None and r.reason


# --- 영상 전체 -------------------------------------------------------------------
def motions(clip, s=0.5):
    out, prev = [None], None
    for i, f in enumerate(clip.frames):
        g = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            M = va.camera_motion(prev, g, [b for _, _, b in clip.boxes[i - 1]], s)
            out.append(None if M is None else M.ravel().tolist())
        prev = g
    return out


@pytest.fixture(scope="module")
def video_data(clip):
    store = pitch.LineMaskStore((0, 0))
    for t in range(len(clip.frames)):
        store.append(*masks_at(clip, t))
    return motions(clip), store


def test_auto_keyframes_cover_clip_without_manual_input(clip, video_data):
    motion, store = video_data
    kfs = ac.auto_keyframes(motion, store, (960, 540), fps=30, every_s=1.0)
    assert len(kfs) >= 2 and all(k['source'] == 'auto' and len(k['homography']) == 9 for k in kfs)
    Hs, _ = pitch.track_homographies(motion, {k['frame']: np.array(k['homography']).reshape(3, 3) for k in kfs},
                                     store, pitch.LineRefiner(scale=0.5))
    errors = [err(H, clip, t) for t, H in enumerate(Hs)]
    assert all(H is not None for H in Hs) and np.mean(errors) < 0.2


def test_inconsistent_auto_keyframe_is_dropped(clip, video_data):
    motion, store = video_data

    class Fake:
        """프레임 30 에서만 10 m 어긋난(높은 점수의) 잘못된 해석을 내는 보정기"""
        def calibrate(self, mask, valid):
            t = Fake.t
            Fake.t += 30
            H = clip.H[t]
            score = 0.8
            if t == 30:
                H = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]]) @ H
                score = 0.7
            return ac.AutoResult(H, score, 0.9, 0.8, 0.5, 6, 100)
    Fake.t = 0
    kfs = ac.auto_keyframes(motion, store, (960, 540), fps=30, every_s=1.0, calibrator=Fake())
    assert [k['frame'] for k in kfs] == [0]                    # 30 은 0 과 맞지 않고 점수가 낮아 제거


def test_auto_calibrate_keeps_manual_keyframes(clip, video_data):
    motion, store = video_data
    frames = [[] for _ in clip.frames]
    result = {'fps': 30.0, 'width': 960, 'height': 540, 'frames': frames, 'motion': motion,
              'frame_count': len(frames), 'mode': 'precise', 'teams': {}}
    pts = []
    for x, y in [(0, 34), (0, 9.15), (0, -9.15), (-36, 20.16), (36, 20.16), (36, -20.16)]:
        q = clip.G[10] @ [x, y, 1]
        u, v = q[0] / q[2], q[1] / q[2]
        if 0 <= u < 960 and 0 <= v < 540:
            pts.append({'image': [float(u), float(v)], 'pitch': [x, y]})
    result['calibration'] = {'keyframes': [{'frame': 10, 'points': pts}]}
    info = va.auto_calibrate(result, store, every_s=0.5)          # 0, 15, 30, 45 프레임 시도
    kfs = result['calibration']['keyframes']
    assert info['manual'] == 1 and info['auto'] >= 1
    assert any(k['frame'] == 10 and k.get('source') != 'auto' for k in kfs)
    assert all(abs(k['frame'] - 10) > 30 for k in kfs if k.get('source') == 'auto')   # 수동 ±1 초는 자동 제외
    assert info['calibrated_frames'] == len(frames) and result['stats'] is not None

    empty = pitch.LineMaskStore((0, 0))
    for _ in frames:
        empty.append(np.zeros((270, 480), np.uint8), np.zeros((270, 480), np.uint8))
    with pytest.raises(ValueError):
        va.auto_calibrate({**result, 'calibration': None}, empty)


def test_keyframe_homography_accepts_direct_matrix():
    H = va.keyframe_homography([], np.eye(3).ravel().tolist())
    assert np.allclose(H, np.eye(3))
    with pytest.raises(ValueError):
        va.keyframe_homography([], [0.0] * 9)


# --- API ----------------------------------------------------------------------
class TruthDetector:
    def __init__(self, clip):
        self.clip, self.t = clip, 0

    def detect(self, frame):
        from tracking.detector import Detection
        t = min(self.t, len(self.clip.boxes) - 1)
        self.t += 1
        return [Detection(tuple(int(v) for v in b), 0.9, 0, 'player_person', 'player') for _, _, b in self.clip.boxes[t]]


@pytest.fixture
def client(tmp_path, monkeypatch, clip):
    from fastapi.testclient import TestClient
    from backend import server
    from tracking.tracker import SoccerTracker
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(server, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(server, "analysis_jobs", {})
    profiles = {'precise': {'mode': 'precise', 'stride': 1, 'postprocess': True, 'line_refine': True,
                            'auto_calibrate': True},
                'realtime': {'mode': 'realtime', 'stride': 2, 'postprocess': False, 'line_refine': False}}
    fake = types.SimpleNamespace(detector=TruthDetector(clip), config={},
                                 analysis_profile=lambda mode=None: dict(profiles[mode or 'precise']),
                                 make_tracker=lambda: SoccerTracker(tracker_type='simple'))
    monkeypatch.setattr(server, "_get_pipeline", lambda: fake)
    syn.write_video(clip, tmp_path / "match.avi")
    with TestClient(server.app) as c:
        c.fake = fake
        yield c


def wait(client, name, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/analysis/{name}/status").json()
        if s["state"] in ("done", "error", "cancelled"):
            return s
        time.sleep(0.2)
    raise AssertionError("작업이 끝나지 않음")


def test_precise_analysis_calibrates_automatically(client, clip):
    client.post("/analysis/match.avi?mode=precise")
    s = wait(client, "match.avi")
    assert s["state"] == "done" and s["calibrated"] is True and s["auto_calibration"]["status"] == "ok"
    res = client.get("/analysis/match.avi").json()
    assert all(k["source"] == "auto" for k in res["calibration"]["keyframes"])
    errors = [err(np.array(h).reshape(3, 3), clip, t) for t, h in enumerate(res["homographies"]) if h]
    assert len(errors) == len(clip.frames) and np.mean(errors) < 0.25
    assert client.get("/analysis/match.avi/stats").status_code == 200


def test_auto_calibration_endpoint_on_realtime_result(client):
    assert client.post("/analysis/match.avi/calibration/auto").status_code == 404     # 분석 전
    client.post("/analysis/match.avi?mode=realtime")
    s = wait(client, "match.avi")
    assert s["state"] == "done" and s["calibrated"] is False and s["line_refine"] is False
    r = client.post("/analysis/match.avi/calibration/auto")
    assert r.status_code == 200
    s = wait(client, "match.avi")
    assert s["calibrated"] is True and s["auto_calibration"]["status"] == "ok"
    assert s["line_refine"] is True                       # 라인 마스크를 만들어 저장
    # 수동 저장 시 자동 키프레임(homography)을 그대로 함께 보내도 받아들임
    kfs = client.get("/analysis/match.avi").json()["calibration"]["keyframes"]
    r = client.post("/analysis/match.avi/calibration", json={"keyframes": kfs})
    assert r.status_code == 200 and r.json()["calibrated_frames"] > 0
    bad = [{**kfs[0], "homography": [1.0, 2.0]}]
    assert client.post("/analysis/match.avi/calibration", json={"keyframes": bad}).status_code == 400
