"""핵심 모듈 단위 테스트 (torch/ultralytics 없이 실행 가능)

실행: python -m pytest tests -q
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.detector import Detection, BALL_OFFSET
from tracking.homography import HomographyTransformer
from tracking.tracker import BallTracker, SimpleTracker, SoccerTracker, make_tracker
from tracking import detector as detector_mod


def det(bbox, conf=0.9, class_id=0, class_name="player_player", type_="player"):
    return Detection(bbox=tuple(bbox), confidence=conf, class_id=class_id, class_name=class_name, type=type_)


def ball(bbox, conf=0.8):
    return det(bbox, conf, BALL_OFFSET, "ball_ball", "ball")


# --- homography ---------------------------------------------------------
def test_homography_corners_map_to_field():
    h = HomographyTransformer()
    assert h.compute_from_field_lines([(100, 100), (1180, 100), (1180, 620), (100, 620)])
    assert np.allclose(h.pixel_to_world(100, 100), (-52.5, -34.0), atol=1e-3)
    assert np.allclose(h.pixel_to_world(640, 360), (0.0, 0.0), atol=1e-3)
    px, py = h.world_to_pixel(52.5, 34.0)
    assert np.allclose((px, py), (1180, 620), atol=1e-2)


def test_homography_batch_matches_single():
    h = HomographyTransformer()
    h.compute_from_field_lines([(100, 100), (1180, 100), (1180, 620), (100, 620)])
    pts = [(100, 100), (640, 360), (900, 500)]
    batch = h.pixels_to_world(pts)
    for (px, py), w in zip(pts, batch):
        assert np.allclose(h.pixel_to_world(px, py), w, atol=1e-4)
    assert h.pixels_to_world([]).shape == (0, 2)


# --- tracker ------------------------------------------------------------
def test_simple_tracker_keeps_id():
    t = SimpleTracker(match_threshold=0.3)
    a = t.update(np.array([[100, 100, 150, 200, 0.9, 1]]))
    b = t.update(np.array([[102, 101, 152, 201, 0.9, 1]]))
    assert a[0][4] == b[0][4] and b[0][6] == 1  # 7번째 열 = class_id


def test_make_tracker():
    assert isinstance(make_tracker('simple'), SimpleTracker)
    with pytest.raises(ValueError):
        make_tracker('nope')
    # boxmot 미설치 시에도 예외 없이 SimpleTracker 로 대체
    assert hasattr(make_tracker('bytetrack'), 'update')


def test_detection_label_and_is_ball():
    # 선수 모델의 0번 클래스는 공이 아니다 (공 판정은 class_id 100~199)
    p0 = det((0, 0, 10, 10), class_id=0, class_name="player_person")
    assert not p0.is_ball and p0.label == "person"
    gk = det((0, 0, 10, 10), class_id=1, class_name="player_goalkeeper")
    assert gk.label == "goalkeeper"
    b = ball((0, 0, 10, 10))
    assert b.is_ball and b.label == "ball"
    assert not det((0, 0, 1, 1), class_id=200, class_name="field_x", type_="field").is_ball
    assert p0.center == (5, 5) and p0.area == 100


def test_soccer_tracker_ids_and_velocity():
    t = SoccerTracker(tracker_type='simple')
    objs = t.update([det((100, 100, 150, 200)), ball((300, 300, 320, 320))])
    assert len(objs) == 2
    objs2 = t.update([det((104, 100, 154, 200)), ball((310, 300, 330, 320))])
    assert sorted(o.track_id for o in objs) == sorted(o.track_id for o in objs2)
    player = next(o for o in objs2 if o.label == "player")
    assert player.velocity == (4.0, 0.0)  # 두 번째 프레임부터 속도 계산


def test_ball_always_single_fixed_id():
    t = SoccerTracker(tracker_type='simple')
    # 공 후보가 여러 개(오탐 포함)여도 공은 항상 1개, 같은 ID
    frames = [
        [ball((300, 300, 320, 320))],
        [ball((330, 300, 350, 320)), ball((900, 600, 910, 610), conf=0.95)],  # 먼 곳 오탐(신뢰도는 더 높음)
        [],                                                                    # 가려짐
        [ball((390, 300, 410, 320))],                                          # 등속 예측으로 재포착
    ]
    ball_objs = []
    for f in frames:
        balls = [o for o in t.update(f) if o.label == "ball"]
        assert len(balls) <= 1
        ball_objs += [(o.track_id, o.center) for o in balls]  # TrackedObject 는 제자리 갱신되므로 값 복사
    assert [tid for tid, _ in ball_objs] == [SoccerTracker.BALL_TRACK_ID] * 3
    assert [c for _, c in ball_objs] == [(310.0, 310.0), (340.0, 310.0), (400.0, 310.0)]  # 오탐 무시, 예측으로 재포착


def test_ball_tracker_reacquires_after_long_miss():
    bt = BallTracker(max_jump=50, max_missing=2)
    assert bt.update([ball((0, 0, 10, 10))]) is not None
    far = ball((800, 800, 810, 810))
    assert bt.update([far]) is None      # 너무 멀어서 거부
    assert bt.update([far]) is None
    assert bt.update([far]) is far       # max_missing 초과 → 재획득


def test_soccer_tracker_visualize_runs():
    t = SoccerTracker(tracker_type='simple')
    objs = t.update([det((100, 100, 150, 200))])
    out = t.visualize(np.zeros((300, 300, 3), np.uint8), objs)
    assert out.shape == (300, 300, 3) and out.any()


def test_soccer_tracker_empty_frame_and_reset():
    t = SoccerTracker(tracker_type='simple')
    assert t.update([]) == []
    t.update([ball((0, 0, 10, 10))])
    t.reset()
    assert t.tracked_objects == {} and t.ball_tracker.last is None


def test_soccer_tracker_backend_injection():
    calls = []

    class FakeBackend:
        def update(self, det_array, frame):
            calls.append(det_array.copy())
            return np.array([[*det_array[0, :4], 42, det_array[0, 4], det_array[0, 5]]])

    t = SoccerTracker(backend_factory=FakeBackend)
    objs = t.update([det((0, 0, 10, 20), class_id=1, class_name="player_goalkeeper"), ball((50, 50, 55, 55))])
    assert calls[0].shape == (1, 6)  # 공은 다중 객체 추적기로 가지 않음
    assert {(o.track_id, o.label) for o in objs} == {(43, "goalkeeper"), (SoccerTracker.BALL_TRACK_ID, "ball")}


# --- detector (가짜 YOLO 모델) ------------------------------------------
class _FakeBoxes:
    def __init__(self, rows):
        self.xyxy = np.array([r[:4] for r in rows], dtype=float)
        self.conf = np.array([r[4] for r in rows], dtype=float)
        self.cls = np.array([r[5] for r in rows], dtype=float)

    def __len__(self):
        return len(self.conf)


class _FakeModel:
    names = {0: "player"}

    def __init__(self, rows):
        self.rows = rows

    def predict(self, **_):
        return [types.SimpleNamespace(boxes=_FakeBoxes(self.rows) if self.rows else None)]


def test_coco_class_filter_only_for_generic_models():
    coco = types.SimpleNamespace(names={0: "person", 32: "sports ball", 56: "chair"})
    football = types.SimpleNamespace(names={0: "ball", 1: "goalkeeper", 2: "player"})
    assert detector_mod._coco_class_ids(coco, "sports ball") == [32]
    assert detector_mod._coco_class_ids(coco, "person") == [0]
    assert detector_mod._coco_class_ids(football, "person") is None  # 전용 모델은 필터 없음


def test_detector_detect_with_offsets():
    d = detector_mod.RoboflowSoccerDetector.__new__(detector_mod.RoboflowSoccerDetector)
    d.confidence_threshold, d.iou_threshold, d.device = 0.5, 0.45, "cpu"
    d.imgsz, d.ball_imgsz, d.player_classes, d.ball_classes = 640, 1280, None, None
    d.players_model = _FakeModel([[10, 20, 30, 60, 0.9, 0]])
    d.ball_model = _FakeModel([[5, 5, 9, 9, 0.8, 0]])
    d.field_model = None
    d.class_names = d._get_class_names()
    out = d.detect(np.zeros((10, 10, 3), np.uint8))
    assert [o.type for o in out] == ["player", "ball"]
    assert out[0] == Detection(bbox=(10, 20, 30, 60), confidence=0.9, class_id=0,
                               class_name="player_player", type="player")
    assert not out[0].is_ball and out[0].label == "player"
    assert out[1].class_id == 100 and out[1].is_ball and out[1].label == "ball"
    assert d.draw_detections(np.zeros((80, 80, 3), np.uint8), out).any()


# --- pipeline (detector → tracker → homography 통합) ---------------------
def test_pipeline_process_frame_with_injection():
    from pipeline.main_pipeline import Soccer3DPipeline  # torch 없이 import 가능해야 함

    frame_dets = [det((630, 350, 650, 370)), ball((600, 300, 610, 310))]
    p = Soccer3DPipeline(
        config_path="does_not_exist.yaml",
        detector=types.SimpleNamespace(detect=lambda f: frame_dets),
        tracker=SoccerTracker(tracker_type='simple'),
    )
    p.homography.compute_from_field_lines([(100, 100), (1180, 100), (1180, 620), (100, 620)])

    fd = p.process_frame(np.zeros((720, 1280, 3), np.uint8))
    assert fd.detections is frame_dets
    by_label = {w["label"]: w for w in fd.world_coordinates}
    assert set(by_label) == {"player", "ball"}
    assert abs(by_label["player"]["world_x"]) < 0.1 and abs(by_label["player"]["world_y"]) < 0.1
    assert by_label["ball"]["track_id"] == SoccerTracker.BALL_TRACK_ID


# --- backend ------------------------------------------------------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.server import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_tracking_data_deterministic(client):
    a = client.get("/get_tracking_data/5").json()
    b = client.get("/get_tracking_data/5").json()
    assert a["players"] == b["players"] and len(a["players"]) == 11


def test_websocket_streams_without_client_messages(client):
    with client.websocket_connect("/ws/stream") as ws:
        frames = [ws.receive_json()["frame"] for _ in range(3)]
    assert frames == [0, 1, 2]


def test_websocket_ignores_bad_json(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text("not json")
        ws.send_text('{"type": "start", "frame": 100}')
        seen = [ws.receive_json()["frame"] for _ in range(5)]
    assert 100 in seen


def test_websocket_seek_while_paused_sends_that_frame_once(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text('{"type": "seek"}')  # frame 누락 → 무시되고 연결 유지
        ws.send_text('{"type": "seek", "frame": 450, "paused": true}')
        frames = []
        while not frames or frames[-1] != 450:
            frames.append(ws.receive_json()["frame"])
        time.sleep(0.2)  # 정지 중: 450 이후 추가 전송이 없어야 함
        ws.send_text('{"type": "seek", "frame": 900, "paused": false}')
        after = [ws.receive_json()["frame"] for _ in range(3)]
    # 정지 중 450 이 반복 전송되거나 프레임이 증가했다면 900 보다 먼저 도착했을 것
    assert after == [900, 901, 902]


def test_upload_rejects_path_traversal(client):
    r = client.post("/upload_video", files={"file": ("../../evil.mp4", b"x", "video/mp4")})
    uploads = Path(__file__).resolve().parent.parent / "uploads"
    assert not (uploads.parent.parent / "evil.mp4").exists()
    assert r.status_code == 400  # 비디오가 아니므로 거부되고, 어디에도 남지 않음
    assert not (uploads / "evil.mp4").exists()


def _write_clip(path, frames=10, fps=10):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for _ in range(frames):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()


def test_upload_valid_video(client, tmp_path):
    src = tmp_path / "clip.avi"
    _write_clip(src)
    uploads = Path(__file__).resolve().parent.parent / "uploads"
    r = client.post("/upload_video", files={"file": ("clip.avi", src.read_bytes(), "video/x-msvideo")})
    try:
        assert r.status_code == 200
        body = r.json()
        assert body["frame_count"] == 10 and body["fps"] == 10 and abs(body["duration"] - 1.0) < 1e-6
        # 목록에 노출되고, 브라우저 탐색용 Range 요청(206)으로 제공되어야 함
        assert any(v["name"] == "clip.avi" and v["url"] == body["video_url"] for v in client.get("/videos").json())
        part = client.get(body["video_url"], headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and len(part.content) == 100
        assert part.content == src.read_bytes()[:100]
    finally:
        (uploads / "clip.avi").unlink(missing_ok=True)


def test_upload_rejects_non_video(client):
    uploads = Path(__file__).resolve().parent.parent / "uploads"
    r = client.post("/upload_video", files={"file": ("junk.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400 and "비디오" in r.json()["detail"]
    assert not (uploads / "junk.mp4").exists()


def test_ball_id_never_collides_with_player_ids():
    t = SoccerTracker(tracker_type='simple')
    objs = t.update([det((i * 60, 0, i * 60 + 50, 100)) for i in range(5)] + [ball((500, 500, 510, 510))])
    ids = [o.track_id for o in objs]
    assert len(set(ids)) == len(ids) and ids.count(SoccerTracker.BALL_TRACK_ID) == 1
