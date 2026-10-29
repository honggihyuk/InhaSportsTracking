"""경기장 보정 추적 테스트 — 앵커 재보정·양방향 연결·적응형 평활화·신뢰도, 정지 오버레이 제외, 라인 정렬 수정

합성 중계 영상(pipeline/synthetic.py)의 정답 호모그래피와 비교한다.
실행: python -m pytest tests -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import calib_track as ct, pitch, postprocess as pp, synthetic as syn, video_analysis as va


@pytest.fixture(scope="module")
def clip():
    return syn.render_clip(n=60, seed=3)


@pytest.fixture(scope="module")
def data(clip):
    s = 0.5
    store = pitch.LineMaskStore((0, 0))
    est = va.CameraMotion()
    motion = []
    for i, f in enumerate(clip.frames):
        small = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        store.append(*pitch.line_mask(small, [b for _, _, b in clip.boxes[i]], box_scale=s))
        M = est(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), [b for _, _, b in clip.boxes[i - 1]] if i else [])
        motion.append(None if M is None else M.ravel().tolist())
    return motion, store


def err(H, clip, t):
    return syn.pitch_error(H, clip.G[t], 960, 540)


def stands_view():
    """화면 위쪽이 지평선 너머(관중석)인 장면 — 먼 터치라인 쪽을 넓게 비춤"""
    cam = syn.BroadcastCamera()
    _, G = cam.matrices((-8.0, 30.0), 960 * 0.85)
    T = np.array([[1 / syn.PPM, 0, syn.X0], [0, -1 / syn.PPM, syn.Y1], [0, 0, 1]])
    img = cv2.warpPerspective(syn._topdown_texture(0), G @ T, (960, 540))
    return img, G


# --- 부호 정규화 -------------------------------------------------------------------
def test_orient_makes_pitch_side_positive():
    img, G = stands_view()
    H = np.linalg.inv(G)
    assert (H @ [0, 0, 1.0])[2] * (H @ [480, 500, 1.0])[2] < 0      # 전제: 왼쪽 위는 지평선 너머
    Ho = ct.orient(-H / H[2, 2], 960, 540)                          # 예전 저장 방식(H/H[2,2])은 부호가 뒤집힘
    Gi = np.linalg.inv(Ho)
    w = Gi[2] @ [0.0, 0.0, 1.0]                                     # 센터 스폿 — 카메라 앞
    assert w > 0


# --- 라인 정렬: 화면 위가 관중석이어도 채택 -----------------------------------------------
def test_refiner_accepts_when_top_of_frame_is_stands():
    img, G = stands_view()
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mask, valid = pitch.line_mask(small)
    shift = np.array([[1, 0, 0.6], [0, 1, -0.4], [0, 0, 1.0]])       # 경기장에서 0.7 m 어긋난 예측값
    H_pred = shift @ np.linalg.inv(G)
    r = pitch.LineRefiner(scale=0.5).refine(H_pred, mask, valid)
    assert r.accepted
    assert syn.pitch_error(r.H, G, 960, 540) < 0.5 * syn.pitch_error(H_pred, G, 960, 540)


# --- 정지 오버레이 -----------------------------------------------------------------
def test_camera_motion_ignores_static_overlay():
    rng = np.random.default_rng(0)
    world = cv2.GaussianBlur(rng.integers(0, 255, (400, 1400), dtype=np.uint8), (0, 0), 1.5)
    est = va.CameraMotion(scale=1.0)
    total_plain, total_est, true = 0.0, 0.0, 0.0
    prev = None
    for t in range(60):
        dx = 6.0 if t < 20 else 1.0                                # 빠른 팬 뒤 느린 팬 (프레임당 1 px)
        true += dx if t else 0.0
        M = np.float32([[1, 0, -true], [0, 1, 0]])
        frame = cv2.warpAffine(world, M, (480, 270), flags=cv2.INTER_LINEAR)
        frame[10:60, 20:200] = 0                                   # 점수판 (화면에 고정)
        cv2.putText(frame, '02:42 RMA 0-0 FCB', (24, 45), 0, 0.6, 255, 2)
        Me = est(frame, [])
        if prev is not None:
            Mp = va.camera_motion(prev, frame, [], 1.0)
            if t >= 20:                                            # 느린 팬 구간의 누적 이동 비교
                total_plain += -Mp[0, 2]
                total_est += -Me[0, 2]
        prev = frame
    slow_true = 1.0 * 40
    assert est.exclude_mask() is not None and est.exclude_mask()[30, 100] > 0      # 점수판을 찾아 뺌
    assert abs(total_est - slow_true) < abs(total_plain - slow_true) or abs(total_est - slow_true) < 0.5


# --- 앵커 추적 -------------------------------------------------------------------
def test_anchored_tracking_corrects_motion_drift(clip, data):
    motion, store = data
    # 카메라 움직임에 프레임당 0.6 px 치우침을 넣어 누적 오차를 만든다 (키프레임 1 개, 60 프레임 → 36 px)
    drift = np.array([[1, 0, 0.6], [0, 1, 0], [0, 0, 1.0]])
    biased = [None if m is None else (drift @ np.array(m).reshape(3, 3)).ravel().tolist() for m in motion]
    Hs_plain = va.frame_homographies(biased, {0: clip.H[0]})
    Hs, conf, info = ct.anchored_homographies(biased, {0: clip.H[0]}, (960, 540), 30.0, masks=store, scale=0.5)
    e_plain = np.mean([err(H, clip, t) for t, H in enumerate(Hs_plain)])
    e = [err(H, clip, t) for t, H in enumerate(Hs)]
    assert info['tracked'] >= 10
    assert np.mean(e) < 0.2 and np.mean(e) < 0.5 * e_plain
    assert all(c is not None and 0.0 <= c <= 1.0 for c in conf) and np.mean(conf) > 0.7


def test_blending_between_disagreeing_keyframes_has_no_jump(clip, data):
    motion, _ = data
    # 라인 없이 키프레임 둘만: 오른쪽 키프레임이 1.5 m 어긋나 있음 → 중간에서 한 번에 튀지 않고 서서히 넘어가야
    off = np.array([[1, 0, 1.5], [0, 1, 0], [0, 0, 1.0]])
    Hs, conf, info = ct.anchored_homographies(motion, {0: clip.H[0], 59: off @ clip.H[59]}, (960, 540), 30.0)
    slide = ct.slide_speed(Hs, ct.MotionChain(motion), (960, 540), 30.0)
    assert slide.max() < 2.0                                         # 1.5 m / 2 초 ≈ 0.76 m/s 로 고르게
    nearest = va.frame_homographies(motion, {0: clip.H[0], 59: off @ clip.H[59]})
    jump = ct.slide_speed(nearest, ct.MotionChain(motion), (960, 540), 30.0)
    assert jump.max() > 20                                           # 예전 방식: 담당 구간이 바뀌는 곳에서 튐


def test_wrong_keyframe_lowers_confidence_and_tracking_prefers_lines(clip, data):
    motion, store = data
    bad = np.array([[1, 0, 4.0], [0, 1, 0], [0, 0, 1.0]]) @ clip.H[30]      # 4 m 어긋난 자동 키프레임
    Hs, conf, info = ct.anchored_homographies(motion, {0: clip.H[0], 30: bad}, (960, 540), 30.0,
                                              masks=store, scale=0.5, keyframe_conf={30: 0.6})
    assert info['replaced'] >= 1                                     # 이웃에서 추적한 값이 더 잘 맞아 교체
    assert err(Hs[30], clip, 30) < 0.5


# --- 신뢰도 → 속력 평활화 -----------------------------------------------------------
def test_low_confidence_frames_do_not_spike_speed():
    fps = 30.0
    world, conf = [], []
    for t in range(60):
        x = 0.1 * t + (3.0 if 28 <= t < 32 else 0.0)                 # 4 프레임 동안 보정이 3 m 튐
        world.append([[1, x, 0.0]])
        conf.append(0.0 if 27 <= t < 33 else 1.0)
    plain = max(e[3] for w in pp.smooth_world(world, fps) for e in w)
    weighted = max(e[3] for w in pp.smooth_world(world, fps, frame_conf=conf) for e in w)
    base = 3.0                                                       # 실제 속력 0.1 m/프레임 = 3 m/s
    assert plain > base + 0.5 and weighted - base < 0.3 * (plain - base)
    assert pp.conf_noise(None) == 1.0 and pp.conf_noise(1.0) == 1.0 and pp.conf_noise(0.0) == pytest.approx(30.0)


def test_calibrate_records_confidence_and_oriented_homographies(clip, data):
    motion, store = data
    result = {'fps': 30.0, 'width': 960, 'height': 540, 'frames': [[] for _ in clip.frames], 'motion': motion,
              'mode': 'precise', 'teams': {}}
    pts = []
    for x, y in [(0, 34), (0, 9.15), (0, -9.15), (-36, 20.16), (36, 20.16), (-36, -20.16), (36, -20.16)]:
        q = clip.G[0] @ [x, y, 1]
        u, v = q[0] / q[2], q[1] / q[2]
        if 0 <= u < 960 and 0 <= v < 540:
            pts.append({'image': [u, v], 'pitch': [x, y]})
    va.calibrate(result, [{'frame': 0, 'points': pts}], line_masks=store, smooth=False)
    cal = result['calibration']
    assert cal['method'] == 'anchored' and len(cal['confidence']) == len(clip.frames)
    assert cal['refined_frames'] > len(clip.frames) // 2
    for h in result['homographies']:
        Gi = np.linalg.inv(np.array(h).reshape(3, 3))
        assert (Gi @ [0, 0, 1.0])[2] > 0                               # 화면 쪽 경기장이 카메라 앞 (w > 0)
