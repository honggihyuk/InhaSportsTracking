"""
정답(ground truth)이 있는 합성 중계 영상 생성기

실제 중계 영상에는 프레임별 정답 호모그래피·선수 좌표가 없어 보정·추적 정확도를 수치로 비교하기 어렵다.
이 모듈은 고정 위치에서 팬·틸트·줌하는 중계 카메라로 FIFA 규격 경기장(잔디 줄무늬, 흰 라인, 관중석)과
선수를 렌더링하고, 프레임별 정답(경기장 → 이미지 호모그래피, 선수·공 경기장 좌표)을 함께 돌려준다.
테스트(tests/test_precise.py)와 벤치마크(scripts/benchmark_precise.py)에서 사용한다.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from .pitch import HALF_LENGTH, HALF_WIDTH, pitch_polylines

PPM = 10                                  # 탑다운 텍스처 해상도 (px/m)
X0, X1, Y0, Y1 = -66.0, 66.0, -46.0, 80.0  # 텍스처가 덮는 경기장 좌표 범위 (관중석 포함)


def _topdown_texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = int((X1 - X0) * PPM), int((Y1 - Y0) * PPM)
    tex = np.zeros((h, w, 3), np.uint8)
    xs = X0 + (np.arange(w) + 0.5) / PPM
    ys = Y1 - (np.arange(h) + 0.5) / PPM
    on_pitch = (np.abs(ys)[:, None] <= HALF_WIDTH + 3) & (np.abs(xs)[None, :] <= HALF_LENGTH + 3)
    stripe = (np.floor((xs + HALF_LENGTH) / 5.5).astype(int) % 2)[None, :].repeat(h, 0)
    grass = np.where(stripe[..., None] == 0, np.array([40, 125, 45]), np.array([50, 142, 55]))
    noise = rng.normal(0, 6, (h, w, 1))
    tex[:] = np.clip(grass + noise, 0, 255)
    # 관중석 / 광고판: 잔디가 아닌 질감
    stands = rng.integers(40, 200, (h // 4 + 1, w // 4 + 1, 3), dtype=np.uint8)
    stands = cv2.resize(stands, (w, h), interpolation=cv2.INTER_NEAREST)
    tex[~on_pitch] = stands[~on_pitch]
    to_px = lambda p: (int(round((p[0] - X0) * PPM * 8)), int(round((Y1 - p[1]) * PPM * 8)))
    for poly in pitch_polylines():
        pts = np.array([to_px(p) for p in poly], np.int32)
        cv2.polylines(tex, [pts], False, (235, 240, 235), thickness=int(0.14 * PPM) + 1,
                      lineType=cv2.LINE_AA, shift=3)
    return tex


@dataclass
class BroadcastCamera:
    """고정 위치(관중석 높은 곳) 카메라. 바라보는 경기장 지점과 초점거리로 팬·틸트·줌"""
    position: Tuple[float, float, float] = (0.0, -62.0, 22.0)
    width: int = 960
    height: int = 540

    def matrices(self, target: Tuple[float, float], focal: float):
        C = np.array(self.position, float)
        fwd = np.array([target[0], target[1], 0.0]) - C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])
        t = -R @ C
        K = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
        P = K @ np.column_stack([R, t])           # 3D (x, y, z) → 이미지
        G = P[:, [0, 1, 3]]                        # 지면 z = 0 → 이미지 호모그래피
        return P, G / G[2, 2]


@dataclass
class SyntheticClip:
    frames: List[np.ndarray]
    G: List[np.ndarray]                     # 경기장 → 이미지 (정답)
    players: List[List[Tuple[int, str, float, float]]]   # (id, team, x, y)
    ball: List[Tuple[float, float]]
    boxes: List[List[Tuple[int, str, Tuple[int, int, int, int]]]] = field(default_factory=list)

    @property
    def H(self) -> List[np.ndarray]:        # 이미지 → 경기장 (정답)
        return [np.linalg.inv(g) / np.linalg.inv(g)[2, 2] for g in self.G]


KITS = {'home': (60, 60, 200), 'away': (235, 235, 235), 'other': (20, 200, 230)}


def render_clip(n: int = 120, width: int = 960, height: int = 540, seed: int = 0,
                pan: float = 22.0, zoom: float = 0.35, noise: float = 3.0, jpeg: int = 85,
                n_players: int = 12) -> SyntheticClip:
    """
    팬(좌우 ±pan m)·줌(초점거리 ±zoom 비율)하는 중계 화면 n 프레임 렌더링

    noise·jpeg 로 센서 잡음과 압축 손실을 흉내 낸다 (광류 카메라 움직임 추정에 실제 같은 오차가 생기도록).
    """
    rng = np.random.default_rng(seed)
    tex = _topdown_texture(seed)
    T = np.array([[1.0 / PPM, 0, X0], [0, -1.0 / PPM, Y1], [0, 0, 1]])   # 텍스처 px → 경기장 m
    cam = BroadcastCamera(width=width, height=height)
    f0 = width * 1.1

    start = np.column_stack([rng.uniform(-30, 30, n_players), rng.uniform(-28, 28, n_players)])
    vel = rng.normal(0, 3.0, (n_players, 2))                                 # m/s
    teams = ['home' if i % 2 == 0 else 'away' for i in range(n_players)]
    teams[-1] = 'other'                                                      # 심판
    fps = 30.0

    frames, Gs, players, balls, boxes = [], [], [], [], []
    for i in range(n):
        s = i / max(n - 1, 1)
        target = (pan * np.sin(2 * np.pi * s), -4.0 + 3.0 * np.sin(np.pi * s))
        focal = f0 * (1 + zoom * np.sin(2 * np.pi * s * 0.75))
        P, G = cam.matrices(target, focal)
        img = cv2.warpPerspective(tex, G @ T, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
        pos = start + vel * (i / fps) + 1.5 * np.sin(np.arange(n_players)[:, None] + i / 15.0)
        ball = (float(pos[0, 0] + 1.0), float(pos[0, 1] + 0.5))
        frame_players, frame_boxes = [], []
        order = np.argsort(-pos[:, 1])  # 먼 선수부터 그려 가까운 선수가 위에 보이도록
        for j in order:
            x, y = pos[j]
            foot = P @ [x, y, 0, 1]
            head = P @ [x, y, 1.8, 1]
            fx, fy = foot[:2] / foot[2]
            hx, hy = head[:2] / head[2]
            hgt = fy - hy
            wid = hgt * 0.38
            box = (int(fx - wid / 2), int(hy), int(fx + wid / 2), int(fy))
            cv2.rectangle(img, (box[0], int(hy + hgt * 0.15)), (box[2], int(hy + hgt * 0.55)), KITS[teams[j]], -1)
            cv2.rectangle(img, (box[0], int(hy + hgt * 0.55)), (box[2], box[3]), (30, 30, 30), -1)
            cv2.circle(img, (int(fx), int(hy + hgt * 0.08)), max(1, int(hgt * 0.08)), (120, 160, 210), -1)
            frame_players.append((int(j) + 1, teams[j], float(x), float(y)))
            frame_boxes.append((int(j) + 1, teams[j], box))
        b = P @ [ball[0], ball[1], 0.11, 1]
        cv2.circle(img, (int(b[0] / b[2]), int(b[1] / b[2])), 3, (250, 250, 250), -1)
        if noise:
            img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        if jpeg:
            img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
        frames.append(img)
        Gs.append(G)
        players.append(frame_players)
        balls.append(ball)
        boxes.append(frame_boxes)
    return SyntheticClip(frames, Gs, players, balls, boxes)


def write_video(clip: SyntheticClip, path, fps: float = 30.0):
    h, w = clip.frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    for f in clip.frames:
        writer.write(f)
    writer.release()


def pitch_error(H_est: np.ndarray, G_true: np.ndarray, width: int, height: int, step: int = 8) -> float:
    """화면에 보이는 경기장 격자점(5 m 간격)을 추정 호모그래피로 되돌렸을 때의 평균 위치 오차 (m)"""
    xs, ys = np.meshgrid(np.arange(-50, 51, 5.0), np.arange(-32, 33, 4.0))
    P = np.column_stack([xs.ravel(), ys.ravel()])
    q = P @ G_true[:, :2].T + G_true[:, 2]
    img = q[:, :2] / q[:, 2:3]
    vis = (q[:, 2] > 0) & (img[:, 0] >= 0) & (img[:, 0] < width) & (img[:, 1] >= 0) & (img[:, 1] < height)
    if not vis.any():
        return float('nan')
    back = cv2.perspectiveTransform(img[vis].reshape(-1, 1, 2).astype(np.float64), H_est).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(back - P[vis], axis=1)))
