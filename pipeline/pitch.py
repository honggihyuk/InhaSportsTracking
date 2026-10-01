"""
경기장 라인 기반 호모그래피 정밀화 (정밀 분석 모드)

카메라 움직임(광류 호모그래피)을 프레임마다 곱해 보정을 전파하면 키프레임에서 멀어질수록
오차가 누적된다. 이 모듈은 매 프레임 화면에 보이는 **흰 경기장 라인**에 FIFA 규격 라인 모델을
정렬해 호모그래피를 다시 맞춘다 — 전파된 값은 초기값으로만 쓰므로 오차가 쌓이지 않는다.

  1. line_mask()        : 잔디 위 얇고 밝은 구조(top-hat) → 라인 픽셀 마스크 (선수 박스 제외)
  2. LineRefiner.refine : 라인 모델 샘플점을 화면에 투영 → 가장 가까운 라인 픽셀과 대응(ICP)
                          → 점-직선 거리 + 예측값 유지 prior 를 강건 최소제곱(soft-L1)으로 최소화
                          → 정렬도가 나아지고 서로 다른 방향의 라인이 충분할 때만 채택
  3. track_homographies : 키프레임에서 앞·뒤로 "예측(카메라 움직임) → 라인 정렬" 을 반복

좌표 규칙은 video_analysis 와 같다 (센터 스폿 원점, x 길이 방향, y 화면 먼 쪽 터치라인 +, 미터).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
CIRCLE_R = 9.15
PB_DEPTH, PB_HALF = 16.5, 20.16      # 페널티 박스
GA_DEPTH, GA_HALF = 5.5, 9.16        # 골 에어리어
SPOT_DIST = 11.0                     # 골라인 → 페널티 스폿


# ---------------------------------------------------------------------------
# 라인 모델
# ---------------------------------------------------------------------------
def _seg(a, b) -> np.ndarray:
    return np.array([a, b], float)


def _arc(cx, cy, r, a0, a1, n=48) -> np.ndarray:
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def pitch_polylines() -> List[np.ndarray]:
    """경기장 라인 (미터) — 직선·원호를 폴리라인 목록으로"""
    L, W = HALF_LENGTH, HALF_WIDTH
    lines = [
        _seg((-L, W), (L, W)), _seg((-L, -W), (L, -W)),          # 터치라인
        _seg((-L, -W), (-L, W)), _seg((L, -W), (L, W)),          # 골라인
        _seg((0, -W), (0, W)),                                   # 하프라인
        _arc(0, 0, CIRCLE_R, 0, 360, 96),                        # 센터서클
    ]
    # 페널티 아크: 스폿 중심 원 중 박스 밖 부분 (박스 앞선 x = L - 16.5)
    half_angle = np.degrees(np.arccos((PB_DEPTH - SPOT_DIST) / CIRCLE_R))
    for s in (-1, 1):
        gx, pb, ga = s * L, s * (L - PB_DEPTH), s * (L - GA_DEPTH)
        lines += [
            _seg((gx, PB_HALF), (pb, PB_HALF)), _seg((gx, -PB_HALF), (pb, -PB_HALF)), _seg((pb, -PB_HALF), (pb, PB_HALF)),
            _seg((gx, GA_HALF), (ga, GA_HALF)), _seg((gx, -GA_HALF), (ga, -GA_HALF)), _seg((ga, -GA_HALF), (ga, GA_HALF)),
        ]
        spot = s * (L - SPOT_DIST)
        center_angle = 0.0 if s < 0 else 180.0  # 아크는 경기장 중앙 쪽으로 볼록
        lines.append(_arc(spot, 0, CIRCLE_R, center_angle - half_angle, center_angle + half_angle, 32))
    return lines


def sample_pitch_lines(step: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """라인 위 등간격 샘플점과 각 점의 접선 방향 (N, 2), (N, 2)"""
    pts, tans = [], []
    for poly in pitch_polylines():
        for a, b in zip(poly[:-1], poly[1:]):
            d = b - a
            length = float(np.hypot(*d))
            if length < 1e-9:
                continue
            n = max(1, int(np.ceil(length / step)))
            t = (np.arange(n) / n)[:, None]
            pts.append(a + t * d)
            tans.append(np.repeat((d / length)[None], n, axis=0))
    return np.concatenate(pts), np.concatenate(tans)


# ---------------------------------------------------------------------------
# 라인 픽셀 검출
# ---------------------------------------------------------------------------
def _grass(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))


def pitch_region(frame: np.ndarray, k: int = 11) -> np.ndarray:
    """
    경기장(잔디) 영역 — 관중석 속 초록 점들은 빼고, 라인·선수가 만든 구멍은 메운다

    잔디 마스크 → 열림(작은 점 제거) → 닫힘(라인 메우기) → 큰 연결 영역만 → 내부 구멍 채우기 → 살짝 팽창(바깥 라인 포함)
    """
    h, w = frame.shape[:2]
    grass = _grass(frame)
    ell = lambda s: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, ell(5))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, ell(2 * k + 1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grass, connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.03 * h * w]
    field = np.isin(labels, big).astype(np.uint8) * 255 if big else np.zeros_like(grass)
    contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(field, contours, -1, 255, thickness=cv2.FILLED)
    return cv2.dilate(field, ell(k))


def line_mask(frame: np.ndarray, boxes: Sequence[Sequence[float]] = (), box_scale: float = 1.0
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    frame(BGR) → (라인 픽셀 마스크, 유효 영역 마스크) — 둘 다 uint8 0/255, frame 과 같은 크기

    라인 = 잔디 영역 안에서 주변보다 밝고(top-hat) 얇은 구조. 흰 유니폼 등 선수 박스 안은 제외한다.
    유효 영역 = 라인이 보일 수 있는 곳(구멍을 메운 잔디 영역 − 선수 박스). 투영된 라인 모델점 중
    유효 영역 밖(관중석·가려진 곳)에 떨어진 점은 정렬에 쓰지 않는다.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(7, (h // 50) | 1)                         # 540p 에서 11 px — 라인 폭보다 넉넉히
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    tophat = cv2.GaussianBlur(tophat.astype(np.float32), (0, 0), 1.0)
    bright = tophat > 18

    # 라인 중심선만 남김 (능선 검출): 굵은 라인의 가장자리 픽셀에 맞추면 라인 폭 절반만큼 치우친다.
    # 라인에 수직인 방향(곡률이 큰 축)에서 극대인 픽셀만 유지
    pad = np.pad(tophat, 1, mode='edge')
    left, right = pad[1:-1, :-2], pad[1:-1, 2:]
    up, down = pad[:-2, 1:-1], pad[2:, 1:-1]
    dxx, dyy = left + right - 2 * tophat, up + down - 2 * tophat
    ridge = np.where(dxx < dyy, (tophat > left) & (tophat >= right), (tophat > up) & (tophat >= down))
    bright = (bright & ridge).astype(np.uint8) * 255

    field = pitch_region(frame, k)
    mask = cv2.bitwise_and(bright, field)

    valid = field.copy()
    for x1, y1, x2, y2 in boxes:
        bw, bh = (x2 - x1) * box_scale, (y2 - y1) * box_scale
        a = (max(0, int(x1 * box_scale - 0.1 * bw)), max(0, int(y1 * box_scale - 0.05 * bh)))
        b = (min(w - 1, int(x2 * box_scale + 0.1 * bw)), min(h - 1, int(y2 * box_scale + 0.05 * bh)))
        cv2.rectangle(mask, a, b, 0, -1)
        cv2.rectangle(valid, a, b, 0, -1)

    # 점처럼 작은 조각(잔디 반사·노이즈) 제거
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < 6)[0]
    if len(small) > 1:
        mask[np.isin(labels, small[small > 0])] = 0
    return mask, valid


# ---------------------------------------------------------------------------
# ICP 정밀화
# ---------------------------------------------------------------------------
def scale_matrix(s: float) -> np.ndarray:
    """원본 픽셀 → 축소 이미지 픽셀 (cv2.resize 의 픽셀 중심 규약: x_s = s·(x + 0.5) − 0.5)"""
    o = 0.5 * s - 0.5
    return np.array([[s, 0, o], [0, s, o], [0, 0, 1.0]])


def _project(G: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pitch(N,2) → image(N,2), 동차 좌표 w (카메라 앞이면 > 0)"""
    q = P @ G[:, :2].T + G[:, 2]
    return q[:, :2] / q[:, 2:3], q[:, 2]


@dataclass
class RefineResult:
    H: np.ndarray            # 이미지(원본 해상도) → 경기장
    accepted: bool
    inliers: int             # 라인과 1.5 px 이내로 맞은 모델점 수 (작업 해상도 기준)
    ratio: float             # 보이는 모델점 중 맞은 비율
    ratio_before: float


class LineRefiner:
    """
    라인 마스크에 경기장 라인 모델을 정렬해 호모그래피를 정밀화

    Args:
        scale: 마스크 해상도 / 원본 해상도 (예: 0.5)
        step: 모델 샘플 간격 (m)
        min_inliers: 채택에 필요한 최소 정렬 점 수
        prior_weight: 예측 호모그래피를 유지하려는 정도 (라인이 한 방향만 보일 때 미끄러짐 방지)
    """

    def __init__(self, scale: float = 0.5, step: float = 0.5, min_inliers: int = 40,
                 prior_weight: float = 0.05, radii: Sequence[float] = (12, 6, 3), inlier_px: float = 1.5):
        self.scale = scale
        self.P, self.T = sample_pitch_lines(step)
        self.min_inliers = min_inliers
        self.prior_weight = prior_weight
        self.radii = tuple(radii)
        self.inlier_px = inlier_px

    # -- 거리 변환: 각 픽셀에서 가장 가까운 라인 픽셀까지의 거리와 그 좌표 --------------------
    @staticmethod
    def _nearest(mask: np.ndarray):
        src = np.where(mask > 0, 0, 255).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(src == 0)                 # 라벨 k (1 부터) = 행 우선 순서 k 번째 0 픽셀
        lut = np.stack([np.concatenate([[0], xs]), np.concatenate([[0], ys])], axis=1).astype(np.float32)
        return dist, labels, lut

    def _visible(self, G, valid):
        h, w = valid.shape
        p, z = _project(G, self.P)
        ix, iy = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        inside = (z > 0) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        vis = np.zeros(len(p), bool)
        vis[inside] = valid[iy[inside], ix[inside]] > 0
        return p, vis, ix, iy

    def _fit_quality(self, G, dist, valid) -> Tuple[int, float]:
        p, vis, ix, iy = self._visible(G, valid)
        if vis.sum() == 0:
            return 0, 0.0
        good = dist[iy[vis], ix[vis]] <= self.inlier_px
        return int(good.sum()), float(good.mean())

    def refine(self, H: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> RefineResult:
        """H: 원본 이미지 → 경기장 (예측값). mask/valid: line_mask() 결과 (scale 해상도)"""
        from scipy.optimize import least_squares

        S = scale_matrix(self.scale)
        try:
            G0 = S @ np.linalg.inv(H)                 # 경기장 → 작업 해상도 이미지
        except np.linalg.LinAlgError:
            return RefineResult(H, False, 0, 0.0, 0.0)
        G0 = G0 / G0[2, 2] if abs(G0[2, 2]) > 1e-12 else G0
        h, w = mask.shape
        if mask.max() == 0:
            return RefineResult(H, False, 0, 0.0, 0.0)
        dist, labels, lut = self._nearest(mask)
        n_before, ratio_before = self._fit_quality(G0, dist, valid)

        # 정규화 좌표계 (조건수 개선): C · 이미지 → [-1, 1]
        C = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
        Ci = np.linalg.inv(C)
        grid = np.array([[x, y] for x in (0.1 * w, 0.5 * w, 0.9 * w) for y in (0.1 * h, 0.5 * h, 0.9 * h)])
        grid_pitch, _ = _project(np.linalg.inv(G0), grid)

        def compose(d, base):
            D = np.array([[d[0], d[1], d[2]], [d[3], d[4], d[5]], [d[6], d[7], 0.0]])
            return Ci @ (np.eye(3) + D) @ C @ base

        G = G0
        for radius in self.radii:
            p, vis, ix, iy = self._visible(G, valid)
            if vis.sum() < self.min_inliers:
                break
            idx = np.where(vis)[0]
            d = dist[iy[idx], ix[idx]]
            near = d <= radius
            idx = idx[near]
            if len(idx) < self.min_inliers:
                break
            q = lut[labels[iy[idx], ix[idx]]]
            # 대응점의 법선 = 투영된 모델 접선의 수직 방향 (점-직선 거리 → 라인을 따라 미끄러져도 비용 없음)
            p_t, _ = _project(G, self.P[idx] + 0.5 * self.T[idx])
            tang = p_t - p[idx]
            tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
            normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
            Pm, base = self.P[idx], G

            def residuals(dvec):
                Gc = compose(dvec, base)
                proj, _ = _project(Gc, Pm)
                line_r = np.sum((proj - q) * normal, axis=1)
                prior, _ = _project(Gc, grid_pitch)
                prior_r = self.prior_weight * (prior - grid).ravel()
                return np.concatenate([line_r, prior_r])

            sol = least_squares(residuals, np.zeros(8), loss='soft_l1', f_scale=1.0, max_nfev=60, x_scale=1e-3)
            G = compose(sol.x, base)

        n_after, ratio_after = self._fit_quality(G, dist, valid)
        ok = (n_after >= self.min_inliers and ratio_after >= ratio_before
              and self._well_constrained(G, dist, valid) and self._plausible(G0, G, w, h))
        if not ok:
            return RefineResult(H, False, n_before, ratio_before, ratio_before)
        Hn = np.linalg.inv(np.linalg.inv(S) @ G)
        return RefineResult(Hn / Hn[2, 2], True, n_after, ratio_after, ratio_before)

    def _well_constrained(self, G, dist, valid) -> bool:
        """맞은 점들의 라인 방향이 두 방향 이상이어야 호모그래피가 결정된다 (한 직선만 보이면 거부)"""
        p, vis, ix, iy = self._visible(G, valid)
        good = np.zeros(len(p), bool)
        good[vis] = dist[iy[vis], ix[vis]] <= self.inlier_px
        if good.sum() < self.min_inliers:
            return False
        p_t, _ = _project(G, self.P[good] + 0.5 * self.T[good])
        t = p_t - p[good]
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        ev = np.linalg.eigvalsh(t.T @ t / len(t))
        return ev[0] / max(ev[1], 1e-9) >= 0.04

    @staticmethod
    def _plausible(G0, G, w, h, max_shift: float = 0.08) -> bool:
        """한 프레임 정렬로 화면 격자가 너무 크게(화면 폭 8 % 이상) 움직이면 잘못 맞춘 것으로 본다"""
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], float)
        Pc, _ = _project(np.linalg.inv(G0), corners)
        moved, z = _project(G, Pc)
        if np.any(z <= 0):
            return False
        return float(np.max(np.linalg.norm(moved - corners, axis=1))) <= max_shift * w


# ---------------------------------------------------------------------------
# 영상 전체 라인 마스크 & 드리프트 보정 전파
# ---------------------------------------------------------------------------
class LineMaskStore:
    """영상 전체의 (라인, 유효 영역) 마스크를 비트 압축해 메모리에 보관 (540p 기준 프레임당 약 130 KB)"""

    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape
        self._data: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []

    def append(self, mask: Optional[np.ndarray], valid: Optional[np.ndarray] = None):
        if mask is None:
            self._data.append(None)
        else:
            if not self._data or self.shape == (0, 0):
                self.shape = mask.shape[:2]       # 빈 저장소는 첫 마스크 크기를 따른다
            self._data.append((np.packbits(mask > 0), np.packbits(valid > 0)))

    def __len__(self):
        return len(self._data)

    def save(self, path):
        """압축 저장 (라인·유효 영역 마스크는 희소/단순해 프레임당 수 KB 로 줄어든다)"""
        nbytes = (self.shape[0] * self.shape[1] + 7) // 8
        present = np.array([d is not None for d in self._data], bool)
        lines = np.zeros((len(self._data), nbytes), np.uint8)
        valid = np.zeros_like(lines)
        for i, d in enumerate(self._data):
            if d is not None:
                lines[i], valid[i] = d
        tmp = str(path) + '.tmp.npz'
        np.savez_compressed(tmp, shape=np.array(self.shape), present=present, lines=lines, valid=valid)
        import os
        os.replace(tmp, path)

    @classmethod
    def load(cls, path) -> 'LineMaskStore':
        z = np.load(path)
        store = cls(tuple(int(v) for v in z['shape']))
        store._data = [(l, v) if p else None for p, l, v in zip(z['present'], z['lines'], z['valid'])]
        return store

    def __call__(self, t: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not 0 <= t < len(self._data) or self._data[t] is None:
            return None
        n = self.shape[0] * self.shape[1]
        m, v = self._data[t]
        unpack = lambda b: (np.unpackbits(b, count=n).reshape(self.shape) * 255).astype(np.uint8)
        return unpack(m), unpack(v)


def video_line_masks(video_path, frames_objs: Optional[List[list]] = None, scale: float = 0.5,
                     max_frames: Optional[int] = None,
                     progress: Callable[[int, int], None] = lambda d, t: None) -> LineMaskStore:
    """영상을 한 번 읽어 프레임별 라인 마스크를 만든다. frames_objs: 분석 결과 frames (선수 박스 제외용)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    store = None
    try:
        for t in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if store is None:
                store = LineMaskStore(small.shape[:2])
            boxes = [o[2:6] for o in frames_objs[t]] if frames_objs and t < len(frames_objs) else ()
            store.append(*line_mask(small, boxes, box_scale=scale))
            progress(t + 1, total)
    finally:
        cap.release()
    return store or LineMaskStore((1, 1))


def track_homographies(motion: List[Optional[list]], keyframes: Dict[int, np.ndarray],
                       masks: Optional[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
                       refiner: Optional[LineRefiner] = None, refine_keyframes: bool = True
                       ) -> Tuple[List[Optional[np.ndarray]], List[Optional[float]]]:
    """
    키프레임 보정을 전 프레임에 전파하되, 매 프레임 라인 정렬로 누적 오차를 제거

    예측: H_t = H_{t-1} · M_t⁻¹ (앞으로), H_t = H_{t+1} · M_{t+1} (뒤로) → 라인 정렬로 정밀화.
    라인이 부족한 프레임은 예측값을 그대로 쓰고 다음 프레임에서 다시 정렬한다.
    masks 가 없으면 video_analysis.frame_homographies 와 같은 결과 (정렬 없음).

    Returns:
        (프레임별 이미지 → 경기장 호모그래피, 프레임별 라인 정렬 비율 — 정렬 안 됐으면 None)
    """
    refiner = refiner or LineRefiner()
    n = len(motion)
    Ms = [None if m is None else np.array(m, float).reshape(3, 3) for m in motion]
    H: List[Optional[np.ndarray]] = [None] * n
    quality: List[Optional[float]] = [None] * n
    steps = [np.inf] * n

    def refined(Ht, t):
        data = masks(t) if masks else None
        if data is None:
            return Ht, None
        r = refiner.refine(Ht, *data)
        return r.H, (round(r.ratio, 3) if r.accepted else None)

    for k, Hk in keyframes.items():
        if not 0 <= k < n:
            continue
        Hk_ref, qk = refined(Hk, k) if refine_keyframes else (Hk, None)
        H[k], quality[k], steps[k] = Hk_ref, qk, 0
        for direction in (1, -1):
            cur = Hk_ref
            t = k + direction
            while 0 <= t < n:
                M = Ms[t] if direction > 0 else Ms[t + 1]
                if M is None:
                    break                                  # 장면 전환 — 그 너머로는 전파하지 않음
                pred = cur @ np.linalg.inv(M) if direction > 0 else cur @ M
                if abs(t - k) < steps[t]:
                    cur, q = refined(pred, t)
                    H[t], quality[t], steps[t] = cur, q, abs(t - k)
                else:
                    break                                  # 더 가까운 다른 키프레임이 이미 맡은 구간
                t += direction
    return H, quality
