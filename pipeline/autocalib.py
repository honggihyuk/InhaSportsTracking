"""
자동 초기 보정 — 학습 모델 없이 경기장 라인 기하로 첫 호모그래피를 찾는다

4부(pitch.py)의 라인 정렬은 "대략 맞는 초기값"을 정밀하게 만드는 단계였고, 첫 키프레임은 사람이 찍어야 했다.
이 모듈은 그 초기값 자체를 찾는다.

  1. detect_lines()      : 라인 마스크에서 긴 직선을 찾아 무한 직선(a·x + b·y = c)으로 정리
  2. 가설 열거            : 직선 2 개를 길이 방향 라인(y = 상수: 터치라인·박스선)에, 2 개를 폭 방향 라인
                           (x = 상수: 골라인·박스 옆선·하프라인)에 대응 → 네 교점으로 호모그래피
                           - 화면 위쪽 직선 = 먼 쪽(y 큼), 화면 왼쪽 직선 = x 작음 (중계 카메라는 거울상이 없음)
                           - 같은 방향 직선끼리는 화면 안에서 만나지 않음 (경기장에서 평행)
  3. 1 단계 점수 (일괄)   : 투영한 모델 라인 점이 라인 픽셀에 얼마나 떨어지는가 (정밀도 × 맞은 점 수)
  4. 2 단계 점수 (상위)   : 라인 픽셀이 투영 모델로 얼마나 설명되는가(재현율) → F1
  5. 정밀화·확신도        : 상위 가설을 LineRefiner 로 정렬한 뒤 다시 채점, 서로 다른 해석(화면 중심이
                           4 m 이상 떨어진 가설) 중 2 등과의 차이가 충분해야 채택

auto_keyframes() 는 영상 전체에서 일정 간격(+ 장면 전환 직후) 프레임에 이를 적용하고, 이웃 자동 키프레임끼리
카메라 움직임으로 서로 맞는지 교차 검증해 틀린 키프레임을 버린다.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .pitch import (GA_DEPTH, GA_HALF, HALF_LENGTH, HALF_WIDTH, PB_DEPTH, PB_HALF, LineRefiner,
                    pitch_polylines, sample_pitch_lines, scale_matrix)

# 길이 방향(y = 상수) / 폭 방향(x = 상수) 직선 모델
Y_LINES = (HALF_WIDTH, PB_HALF, GA_HALF, -GA_HALF, -PB_HALF, -HALF_WIDTH)
X_LINES = (-HALF_LENGTH, -(HALF_LENGTH - GA_DEPTH), -(HALF_LENGTH - PB_DEPTH), 0.0,
           HALF_LENGTH - PB_DEPTH, HALF_LENGTH - GA_DEPTH, HALF_LENGTH)


@dataclass
class ImageLine:
    n: np.ndarray        # 단위 법선 (a, b)
    c: float             # a·x + b·y = c
    length: float        # 지지 픽셀 길이
    angle: float         # 직선 방향 (도, 수평 0, -90 ~ 90)

    def at_x(self, x: float) -> float:
        a, b = self.n
        return (self.c - a * x) / b if abs(b) > 1e-9 else np.inf

    def at_y(self, y: float) -> float:
        a, b = self.n
        return (self.c - b * y) / a if abs(a) > 1e-9 else np.inf


def _intersect(l1: ImageLine, l2: ImageLine) -> Optional[np.ndarray]:
    A = np.array([l1.n, l2.n])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [l1.c, l2.c])


# ---------------------------------------------------------------------------
# 직선 검출
# ---------------------------------------------------------------------------
def detect_lines(mask: np.ndarray, max_lines: int = 9, min_length: float = 50.0) -> List[ImageLine]:
    """라인 마스크(중심선) → 긴 직선 목록 (지지 길이 순)"""
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=25, minLineLength=int(min_length * 0.6), maxLineGap=8)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lengths)
    ys, xs = np.nonzero(mask)
    pix = np.column_stack([xs, ys]).astype(float)
    lines: List[ImageLine] = []
    used = np.zeros(len(segs), bool)
    for i in order:
        if used[i]:
            continue
        x1, y1, x2, y2 = segs[i]
        d = np.array([x2 - x1, y2 - y1]) / max(lengths[i], 1e-9)
        n = np.array([-d[1], d[0]])
        c = float(n @ [x1, y1])
        # 같은 직선 위의 다른 조각을 모두 흡수
        for j in order:
            if used[j]:
                continue
            p, q = segs[j, :2], segs[j, 2:]
            if abs(n @ p - c) < 3 and abs(n @ q - c) < 3:
                used[j] = True
        # 직선 근처 라인 픽셀로 다시 맞춤 (Hough 양자화 제거) → 가장 긴 연속 구간만 직선으로 인정
        fit = _fit_run(pix, n, c, min_length)
        if fit is None:
            continue
        n, c, support, (vx, vy) = fit
        if any(abs(L.n @ n) > 0.9995 and abs(abs(L.c) - abs(c)) < 4 for L in lines):
            continue  # 이미 찾은 직선과 같음
        angle = float(np.degrees(np.arctan2(vy, vx)))
        angle = (angle + 90) % 180 - 90
        lines.append(ImageLine(n / np.linalg.norm(n), c, support, angle))
        if len(lines) >= max_lines:
            break
    lines.sort(key=lambda L: -L.length)
    return lines


def _fit_run(pix: np.ndarray, n: np.ndarray, c: float, min_length: float):
    """
    직선 (n, c) 근처 픽셀 중 가장 긴 연속 구간(틈 8 px 이하)으로 다시 맞춤.
    곡선(센터서클·아크의 접선 부분)은 구간 앞·뒤 1/3 의 방향이 달라 거부된다.
    Returns: (n, c, 구간 길이, 방향) 또는 None
    """
    d = np.array([n[1], -n[0]])
    for tol in (2.5, 1.5):
        near = np.abs(pix @ n - c) < tol
        if near.sum() < min_length * 0.7:
            return None
        pts = pix[near]
        t = pts @ d
        order = np.argsort(t)
        t, pts = t[order], pts[order]
        breaks = np.where(np.diff(t) > 8)[0] + 1
        runs = np.split(np.arange(len(t)), breaks)
        run = max(runs, key=lambda r: t[r[-1]] - t[r[0]] if len(r) else -1)
        if t[run[-1]] - t[run[0]] < min_length:
            return None
        sel = pts[run].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        n = np.array([-vy, vx], float)
        c = float(n @ [x0, y0])
        d = np.array([vx, vy], float)
    third = max(3, len(sel) // 3)
    dirs = []
    for part in (sel[:third], sel[-third:]):
        if np.ptp(part @ d) < 12:
            return None
        ux, uy = cv2.fitLine(part, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
        dirs.append(np.array([ux, uy]))
    if abs(dirs[0] @ dirs[1]) < np.cos(np.radians(2.0)):
        return None                                       # 휘어 있음 → 곡선의 일부
    rms = float(np.sqrt(np.mean((sel @ n - c) ** 2)))
    if rms > 0.8:
        return None
    length = float((sel @ d).max() - (sel @ d).min())
    return n, c, length, (float(vx), float(vy))


# ---------------------------------------------------------------------------
# 호모그래피 일괄 계산 (DLT, 4 점)
# ---------------------------------------------------------------------------
def _batch_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src, dst: (K, 4, 2) → (K, 3, 3) (src → dst), 특이하면 nan"""
    K = len(src)
    A = np.zeros((K, 8, 8))
    b = np.zeros((K, 8))
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    A[:, 0::2, 0], A[:, 0::2, 1], A[:, 0::2, 2] = x, y, 1
    A[:, 0::2, 6], A[:, 0::2, 7] = -u * x, -u * y
    A[:, 1::2, 3], A[:, 1::2, 4], A[:, 1::2, 5] = x, y, 1
    A[:, 1::2, 6], A[:, 1::2, 7] = -v * x, -v * y
    b[:, 0::2], b[:, 1::2] = u, v
    H = np.full((K, 3, 3), np.nan)
    ok = np.abs(np.linalg.det(A)) > 1e-12
    if ok.any():
        h = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
        H[ok] = np.concatenate([h, np.ones((ok.sum(), 1))], axis=1).reshape(-1, 3, 3)
    return H


# ---------------------------------------------------------------------------
# 단일 프레임 자동 보정
# ---------------------------------------------------------------------------
@dataclass
class AutoResult:
    H: Optional[np.ndarray]      # 원본 이미지 → 경기장 (확신할 때만, 아니면 None)
    f1: float
    precision: float             # 보이는 모델 점 중 라인 위에 떨어진 비율
    recall: float                # 라인 픽셀 중 투영 모델로 설명되는 비율
    margin: float                # 2 등 해석의 F1 / 1 등 F1 (작을수록 확실)
    lines: int
    hypotheses: int
    reason: str = ''


class AutoCalibrator:
    """
    Args:
        scale: 라인 마스크 해상도 / 원본 해상도
        min_f1, min_recall: 채택 기준
        max_margin: 2 등 해석 F1 이 1 등의 이 비율을 넘으면 애매한 장면으로 보고 거부
    """

    def __init__(self, scale: float = 0.5, min_f1: float = 0.6, min_recall: float = 0.55,
                 max_margin: float = 0.85, hit_px: float = 2.5):
        self.scale = scale
        self.min_f1, self.min_recall, self.max_margin = min_f1, min_recall, max_margin
        self.hit_px = hit_px
        self.P, _ = sample_pitch_lines(1.0)
        self.polylines = pitch_polylines()
        self.refiner = LineRefiner(scale=scale)

    # -- 채점 --------------------------------------------------------------
    def _stage1(self, G: np.ndarray, dist: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """G: (K, 3, 3) 경기장 → 작업 이미지. 반환: (맞은 점 수, 보이는 점 수)"""
        h, w = dist.shape
        q = np.einsum('kij,nj->kni', G, np.column_stack([self.P, np.ones(len(self.P))]))
        z = q[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.round(q[..., 0] / z)
            v = np.round(q[..., 1] / z)
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui = np.where(inside, u, 0).astype(int)
        vi = np.where(inside, v, 0).astype(int)
        vis = inside & (valid[vi, ui] > 0)
        hit = vis & (dist[vi, ui] <= self.hit_px)
        return hit.sum(1), vis.sum(1)

    def _render(self, G: np.ndarray, shape) -> np.ndarray:
        canvas = np.zeros(shape, np.uint8)
        h, w = shape
        for poly in self.polylines:
            dense = _densify(poly, 1.0)
            q = dense @ G[:, :2].T + G[:, 2]
            ok = q[:, 2] > 1e-6
            pts = q[:, :2] / np.where(ok, q[:, 2], 1)[:, None]
            # 카메라 뒤 점에서 끊고, 화면에서 아주 먼 점은 잘라 오버플로 방지
            for run in np.split(np.arange(len(pts)), np.where(~ok)[0]):
                run = run[ok[run]]
                if len(run) > 1:
                    p = np.clip(pts[run], -4 * w, 4 * w)
                    cv2.polylines(canvas, [np.round(p).astype(np.int32)], False, 255, 1)
        return canvas

    def _f1(self, G: np.ndarray, mask: np.ndarray, dist: np.ndarray, valid: np.ndarray) -> Tuple[float, float, float]:
        hit, vis = self._stage1(G[None], dist, valid)
        precision = float(hit[0] / vis[0]) if vis[0] else 0.0
        near = cv2.dilate(self._render(G, mask.shape), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        line_px = mask > 0
        recall = float((near[line_px] > 0).mean()) if line_px.any() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, precision, recall

    def _plausible(self, G: np.ndarray, shape) -> bool:
        """중계 카메라 기하: 화면 오른쪽 = x 증가, 화면 위 = y 증가(먼 쪽), 화면 폭이 경기장 5~200 m"""
        h, w = shape
        try:
            H = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return False
        pts = np.array([[w * 0.2, h * 0.75], [w * 0.8, h * 0.75], [w * 0.5, h * 0.95], [w * 0.5, h * 0.55]], float)
        q = pts @ H[:, :2].T + H[:, 2]
        if np.any(np.abs(q[:, 2]) < 1e-12):
            return False
        # 화면 아래쪽 점들은 카메라 앞(지면 위)이어야: G 로 되돌렸을 때 동차 좌표 부호가 같아야 함
        P = q[:, :2] / q[:, 2:3]
        back = P @ G[:, :2].T + G[:, 2]
        if np.any(back[:, 2] <= 0):
            return False
        left, right, near, far = P
        span = np.linalg.norm(right - left) / 0.6
        return right[0] > left[0] and far[1] > near[1] and 5 <= span <= 200

    # -- 메인 -------------------------------------------------------------
    def calibrate(self, mask: np.ndarray, valid: np.ndarray, top_k: int = 120, refine_k: int = 8) -> AutoResult:
        h, w = mask.shape
        if (mask > 0).sum() < 150:
            return AutoResult(None, 0, 0, 0, 1, 0, 0, '라인 픽셀 부족')
        lines = detect_lines(mask)
        rows = np.where((valid > 0).any(axis=1))[0]
        top = float(rows[0]) if len(rows) else 0.0      # 경기장 영역의 가장 윗줄 (그 위는 관중석·하늘)
        src, dst = [], []
        horiz = [L for L in lines if abs(L.angle) <= 40]
        for hp in combinations(horiz, 2):
            # 위쪽 직선(화면 중앙 x 에서 y 가 작은 쪽) = 먼 쪽
            upper, lower = sorted(hp, key=lambda L: L.at_x(w / 2))
            if not _apart(upper, lower, w, h, top):
                continue
            rest = [L for L in lines if L is not upper and L is not lower]
            for vp in combinations(rest, 2):
                left, right = sorted(vp, key=lambda L: L.at_y(h / 2))
                if not _apart(left, right, w, h, top) or abs(left.angle) < 3 and abs(right.angle) < 3:
                    continue
                corners = [_intersect(a, b) for a in (upper, lower) for b in (left, right)]
                if any(c is None or not np.all(np.abs(c) < 20 * w) for c in corners):
                    continue
                img = np.array(corners)             # (upper-left, upper-right, lower-left, lower-right)
                for y_hi, y_lo in combinations(Y_LINES, 2):        # Y_LINES 는 내림차순 → y_hi > y_lo
                    for x_l, x_r in combinations(X_LINES, 2):      # X_LINES 는 오름차순 → x_l < x_r
                        src.append([[x_l, y_hi], [x_r, y_hi], [x_l, y_lo], [x_r, y_lo]])
                        dst.append(img)
        dist = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
        cands: List[Tuple[float, np.ndarray]] = []
        n_hyp = 0
        if src:
            G = _batch_homography(np.array(src, float), np.array(dst, float))
            G = G[np.all(np.isfinite(G.reshape(len(G), -1)), axis=1)]
            n_hyp = len(G)
            hits, vis = np.zeros(n_hyp), np.zeros(n_hyp)
            for s0 in range(0, n_hyp, 2048):
                hits[s0:s0 + 2048], vis[s0:s0 + 2048] = self._stage1(G[s0:s0 + 2048], dist, valid)
            prec = np.divide(hits, vis, out=np.zeros_like(hits), where=vis > 0)
            score = np.where(vis >= 40, hits * prec, -1)
            # 거의 같은 해석(화면 아래쪽 세 점의 경기장 위치가 1.5 m 이내)은 하나만 → 다양한 해석을 2 단계로
            seen = []
            for i in np.argsort(-score):
                if score[i] <= 0 or len(cands) >= top_k:
                    break
                if not self._plausible(G[i], (h, w)):
                    continue
                key = _anchor(G[i], w, h)
                if any(np.linalg.norm(key - k) < 1.5 for k in seen):
                    continue
                seen.append(key)
                cands.append((self._f1(G[i], mask, dist, valid)[0], G[i] / G[i][2, 2]))
        # 센터서클 장면: 하프라인 + 센터서클(타원) + 터치라인
        for Gc in circle_hypotheses(mask, lines, w, h):
            n_hyp += 1
            if self._plausible(Gc, (h, w)):
                cands.append((self._f1(Gc, mask, dist, valid)[0], Gc / Gc[2, 2]))
        if not cands:
            reason = '직선 조합 부족' if n_hyp == 0 else '그럴듯한 가설 없음'
            return AutoResult(None, 0, 0, 0, 1, len(lines), n_hyp, reason)
        cands.sort(key=lambda t: -t[0])

        # 상위 가설 정렬(ICP) 후 재채점
        S = scale_matrix(self.scale)
        Si = np.linalg.inv(S)
        refined = []
        for f1, Gi in cands[:refine_k]:
            r = self.refiner.refine(np.linalg.inv(Si @ Gi), mask, valid)
            Gr = S @ np.linalg.inv(r.H) if r.accepted else Gi
            if not self._plausible(Gr, (h, w)):
                Gr = Gi
            f1r, p, rc = self._f1(Gr, mask, dist, valid)
            refined.append((f1r, p, rc, Gr))
        refined.sort(key=lambda t: -t[0])
        best = refined[0]

        # 확신도: 화면 아래쪽 위치가 4 m 이상 다른 해석 중 최고 F1 과 비교
        c0 = _anchor(best[3], w, h)
        far = lambda Gm: np.max(np.abs(_anchor(Gm, w, h) - c0)) > 4.0
        others = [t[0] for t in refined[1:] if far(t[3])] + [f for f, Gm in cands[refine_k:refine_k + 40] if far(Gm)]
        margin = max(others) / best[0] if others and best[0] > 0 else 0.0
        H_best = np.linalg.inv(Si @ best[3])
        H_best /= H_best[2, 2]
        ok = best[0] >= self.min_f1 and best[2] >= self.min_recall and margin <= self.max_margin
        reason = '' if ok else ('일치도 부족' if best[0] < self.min_f1 or best[2] < self.min_recall else '해석이 애매함')
        return AutoResult(H_best if ok else None, round(best[0], 3), round(best[1], 3), round(best[2], 3),
                          round(margin, 3), len(lines), n_hyp, reason)


# ---------------------------------------------------------------------------
# 센터서클 가설 (원뿔 곡선 기하)
# ---------------------------------------------------------------------------
def _ellipse_conic(ellipse) -> np.ndarray:
    """cv2.fitEllipse 결과 → 3x3 원뿔 행렬 C (점 p 가 위에 있으면 pᵀ C p = 0)"""
    (cx, cy), (W, H), ang = ellipse
    a, b = W / 2, H / 2
    t = np.radians(ang)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    D = np.diag([1 / a ** 2, 1 / b ** 2])
    A = R @ D @ R.T
    c = np.array([cx, cy])
    C = np.zeros((3, 3))
    C[:2, :2] = A
    C[:2, 2] = C[2, :2] = -A @ c
    C[2, 2] = c @ A @ c - 1
    return C


def _line_conic(l: np.ndarray, C: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """동차 직선 l 과 원뿔 C 의 두 교점 (없으면 None)"""
    # 직선 위 두 점 p0 + s·d 를 잡고 2 차 방정식
    a, b, c = l
    nrm = np.hypot(a, b)
    if nrm < 1e-12:
        return None
    p0 = np.array([-a * c, -b * c, nrm ** 2]) / nrm ** 2
    d = np.array([-b, a, 0.0]) / nrm
    A = d @ C @ d
    B = 2 * (p0 @ C @ d)
    Cc = p0 @ C @ p0
    disc = B * B - 4 * A * Cc
    if abs(A) < 1e-15 or disc <= 0:
        return None
    r = np.sqrt(disc)
    s1, s2 = (-B - r) / (2 * A), (-B + r) / (2 * A)
    pts = [p0 + s * d for s in (s1, s2)]
    return tuple(p[:2] / p[2] for p in pts)


def _ellipses(mask: np.ndarray, lines: Sequence[ImageLine], min_pixels: int = 60):
    """직선 픽셀을 지운 나머지(곡선)에서 타원 후보를 찾음 — 하프라인에 둘로 갈린 원도 조각 쌍으로 맞춤"""
    curve = mask.copy()
    ys, xs = np.nonzero(curve)
    pix = np.column_stack([xs, ys]).astype(float)
    for L in lines:
        on = np.abs(pix @ L.n - L.c) < 2.5
        curve[ys[on], xs[on]] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(curve, np.ones((5, 5), np.uint8)), connectivity=8)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_pixels:
            continue
        cy, cx = np.nonzero((labels == i) & (curve > 0))
        if len(cx) >= 20:
            comps.append(np.column_stack([cx, cy]).astype(np.float32))
    comps.sort(key=len, reverse=True)
    comps = comps[:6]
    all_pts = np.concatenate(comps) if comps else np.zeros((0, 2), np.float32)
    found = []
    groups = [[c] for c in comps] + [[a, b] for a, b in combinations(comps, 2)]
    for g in groups:
        pts = np.concatenate(g)
        if len(pts) < 30:
            continue
        e = cv2.fitEllipse(pts)
        (cx, cy), (W, H), _ = e
        if min(W, H) < 12 or max(W, H) < 60 or not np.isfinite([cx, cy, W, H]).all():
            continue
        # 곡선 픽셀 중 타원 위(1.5 px)에 있는 것 = 지지도
        C = _ellipse_conic(e)
        hom = np.column_stack([all_pts, np.ones(len(all_pts))])
        val = np.einsum('ni,ij,nj->n', hom, C, hom)          # ≈ (r² - 1)
        r = np.sqrt(np.clip(val + 1, 0, None))
        support = int((np.abs(r - 1) * min(W, H) / 2 < 1.5).sum())
        if support >= 80:
            found.append((support, e, C))
    found.sort(key=lambda t: -t[0])
    return found[:2]


def circle_hypotheses(mask: np.ndarray, lines: Sequence[ImageLine], w: int, h: int) -> List[np.ndarray]:
    """
    하프라인(x = 0) + 센터서클 + 터치라인(y = ±34)으로 호모그래피 가설 (경기장 → 작업 이미지)

    - 하프라인과 타원의 두 교점 = (0, ±9.15)  (화면 위쪽이 +, 먼 쪽)
    - 하프라인 위에서 (0, 9.15), (0, -9.15), 터치라인 교점 (0, ±34) 세 점의 1 차원 사영 대응으로
      원 중심 (0, 0) 의 화면 위치 c 를 구함 (원근 때문에 타원 중심과 다름)
    - 하프라인의 극점 = 길이 방향 소실점 V_x. 직선 c–V_x 와 타원의 두 교점 = (∓9.15, 0)
    """
    out = []
    halfway = [L for L in lines if abs(L.angle) >= 50]
    touch = [L for L in lines if abs(L.angle) <= 40]
    if not halfway or not touch:
        return out
    for _, ell, C in _ellipses(mask, lines):
        (ex, ey), (W, H), _ = ell
        for hl in halfway:
            if abs(hl.n @ [ex, ey] - hl.c) > 0.35 * max(W, H):
                continue                                   # 하프라인이 원을 지나지 않음
            l = np.array([hl.n[0], hl.n[1], -hl.c])
            ab = _line_conic(l, C)
            if ab is None:
                continue
            A, B = sorted(ab, key=lambda p: p[1])          # 위쪽 = (0, +9.15)
            d = (B - A) / max(np.linalg.norm(B - A), 1e-9)
            s_of = lambda p: float((p - A) @ d)
            Vx = np.linalg.solve(C, l)                     # 하프라인의 극점 (동차)
            for tl in touch:
                T = _intersect(hl, tl)
                if T is None:
                    continue
                sT = s_of(T)
                y_t = HALF_WIDTH if sT < 0 else -HALF_WIDTH   # 원 위쪽에서 만나면 먼 터치라인
                # 1 차원 사영 변환 s = (α·y + β) / (γ·y + 1) 을 세 대응으로 풀고 y = 0 → s_c
                ys_ = np.array([9.15, -9.15, y_t])
                ss = np.array([0.0, s_of(B), sT])
                M = np.column_stack([ys_, np.ones(3), -ss * ys_])
                try:
                    alpha, beta, gamma = np.linalg.solve(M, ss)
                except np.linalg.LinAlgError:
                    continue
                c = A + beta * d                           # y = 0 → s = β
                m = np.cross([c[0], c[1], 1.0], Vx)        # 원 중심을 지나는 길이 방향 직선 y = 0
                lr = _line_conic(m, C)
                if lr is None:
                    continue
                Lp, Rp = sorted(lr, key=lambda p: p[0])
                src = np.float32([[0, 9.15], [0, -9.15], [-9.15, 0], [9.15, 0]])
                dst = np.float32([A, B, Lp, Rp])
                G = cv2.getPerspectiveTransform(src, dst).astype(float)
                if np.all(np.isfinite(G)):
                    out.append(G)
    return out


def _apart(a: ImageLine, b: ImageLine, w: int, h: int, field_top: float = 0.0) -> bool:
    """
    경기장에서 평행한 두 라인의 화면상 교점은 소실점 — 지평선 너머(보이는 경기장 영역 위쪽)에 있어야 한다.
    교점이 화면 안의 경기장 영역(field_top 아래)에 있으면 평행 라인 쌍이 될 수 없다.
    """
    p = _intersect(a, b)
    if p is None:
        return True
    return not (0 <= p[0] <= w and field_top <= p[1] <= h)


def _anchor(G: np.ndarray, w: int, h: int) -> np.ndarray:
    """해석 비교용 대표값: 화면 세 점(아래 왼쪽·가운데·오른쪽)의 경기장 좌표"""
    H = np.linalg.inv(G)
    pts = np.array([[w * 0.25, h * 0.8, 1], [w * 0.5, h * 0.8, 1], [w * 0.75, h * 0.8, 1]]) @ H.T
    return (pts[:, :2] / pts[:, 2:3]).ravel()


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.extend(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.array(out)


# ---------------------------------------------------------------------------
# 영상 전체 자동 키프레임
# ---------------------------------------------------------------------------
def _consistent(H_a: np.ndarray, H_b: np.ndarray, chain_ab: Optional[np.ndarray], w: int, h: int,
                tol: float = 3.0) -> bool:
    """키프레임 a 의 보정을 카메라 움직임(chain_ab: a → b 픽셀)으로 b 에 옮긴 값과 b 의 보정이 맞는지"""
    if chain_ab is None:
        return True                                  # 장면이 달라 비교 불가 → 각자 유지
    H_pred = H_a @ np.linalg.inv(chain_ab)
    pts = np.array([[w * x, h * y] for x in (0.25, 0.5, 0.75) for y in (0.5, 0.8)], np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, H_pred).reshape(-1, 2)
    b = cv2.perspectiveTransform(pts, H_b).reshape(-1, 2)
    return float(np.median(np.linalg.norm(a - b, axis=1))) <= tol


def auto_keyframes(motion: List[Optional[list]], masks: Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]],
                   image_size: Tuple[int, int], fps: float = 30.0, every_s: float = 2.0, scale: float = 0.5,
                   calibrator: Optional[AutoCalibrator] = None,
                   progress: Callable[[int, int], None] = lambda d, t: None) -> List[Dict]:
    """
    일정 간격(every_s)과 장면 전환 직후 프레임에서 자동 보정 → 확신한 프레임만 키프레임으로

    Returns:
        [{frame, homography: [9], source: 'auto', score: {f1, precision, recall, margin}}, ...]
    """
    calibrator = calibrator or AutoCalibrator(scale=scale)
    n = len(motion)
    w, h = image_size
    step = max(1, int(round(every_s * fps)))
    cuts = [t for t in range(1, n) if motion[t] is None]
    frames = sorted(set(range(0, n, step)) | set(cuts) | {t + 3 for t in cuts if t + 3 < n})
    found = []
    for i, t in enumerate(frames):
        data = masks(t)
        if data is not None:
            r = calibrator.calibrate(*data)
            if r.H is not None:
                found.append({'frame': int(t), 'H': r.H,
                              'score': {'f1': r.f1, 'precision': r.precision, 'recall': r.recall, 'margin': r.margin}})
        progress(i + 1, len(frames))

    # 교차 검증: 같은 장면 구간의 이웃 자동 키프레임과 카메라 움직임으로 맞는지. 어긋나면 점수 낮은 쪽 제거
    Ms = [None if m is None else np.array(m, float).reshape(3, 3) for m in motion]

    def chain(a: int, b: int) -> Optional[np.ndarray]:
        C = np.eye(3)
        for t in range(a + 1, b + 1):
            if Ms[t] is None:
                return None
            C = Ms[t] @ C
        return C

    changed = True
    while changed and len(found) > 1:
        changed = False
        for k in range(len(found) - 1):
            a, b = found[k], found[k + 1]
            if not _consistent(a['H'], b['H'], chain(a['frame'], b['frame']), w, h):
                found.pop(k if a['score']['f1'] < b['score']['f1'] else k + 1)
                changed = True
                break
    return [{'frame': kf['frame'], 'homography': np.round(kf['H'], 10).ravel().tolist(), 'source': 'auto',
             'points': [], 'score': kf['score']} for kf in found]
