"""
경기장 보정 추적 — 짧은 간격 앵커 재보정 + 양방향 광류 연결 + 시간 평활화 + 프레임별 신뢰도

기존 방식(가장 가까운 키프레임 하나에서 카메라 움직임을 곱해 전파)은 키프레임에서 멀어질수록 오차가 쌓이고,
두 키프레임의 담당 구간이 바뀌는 지점에서 호모그래피가 튀어 속력이 비현실적으로 커졌다.

  1. 앵커    : 키프레임(자동·수동)과, 그 사이 짧은 간격(기본 0.1 초) 격자 프레임을 이웃 앵커에서 예측한 뒤
               경기장 라인 정렬(LineRefiner)로 다시 맞춘 것 — 라인 일치도(F1)가 낮으면 앵커로 쓰지 않는다.
               광류 카메라 움직임은 프레임당 1 px 안팎의 오차가 있어 0.5 초만 이어도 라인 정렬의 포착 범위를
               벗어나므로, 추적은 촘촘히 한다 (자동 키프레임 탐색은 0.5 초 간격)
  2. 연결    : 앵커 사이 프레임은 왼쪽·오른쪽 앵커를 카메라 움직임으로 각각 옮겨 온 두 값을 거리 비율로 혼합
               (화면 격자점의 경기장 좌표를 섞어 다시 맞춤) → 누적 오차는 앵커에서 끊기고 담당 구간 경계의 점프도 없다
  3. 평활화  : 이웃 프레임(±3σ) 호모그래피를 카메라 움직임으로 현재 프레임에 옮겨 가우시안 가중 평균 → 앵커 떨림 제거.
               카메라 움직임 자체는 광류로 따라가므로 빠른 팬에도 늦지 않는다. σ 는 평소 0.05 초, 앵커끼리 어긋나
               보정이 미끄러지는 곳에서만 0.4 초까지 넓혀 어긋남을 시간에 나눠 푼다 (한 번에 풀면 속력이 튄다)
  4. 신뢰도  : 가까운 앵커의 신뢰도(거리에 따라 감소)와 그 프레임의 라인 일치도로 0~1 값을 프레임마다 기록.
               보정이 미끄러지는 프레임(경기장에 고정된 점이 1.5 m/s 이상으로 움직여 보임)은 더 낮춘다

좌표 규칙은 video_analysis 와 같다 (이미지 원본 픽셀 → 경기장 미터).
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pitch import LineRefiner, scale_matrix

MaskFn = Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------
def orient(H: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    부호·크기 정규화: 화면 아래 가운데 점의 동차 좌표가 1 이 되도록 나눈다.

    H / H[2,2] 는 화면 왼쪽 위(관중석)가 지평선 너머일 때 부호가 뒤집히거나 0 에 가까워 터진다.
    역행렬로 경기장 → 화면을 그릴 때 w > 0 을 '카메라 앞' 으로 판정하므로 부호가 일정해야 한다.
    """
    s = float(H[2] @ [width / 2, height * 0.9, 1.0])
    if abs(s) < 1e-12:
        return H / H[2, 2]
    return H / s


def line_conf(f1: float) -> float:
    """라인 일치도 F1 → 신뢰도 (0.35 이하 0, 0.75 이상 1)"""
    return float(np.clip((f1 - 0.35) / 0.4, 0.0, 1.0))


class MotionChain:
    """프레임 간 카메라 움직임(M_t: t-1 → t 픽셀)을 누적해 임의 두 프레임 사이 변환을 바로 구한다"""

    def __init__(self, motion: Sequence[Optional[list]]):
        n = len(motion)
        self.seg = np.zeros(n, int)
        self.P = np.zeros((n, 3, 3))
        seg, cur = -1, np.eye(3)
        for t in range(n):
            m = motion[t]
            if t == 0 or m is None:
                seg += 1
                cur = np.eye(3)
            else:
                cur = np.array(m, float).reshape(3, 3) @ cur
                cur = cur / cur[2, 2]
            self.seg[t] = seg
            self.P[t] = cur
        self.Pinv = np.linalg.inv(self.P) if n else self.P

    def same(self, a: int, b: int) -> bool:
        return self.seg[a] == self.seg[b]

    def transfer(self, H_a: np.ndarray, a: int, t: int) -> np.ndarray:
        """프레임 a 의 이미지 → 경기장 H 를 프레임 t 로 (t 픽셀 → a 픽셀 → 경기장)"""
        return H_a @ self.P[a] @ self.Pinv[t]

    def segments(self) -> List[Tuple[int, int]]:
        """장면 구간 [(시작, 끝 포함), ...]"""
        out = []
        n = len(self.seg)
        start = 0
        for t in range(1, n + 1):
            if t == n or self.seg[t] != self.seg[t - 1]:
                out.append((start, t - 1))
                start = t
        return out


class LineScorer:
    """호모그래피가 라인 마스크와 얼마나 맞는지 — (F1, 정밀도, 재현율, 보이는 모델 점 수)"""

    def __init__(self, scale: float):
        from .autocalib import AutoCalibrator
        self.scale = scale
        self.cal = AutoCalibrator(scale=scale)
        self.S = scale_matrix(scale)

    def __call__(self, H: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> Tuple[float, float, float, int]:
        import cv2
        h, w = mask.shape
        H = orient(H, w / self.scale, h / self.scale)
        try:
            G = self.S @ np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, 0
        dist = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
        _, vis = self.cal._stage1(G[None], dist, valid)
        f1, p, r = self.cal._f1(G, mask, dist, valid)
        return f1, p, r, int(vis[0])


def recover(H: np.ndarray, mask: np.ndarray, valid: np.ndarray, scorer: 'LineScorer', refiner: LineRefiner,
            image_size: Tuple[int, int], top: int = 4) -> Tuple[float, np.ndarray]:
    """
    추적이 어긋났을 때(라인 정렬 포착 범위 밖) 예측값 주변 가설 탐색으로 다시 잡기

    한쪽 라인(예: 센터서클)으로만 맞춘 보정을 멀리 외삽하면 반대쪽(예: 페널티 박스)이 수십 px 어긋난다.
    예측값에 경기장 축 방향 배율·이동 보정(x 배율 0.8~1.25, x 이동 ±10 m, y 배율·이동 조금)을 곱한 가설
    약 1,000 개를 라인 거리 변환으로 한꺼번에 채점하고, 상위 몇 개만 라인 정렬해 F1 이 가장 높은 것을 고른다.
    Returns: (F1, H)
    """
    import cv2
    w, h = image_size
    H = orient(H, w, h)
    pc = H @ [w / 2, h * 0.65, 1.0]
    pc = pc[:2] / pc[2]                                    # 화면 가운데 아래쪽의 경기장 위치를 중심으로 배율
    C = []
    for sx in np.linspace(0.8, 1.25, 10):
        for tx in np.linspace(-10, 10, 11):
            for sy in (0.93, 1.0, 1.07):
                for ty in (-2.0, 0.0, 2.0):
                    C.append([[sx, 0, pc[0] * (1 - sx) + tx], [0, sy, pc[1] * (1 - sy) + ty], [0, 0, 1]])
    C = np.array(C, float)
    G = np.einsum('ij,kjl->kil', scorer.S @ np.linalg.inv(H), np.linalg.inv(C))
    dist = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    hit, vis = scorer.cal._stage1(G, dist, valid)
    prec = np.divide(hit, vis, out=np.zeros(len(hit)), where=vis > 0)
    score = np.where(vis >= 40, hit * prec, -1)
    best = (scorer(H, mask, valid)[0], H)
    for i in np.argsort(-score)[:top]:
        Hc = C[i] @ H
        r = refiner.refine(Hc, mask, valid)
        Hn = r.H if r.accepted else Hc
        f1 = scorer(Hn, mask, valid)[0]
        if f1 > best[0]:
            best = (f1, Hn)
    return best


def slide_speed(Hs: Sequence[Optional[np.ndarray]], chain: MotionChain, image_size: Tuple[int, int],
                fps: float) -> np.ndarray:
    """
    보정 미끄러짐 속도 (m/s): 경기장에 고정된 점이 보정 변화 때문에 움직여 보이는 속도.
    프레임 t-1 의 보정을 카메라 움직임으로 t 에 옮긴 값과 t 의 보정이 화면 격자점에서 얼마나 다른지.
    여기서 생기는 속도는 모든 선수 속력에 그대로 더해진다.
    """
    w, h = image_size
    Q = np.array([[x * w, y * h, 1.0] for x in (0.2, 0.5, 0.8) for y in (0.6, 0.85)])
    v = np.zeros(len(Hs))
    for t in range(1, len(Hs)):
        if Hs[t] is None or Hs[t - 1] is None or not chain.same(t - 1, t):
            continue
        a = chain.transfer(Hs[t - 1], t - 1, t) @ Q.T
        b = Hs[t] @ Q.T
        if np.any(np.abs(a[2]) < 1e-12) or np.any(np.abs(b[2]) < 1e-12):
            continue
        v[t] = float(np.median(np.linalg.norm(a[:2] / a[2] - b[:2] / b[2], axis=0))) * fps
    return v


def _grid(width: float, height: float) -> np.ndarray:
    """혼합·평활화에 쓰는 화면 격자점 (경기장이 보이는 화면 아래쪽 위주)"""
    xs = np.linspace(0.05, 0.95, 5) * width
    ys = np.linspace(0.5, 0.97, 4) * height
    return np.array([[x, y, 1.0] for y in ys for x in xs])


def _fit(Q: np.ndarray, Hs: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    """
    같은 프레임에 대한 여러 호모그래피 추정값(K, 3, 3)을 가중 평균 → 하나의 호모그래피
    격자점 Q 의 경기장 좌표를 가중 평균해 다시 맞춘다 (카메라 뒤로 넘어가는 점은 제외)
    """
    import cv2
    q = np.einsum('kij,nj->kni', Hs, Q)                # (K, N, 3)
    q = q * np.sign(np.median(q[..., 2], axis=1))[:, None, None]   # 추정마다 부호를 맞춤 (경기장 쪽 w > 0)
    z = q[..., 2]
    ok = np.all(z > 1e-9, axis=0)                      # 모든 추정에서 카메라 앞인 격자점만
    if ok.sum() < 6:
        return None
    pts = q[:, ok, :2] / z[:, ok, None]
    wts = weights / weights.sum()
    mean = np.einsum('k,knd->nd', wts, pts)
    H, _ = cv2.findHomography(Q[ok, :2].astype(np.float64), mean.astype(np.float64), 0)
    return H


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def anchored_homographies(motion: Sequence[Optional[list]], keyframes: Dict[int, np.ndarray],
                          image_size: Tuple[int, int], fps: float = 30.0, masks: Optional[MaskFn] = None,
                          scale: float = 0.5, anchor_every_s: float = 0.1, smooth_sigma_s: float = 0.4,
                          smooth_min_s: float = 0.05,
                          refiner: Optional[LineRefiner] = None, keyframe_conf: Optional[Dict[int, float]] = None,
                          track_min_f1: float = 0.55, track_min_recall: float = 0.45, max_skip_s: float = 1.0,
                          decay_s: float = 1.5, refine_keyframes: bool = True, recover_below: float = 0.75,
                          slide_ok: float = 1.5, slide_scale: float = 3.0,
                          progress: Callable[[int, int], None] = lambda d, t: None
                          ) -> Tuple[List[Optional[np.ndarray]], List[Optional[float]], Dict]:
    """
    Args:
        keyframes: {프레임: 이미지 → 경기장 H}
        masks: t → (라인 마스크, 유효 영역) (scale 해상도). 없으면 앵커 추적·라인 신뢰도 없이 키프레임만으로 연결
        keyframe_conf: 자동 키프레임별 신뢰도. 여기 없는 키프레임은 사람이 찍은 기준점으로 보고(신뢰도 1.0)
                       추적값으로 바꾸지 않는다
    Returns:
        (프레임별 H, 프레임별 신뢰도 0~1 (보정 없음 None), 요약 {anchors, tracked, line_frames})
    """
    n = len(motion)
    w, h = image_size
    chain = MotionChain(motion)
    refiner = refiner or LineRefiner(scale=scale)
    scorer = LineScorer(scale) if masks is not None else None
    keyframe_conf = keyframe_conf or {}
    step = max(1, int(round(anchor_every_s * fps)))
    max_skip = max(1, int(round(max_skip_s * fps / step)))
    Q = _grid(w, h)

    # 1. 앵커: 키프레임 -------------------------------------------------------
    # anchors[t] = (H, 신뢰도, 출처 'manual' | 'key'(자동) | 'track', 라인 F1 또는 None)
    anchors: Dict[int, Tuple[np.ndarray, float, str, Optional[float]]] = {}
    for k, Hk in keyframes.items():
        if not 0 <= k < n:
            continue
        Hk = np.array(Hk, float)
        src = 'key' if k in keyframe_conf else 'manual'
        conf = float(keyframe_conf.get(k, 1.0))
        data = masks(k) if masks is not None else None
        f1 = None
        if data is not None and refine_keyframes:
            r = refiner.refine(Hk, *data)
            if r.accepted:
                Hk = r.H
            f1 = scorer(Hk, *data)[0]
            conf = max(conf * 0.8, min(conf, line_conf(f1)))
        anchors[k] = (orient(Hk, w, h), conf, src, f1)

    # 2. 앵커: 격자 프레임 추적 (예측 → 라인 정렬 → 어긋나면 가설 탐색 → 일치도 확인) ----------------
    tracked = recovered = replaced = 0
    if masks is not None and anchors:
        def track(base: int, g: int, data) -> Tuple[np.ndarray, float, float]:
            nonlocal recovered
            pred = chain.transfer(anchors[base][0], base, g)
            r = refiner.refine(pred, *data)
            Hg = r.H if r.accepted else pred
            f1, _, rec, _ = scorer(Hg, *data)
            if f1 < recover_below:                         # 어긋남 → 예측값 주변 가설 탐색으로 다시 잡기
                f1r, Hr = recover(Hg, *data, scorer, refiner, (w, h))
                if f1r > f1 + 0.03:
                    Hg = Hr
                    f1, _, rec, _ = scorer(Hg, *data)
                    recovered += 1
            return Hg, f1, rec

        def walk(start: int, direction: int, stop: int) -> int:
            """start 앵커에서 direction 쪽 격자 프레임을 차례로 추적 (stop 포함까지). 마지막 앵커 프레임 반환"""
            nonlocal tracked, replaced
            base, fails, last = start, 0, start
            g = (start // step + 1) * step if direction > 0 else ((start - 1) // step) * step  # 다음 격자 프레임
            if direction > 0 and start < stop < g or direction < 0 and g < stop < start:
                g = stop                                   # 다음 키프레임이 격자보다 가까움
            while (g <= stop if direction > 0 else g >= stop) and 0 <= g < n and chain.same(g, start):
                data = masks(g)
                if g in anchors:
                    # 이미 있는 자동 앵커: 이웃에서 추적해 온 값이 라인에 더 잘 맞으면 바꾼다 (틀린 자동 키프레임 보정)
                    Ha, ca, src, fa = anchors[g]
                    if src != 'manual' and data is not None and fa is not None:
                        Hg, f1, rec = track(base, g, data)
                        if f1 > fa + 0.03 and rec >= track_min_recall:
                            anchors[g] = (orient(Hg, w, h), max(ca, line_conf(f1)), src, f1)
                            replaced += 1
                    base, fails, last = g, 0, g
                elif data is not None:
                    Hg, f1, rec = track(base, g, data)
                    if f1 >= track_min_f1 and rec >= track_min_recall:
                        anchors[g] = (orient(Hg, w, h), line_conf(f1), 'track', f1)
                        tracked += 1
                        base, fails, last = g, 0, g
                    else:
                        fails += 1
                else:
                    fails += 1
                if fails > max_skip:
                    break
                nxt = g + direction * step
                if direction > 0 and g < stop < nxt or direction < 0 and nxt < stop < g:
                    nxt = stop
                g = nxt
            return last

        for s0, s1 in chain.segments():
            keys = sorted(k for k in anchors if s0 <= k <= s1 and anchors[k][2] != 'track')
            if not keys:
                continue
            walk(keys[0], -1, s0)
            for a, b in zip(keys[:-1], keys[1:]):
                last = walk(a, +1, b)
                if last < b:
                    walk(b, -1, last)
            walk(keys[-1], +1, s1)

    # 3. 연결: 앵커 사이 양방향 전파 + 혼합 ---------------------------------------
    H_raw: List[Optional[np.ndarray]] = [None] * n
    prox: List[Optional[float]] = [None] * n
    decay = max(1.0, decay_s * fps)
    for s0, s1 in chain.segments():
        keys = sorted(k for k in anchors if s0 <= k <= s1)
        if not keys:
            continue
        j = 0
        for t in range(s0, s1 + 1):
            while j + 1 < len(keys) and keys[j + 1] <= t:
                j += 1
            a = keys[j] if keys[j] <= t else None
            b = next((k for k in keys[j:] if k >= t), None)
            if a is None:
                a = b
            if b is None:
                b = a
            Ha = chain.transfer(anchors[a][0], a, t)
            ca = anchors[a][1] * np.exp(-abs(t - a) / decay)
            if a == b:
                H_raw[t], prox[t] = Ha, ca
                continue
            Hb = chain.transfer(anchors[b][0], b, t)
            cb = anchors[b][1] * np.exp(-abs(b - t) / decay)
            wb = (t - a) / (b - a)
            Ht = _fit(Q, np.stack([Ha, Hb]), np.array([1 - wb, wb]) + 1e-9)
            H_raw[t] = Ht if Ht is not None else (Ha if wb < 0.5 else Hb)
            # 두 앵커가 모두 믿을 만하면 사이 프레임도 믿을 만함 (가까운 쪽 기준 감소)
            prox[t] = max(ca, cb)

    # 4. 시간 평활화 (적응형) ---------------------------------------------------
    # 평소에는 좁게(smooth_min_s) — 라인 정렬된 앵커의 정밀도를 살림. 앵커끼리 어긋나 보정이 미끄러지는 곳
    # (경기장 고정점이 slide_ok m/s 이상으로 움직여 보임)만 넓게(smooth_sigma_s) 해 어긋남을 시간에 나눠 푼다
    for t in range(n):
        if H_raw[t] is not None:
            H_raw[t] = orient(H_raw[t], w, h)
    raw_slide = slide_speed(H_raw, chain, (w, h), fps)
    sig_max = smooth_sigma_s * fps
    sig_min = min(smooth_min_s * fps, sig_max)
    rad_max = int(np.ceil(sig_max))
    near_slide = np.array([raw_slide[max(0, t - rad_max):t + rad_max + 1].max() for t in range(n)]) if n else raw_slide
    Hs: List[Optional[np.ndarray]] = list(H_raw)
    for t in range(n):
        if H_raw[t] is None:
            continue
        sigma = sig_min + (sig_max - sig_min) * float(np.clip((near_slide[t] - slide_ok) / slide_scale, 0, 1))
        if sigma < 0.5:
            continue
        rad = int(np.ceil(3 * sigma))
        idx = [s for s in range(max(0, t - rad), min(n, t + rad + 1)) if H_raw[s] is not None and chain.same(s, t)]
        if len(idx) < 2:
            continue
        est = np.stack([chain.transfer(H_raw[s], s, t) for s in idx])
        wts = np.exp(-0.5 * ((np.array(idx) - t) / sigma) ** 2)
        wts *= np.array([max(prox[s], 0.05) for s in idx])
        Ht = _fit(Q, est, wts)
        if Ht is not None:
            Hs[t] = Ht

    # 5. 신뢰도 ---------------------------------------------------------------
    # 보정 미끄러짐이 큰 프레임(앞뒤 3 프레임 포함)은 신뢰도를 낮춘다 → 속력 평활화에서 가중치가 줄어 튀지 않음
    for t in range(n):
        if Hs[t] is not None:
            Hs[t] = orient(Hs[t], w, h)
    slide = slide_speed(Hs, chain, (w, h), fps)
    slide_near = np.array([slide[max(0, t - 3):t + 4].max() for t in range(n)]) if n else slide
    conf: List[Optional[float]] = [None] * n
    line_frames = 0
    for t in range(n):
        if Hs[t] is None:
            continue
        c = float(prox[t])
        data = masks(t) if masks is not None else None
        if data is not None and (data[0] > 0).sum() >= 150:
            f1, _, _, vis = scorer(Hs[t], *data)
            if vis >= 40:                              # 라인이 충분히 보이는 프레임만 일치도로 판정
                line_frames += 1
                c = 0.5 * c + 0.5 * line_conf(f1)
        c *= float(np.exp(-(max(slide_near[t] - slide_ok, 0.0) / slide_scale) ** 2))
        conf[t] = round(c, 3)
        progress(t + 1, n)
    info = {'anchors': len(anchors), 'tracked': tracked, 'recovered': recovered, 'replaced': replaced,
            'line_frames': line_frames,
            'anchor_every': step, 'max_slide': round(float(slide.max()), 2) if n else 0.0}
    return Hs, conf, info
