"""
정밀 분석 모드 오프라인 후처리

실시간 추적은 지금까지의 프레임만 보고 결정하지만, 업로드 영상 분석은 영상 전체를 이미 갖고 있다.
이 모듈은 전체 궤적을 보고 다시 판단해 정확도를 끌어올린다.

  - stitch_tracks : 가려짐 등으로 끊긴 트랙 조각을 카메라 움직임 보정 위치·속도·유니폼 색으로 이어 붙임 (ID 스위치 감소)
  - fill_gaps     : 같은 트랙의 짧은 공백을 카메라 움직임을 따라 보간 (선수·공)
  - clean_ball    : 궤적에서 튀는 공 오탐 제거
  - smooth_world  : 경기장 좌표를 등속 칼만 필터 + RTS 역방향 평활화 → 위치·속도 안정화
  - assign_goalkeepers : 'other' 중 골문 앞에 머무는(또는 탐지 모델이 골키퍼로 본) 트랙을 골키퍼로, 그 골문을 지키는 팀에 배정

카메라 움직임 체인: motion[t] = M_t (t-1 → t 프레임 픽셀), None 이면 장면 전환(구간 경계).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

KIND_PLAYER, KIND_BALL = 0, 1


# ---------------------------------------------------------------------------
# 카메라 움직임 체인 → 안정화 좌표
# ---------------------------------------------------------------------------
class MotionChain:
    """프레임 픽셀 좌표를 같은 구간(장면 전환 없는 구간) 첫 프레임 좌표로 옮겨 카메라 움직임을 제거"""

    def __init__(self, motion: List[Optional[list]]):
        n = len(motion)
        self.segment = np.zeros(n, int)
        self.C = [np.eye(3)] * n                   # 구간 시작 프레임 → t 프레임
        seg = 0
        for t in range(n):
            m = motion[t]
            if t == 0 or m is None:
                if t > 0:
                    seg += 1
                self.C[t] = np.eye(3)
            else:
                self.C[t] = np.array(m, float).reshape(3, 3) @ self.C[t - 1]
            self.segment[t] = seg
        self.Ci = [np.linalg.inv(c) for c in self.C]

    def same_segment(self, a: int, b: int) -> bool:
        return self.segment[a] == self.segment[b]

    def to_stable(self, t: int, pts: np.ndarray) -> np.ndarray:
        return _apply(self.Ci[t], pts)

    def from_stable(self, t: int, pts: np.ndarray) -> np.ndarray:
        return _apply(self.C[t], pts)

    def transfer(self, pts: np.ndarray, a: int, b: int) -> np.ndarray:
        """a 프레임 픽셀 → b 프레임 픽셀 (같은 구간이어야 함)"""
        return _apply(self.C[b] @ self.Ci[a], pts)


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    q = pts @ M[:2, :2].T + M[:2, 2]
    w = pts @ M[2, :2] + M[2, 2]
    return q / w[:, None]


def _tracks(frames: List[List[list]], kind: Optional[int] = None) -> Dict[int, List[Tuple[int, list]]]:
    out: Dict[int, List[Tuple[int, list]]] = {}
    for t, objs in enumerate(frames):
        for o in objs:
            if kind is None or o[1] == kind:
                out.setdefault(o[0], []).append((t, o))
    return out


def _foot(o) -> np.ndarray:
    return np.array([(o[2] + o[4]) / 2, o[5]], float)


# ---------------------------------------------------------------------------
# 트랙 이어 붙이기
# ---------------------------------------------------------------------------
def stitch_tracks(frames: List[List[list]], motion: List[Optional[list]], fps: float,
                  track_colors: Optional[Dict[int, List[np.ndarray]]] = None,
                  max_gap_s: float = 1.5, color_gate: float = 28.0) -> Dict[int, int]:
    """
    끊긴 선수 트랙 조각을 이어 붙여 같은 ID 로 통일 (frames·track_colors 를 제자리 수정)

    조각 A 가 끝나고 max_gap_s 이내에 시작한 조각 B 에 대해, 카메라 움직임을 제거한 좌표에서
    A 의 마지막 위치·속도로 예측한 위치와 B 의 시작 위치가 가깝고(선수 키 기준), 키가 비슷하며,
    유니폼 색이 비슷하면 연결 후보로 본다. 전체 후보를 헝가리안 알고리즘으로 1:1 배정한다.

    Returns:
        {병합된 ID: 대표 ID}
    """
    from scipy.optimize import linear_sum_assignment

    chain = MotionChain(motion)
    tracks = _tracks(frames, KIND_PLAYER)
    info = {}
    for tid, items in tracks.items():
        ts = [t for t, _ in items]
        feet = np.array([chain.to_stable(t, _foot(o))[0] for t, o in items])
        heights = np.array([o[5] - o[3] for _, o in items], float)
        k = min(len(items), 8)
        span = max(ts[-1] - ts[-k], 1), max(ts[k - 1] - ts[0], 1)
        info[tid] = {
            'start': ts[0], 'end': ts[-1],
            'p_start': feet[0], 'p_end': feet[-1],
            'v_end': (feet[-1] - feet[-k]) / span[0], 'v_start': (feet[k - 1] - feet[0]) / span[1],
            'h_start': float(np.median(heights[:k])), 'h_end': float(np.median(heights[-k:])),
            'color': (np.median(track_colors[tid], axis=0) if track_colors and len(track_colors.get(tid, [])) >= 2
                      else None),
        }

    ids = sorted(info, key=lambda i: info[i]['start'])
    max_gap = max(1, int(round(max_gap_s * fps)))
    enders = [a for a in ids]
    starters = [b for b in ids]
    INF = 1e6
    cost = np.full((len(enders), len(starters)), INF)
    for i, a in enumerate(enders):
        A = info[a]
        for j, b in enumerate(starters):
            B = info[b]
            gap = B['start'] - A['end']
            if a == b or gap <= 0 or gap > max_gap or not chain.same_segment(A['end'], B['start']):
                continue
            h = 0.5 * (A['h_end'] + B['h_start'])
            if h <= 0 or not 0.67 <= B['h_start'] / max(A['h_end'], 1e-6) <= 1.5:
                continue
            pred_fwd = A['p_end'] + A['v_end'] * gap
            pred_bwd = B['p_start'] - B['v_start'] * gap
            d = min(np.linalg.norm(pred_fwd - B['p_start']), np.linalg.norm(pred_bwd - A['p_end']))
            if d > h * (0.6 + 1.2 * gap / fps):          # 공백이 길수록 예측 불확실성 증가
                continue
            cd = 0.0
            if A['color'] is not None and B['color'] is not None:
                cd = float(np.linalg.norm(A['color'] - B['color']))
                if cd > color_gate:
                    continue
            cost[i, j] = d / h + cd / 40.0 + 0.3 * gap / fps
    rows, cols = linear_sum_assignment(cost)
    links = {enders[i]: starters[j] for i, j in zip(rows, cols) if cost[i, j] < INF}

    # 사슬(A→B→C)을 대표 ID(가장 먼저 시작한 조각)로 정리
    parent = {}
    for a in ids:
        if a in parent:
            continue
        root, cur = a, a
        while cur in links:
            cur = links[cur]
            parent[cur] = root
    if not parent:
        return {}
    for objs in frames:
        for o in objs:
            if o[1] == KIND_PLAYER and o[0] in parent:
                o[0] = parent[o[0]]
    if track_colors is not None:
        for child, root in parent.items():
            track_colors.setdefault(root, []).extend(track_colors.pop(child, []))
    return parent


# ---------------------------------------------------------------------------
# 공백 보간 · 공 정리
# ---------------------------------------------------------------------------
def fill_gaps(frames: List[List[list]], motion: List[Optional[list]], max_gap: int,
              kind: int = KIND_PLAYER) -> int:
    """
    같은 트랙의 짧은 공백(max_gap 프레임 이하)을 채움. 위치는 양 끝 박스를 카메라 움직임으로
    해당 프레임에 옮긴 뒤 시간 가중 평균, 크기는 선형 보간. 보간 박스의 신뢰도는 0 으로 표시.

    Returns:
        채운 박스 수
    """
    chain = MotionChain(motion)
    added = 0
    for tid, items in _tracks(frames, kind).items():
        for (t1, a), (t2, b) in zip(items[:-1], items[1:]):
            gap = t2 - t1
            if gap <= 1 or gap > max_gap or not chain.same_segment(t1, t2):
                continue
            fa, fb = _foot(a), _foot(b)
            wa, ha, wb, hb = a[4] - a[2], a[5] - a[3], b[4] - b[2], b[5] - b[3]
            for t in range(t1 + 1, t2):
                al = (t - t1) / gap
                p = (1 - al) * chain.transfer(fa, t1, t)[0] + al * chain.transfer(fb, t2, t)[0]
                w, h = (1 - al) * wa + al * wb, (1 - al) * ha + al * hb
                frames[t].append([tid, kind, int(round(p[0] - w / 2)), int(round(p[1] - h)),
                                  int(round(p[0] + w / 2)), int(round(p[1])), 0.0])
                added += 1
    return added


def clean_ball(frames: List[List[list]], motion: List[Optional[list]], fps: float,
               max_speed_px: float = 60.0) -> int:
    """
    공 궤적에서 튀는 점 제거: 카메라 움직임을 제거한 좌표에서, 앞뒤 공 위치를 잇는 선에서
    크게 벗어나면서 앞뒤 두 점은 서로 가까운 경우(혼자 튄 오탐) 삭제

    Returns:
        삭제한 공 박스 수
    """
    chain = MotionChain(motion)
    items = _tracks(frames, KIND_BALL).get(0, [])
    if len(items) < 3:
        return 0
    pts = np.array([chain.to_stable(t, _foot(o))[0] for t, o in items])
    ts = np.array([t for t, _ in items])
    drop = []
    for i in range(1, len(items) - 1):
        t0, t1, t2 = ts[i - 1], ts[i], ts[i + 1]
        if not (chain.same_segment(t0, t1) and chain.same_segment(t1, t2)):
            continue
        al = (t1 - t0) / max(t2 - t0, 1)
        expect = (1 - al) * pts[i - 1] + al * pts[i + 1]
        dev = np.linalg.norm(pts[i] - expect)
        span = np.linalg.norm(pts[i + 1] - pts[i - 1])
        if dev > max_speed_px * max(1, min(t1 - t0, t2 - t1)) * 0.5 and dev > 2.5 * span:
            drop.append(items[i])
    for t, o in drop:
        frames[t].remove(o)
    return len(drop)


# ---------------------------------------------------------------------------
# 경기장 좌표 평활화 (칼만 + RTS)
# ---------------------------------------------------------------------------
def rts_smooth(ts: np.ndarray, z: np.ndarray, dt: float, accel_std: float, meas_std
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    등속 모델 칼만 필터 + Rauch-Tung-Striebel 역방향 평활화

    Args:
        ts: 측정 프레임 번호 (증가, 공백 허용), z: (N, 2) 측정 위치
        meas_std: 측정 잡음 (m) — 스칼라 또는 측정마다 다른 값 (N,)
    Returns:
        (평활 위치 (N, 2), 평활 속도 (N, 2)) — 측정 프레임에서의 값
    """
    n = len(ts)
    I4 = np.eye(4)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    stds = np.broadcast_to(np.asarray(meas_std, float), (n,))
    xs_f, Ps_f, xs_p, Ps_p, Fs = [], [], [], [], []
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P = np.diag([stds[0] ** 2, stds[0] ** 2, 25.0, 25.0])
    for i in range(n):
        if i > 0:
            h = (ts[i] - ts[i - 1]) * dt
            F = I4.copy()
            F[0, 2] = F[1, 3] = h
            q = accel_std ** 2
            Q1 = q * np.array([[h ** 4 / 4, h ** 3 / 2], [h ** 3 / 2, h ** 2]])
            Q = np.zeros((4, 4))
            Q[np.ix_([0, 2], [0, 2])] = Q1
            Q[np.ix_([1, 3], [1, 3])] = Q1
            x = F @ x
            P = F @ P @ F.T + Q
        else:
            F = I4
        xs_p.append(x.copy())
        Ps_p.append(P.copy())
        Fs.append(F)
        y = z[i] - Hm @ x
        S = Hm @ P @ Hm.T + np.eye(2) * stds[i] ** 2
        K = P @ Hm.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I4 - K @ Hm) @ P
        xs_f.append(x.copy())
        Ps_f.append(P.copy())
    xs = [None] * n
    xs[-1] = xs_f[-1]
    Ps = Ps_f[-1]
    for i in range(n - 2, -1, -1):
        F = Fs[i + 1]
        G = Ps_f[i] @ F.T @ np.linalg.inv(Ps_p[i + 1])
        xs[i] = xs_f[i] + G @ (xs[i + 1] - xs_p[i + 1])
        Ps = Ps_f[i] + G @ (Ps - Ps_p[i + 1]) @ G.T
    X = np.array(xs)
    return X[:, :2], X[:, 2:]


def conf_noise(conf: Optional[float], full: float = 0.7, worst: float = 30.0) -> float:
    """보정 신뢰도 → 측정 잡음 배율 (full 이상 1 배, 0 이면 worst 배)"""
    if conf is None:
        return 1.0
    c = min(max(conf / full, 0.0), 1.0)
    return float(worst ** (1.0 - c))


SPEED_FILTER_DEFAULTS = {
    'enabled': True,
    'max_speed': 11.0,      # m/s, 사람이 낼 수 있는 속력 상한 (축구 선수 최고 약 10~10.5 m/s)
    'window_s': 0.2,        # 점프 앞뒤 위치를 대표할 중앙값 창 길이
    'margin': 1.5,          # m, 창 중앙값에 남는 측정 잡음 여유 (정상 이동은 99 % 가 0.5 m 이하)
    'outlier_noise': 30.0,  # 이상 구간 측정 잡음 배율 → 사실상 결측으로 보고 RTS 가 앞뒤로 메움
    'max_excursion_s': 1.0, # 이 시간 안에 점프 전 궤적으로 돌아오면 끊지 않고 그 사이를 이상값으로
    'max_accel': 8.0,       # m/s², '돌아왔는지' 판정 때 점프 전 등속 예측에 주는 가속 여유
}


def load_speed_filter(path: Optional[str] = None) -> Dict:
    """configs/model_config.yaml 의 speed_filter 블록 (없으면 기본값)"""
    import os
    cfg = dict(SPEED_FILTER_DEFAULTS)
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'configs', 'model_config.yaml')
    try:
        import yaml
        with open(path, encoding='utf-8') as f:
            cfg.update((yaml.safe_load(f) or {}).get('speed_filter') or {})
    except (OSError, ImportError, AttributeError):
        pass
    return cfg


def find_jumps(ts: np.ndarray, z: np.ndarray, fps: float, max_speed: float = 11.0, window_s: float = 0.2,
               margin: float = 1.5, max_excursion_s: float = 1.0, max_accel: float = 8.0,
               **_) -> Tuple[np.ndarray, List[int]]:
    """
    경기장 좌표에서 물리적으로 불가능한 위치 점프 찾기 (박스가 가려진 선수 뒤로 미끄러짐, ID 교체, 잘못 이은 트랙)

    인덱스 i 마다 앞 창 [i-k, i) 와 뒤 창 [i, i+k) 의 중앙값 위치를 비교해, 두 창 사이 시간에 max_speed 로
    갈 수 있는 거리보다 margin 이상 멀면 점프 구간으로 본다. 프레임마다의 발 위치 흔들림(멀리 있는 선수는
    프레임당 2~3 m)은 중앙값이 걸러 주므로, 한두 프레임의 잡음이 아니라 위치가 '옮겨 간' 경우만 잡힌다.

    max_excursion_s 안에 위치가 점프 전 궤적으로 돌아오면(잠깐 다른 사람·몸통으로 튄 경우) 그 사이 전체를
    이상값으로, 돌아오지 않으면(다른 사람으로 넘어감) 점프 구간을 이상값으로 두고 가장 크게 옮겨 간 곳에서 끊는다.

    Returns:
        (이상값 마스크 (N,), 끊을 인덱스 목록 — 그 인덱스부터 새 구간)
    """
    n = len(ts)
    bad = np.zeros(n, bool)
    cuts: List[int] = []
    k = max(3, int(round(window_s * fps)))
    if n < 6:
        return bad, cuts

    def gap(a0, a1, b0, b1):
        """앞 창 [a0,a1) 과 뒤 창 [b0,b1) 중앙값 거리 − max_speed 로 갈 수 있는 거리"""
        d = np.linalg.norm(np.median(z[b0:b1], axis=0) - np.median(z[a0:a1], axis=0))
        return d - max_speed * (np.median(ts[b0:b1]) - np.median(ts[a0:a1])) / fps

    excess = np.full(n, -np.inf)
    for i in range(3, n - 2):
        excess[i] = gap(max(0, i - k), i, i, min(n, i + k))
    def runs(mask):
        """연속으로 True 인 인덱스 구간 [i, j] 목록"""
        out, i = [], 0
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                out.append((i, j))
                i = j + 1
            else:
                i += 1
        return out

    bands = runs(excess > margin)                  # 점프로 보는 구간
    # 돌아오는 점프 후보는 절반 기준으로: 갈 때만 잡히고 돌아올 때 잡음으로 기준을 살짝 못 넘으면
    # 튄 구간이 끊긴 조각으로 남으므로 (돌아왔는지는 returns() 의 궤적 예측이 따로 확인)
    weak_ends = [j for _, j in runs(excess > 0.5 * margin)]

    horizon = max_excursion_s * fps
    kp = max(k, int(round(0.5 * fps)))

    def returns(i, j2):
        """점프 구간 [i, j2] 뒤 창 위치가 점프 전 궤적(직전 0.5 초 등속 직선)의 예측과 맞는지.
        단순히 'max_speed 로 갈 수 있는 거리'로 보면 구간이 길수록 여유가 커져 옮겨 간 위치도 통과하므로,
        예측 위치에서 margin + 가속 여유(½·max_accel·Δt²) 안에 들어와야 돌아온 것으로 본다"""
        pre = np.arange(max(0, i - kp), i)
        pre = pre[~bad[pre]]
        post = slice(j2 + 1, min(n, j2 + 1 + k))
        if len(pre) < 3 or post.stop - post.start < 3:
            return False
        tp = ts[pre].astype(float)
        A = np.c_[tp - tp[-1], np.ones(len(tp))]
        coef = np.linalg.lstsq(A, z[pre], rcond=None)[0]         # (속도/프레임, 위치) x·y
        tc = float(np.median(ts[post]))
        pred = coef[1] + coef[0] * (tc - tp[-1])
        dt_s = (tc - tp[-1]) / fps
        return np.linalg.norm(np.median(z[post], axis=0) - pred) <= margin + 0.5 * max_accel * dt_s ** 2

    b = 0
    while b < len(bands):
        i, j = bands[b]
        # max_excursion_s 안의 (약한) 점프 구간 끝 중 그 뒤 위치가 점프 전 궤적으로 돌아온 가장 먼 것 → 그 사이 전체가 튄 구간
        back = None
        for e in weak_ends:
            if e < j:
                continue
            if ts[e] - ts[i] > horizon:
                break
            if returns(i, e):
                back = e
        if back is not None:
            bad[i:back + 1] = True
            while b < len(bands) and bands[b][0] <= back:
                b += 1
            continue
        # 돌아오지 않음 → 다른 사람으로 넘어감: 점프 구간을 이상값으로 두고 가장 크게 옮겨 간 곳에서 끊음
        bad[i:j + 1] = True
        if i < 3 or n - (j + 1) < 3:
            cuts.append(i if i >= 3 else j + 1)   # 끝자락 점프: 비교할 창이 모자라면 경계에서 끊음
        else:
            cuts.append(int(np.argmax(excess[i:j + 1])) + i)
        b += 1
    cuts = sorted({c for c in cuts if 0 < c < n})
    return bad, cuts


def smooth_world(world: List[Optional[List[list]]], fps: float, ball_id: int = 0,
                 player_accel: float = 5.0, ball_accel: float = 40.0,
                 meas_std: float = 0.4, max_gap_s: float = 1.0,
                 frame_conf: Optional[List[Optional[float]]] = None,
                 speed_filter=None, report: Optional[Dict] = None) -> List[Optional[List[list]]]:
    """
    world[t] = [[id, x, y], ...] → [[id, x, y, speed(m/s)], ...] (평활화된 위치와 속력)

    트랙마다 max_gap_s 보다 긴 공백에서 구간을 나눠 따로 평활화한다. 보정 안 된 프레임(None)은 그대로 둔다.
    frame_conf: 프레임별 보정 신뢰도(0~1). 신뢰도가 낮은 프레임의 측정은 잡음을 크게 잡아(최대 30 배)
                필터가 앞뒤 믿을 만한 프레임으로 보간하듯 따라가게 한다 → 보정 오차가 속력으로 번지지 않음
    speed_filter: 선수 위치 점프 처리 (find_jumps). None 이면 설정 파일의 speed_filter 블록, False 면 끔, dict 면 그 값.
                  점프 구간은 측정 잡음을 outlier_noise 배로 키워 결측처럼 다루고, 다른 사람으로 넘어간 점프는
                  궤적을 끊는다 → 등속 평활화가 순간 이동을 '빠른 달리기'로 퍼뜨리지 않음 (11부)
    report: dict 를 주면 {'jumps', 'cuts', 'outlier_points', 'dropped_points', 'tracks'} 처리 요약을 채운다
    """
    if speed_filter is None:
        speed_filter = load_speed_filter()
    elif speed_filter is True:
        speed_filter = dict(SPEED_FILTER_DEFAULTS)
    if not speed_filter or not speed_filter.get('enabled', True):
        speed_filter = None
    summary = {'jumps': 0, 'cuts': 0, 'outlier_points': 0, 'dropped_points': 0, 'tracks': []}
    dt = 1.0 / (fps or 30.0)
    series: Dict[int, List[Tuple[int, float, float]]] = {}
    for t, entries in enumerate(world):
        for e in entries or []:
            series.setdefault(int(e[0]), []).append((t, float(e[1]), float(e[2])))
    out = [None if w is None else [] for w in world]
    max_gap = max(1, int(round(max_gap_s * fps)))
    for tid, s in series.items():
        arr = np.array(s)
        ts = arr[:, 0].astype(int)
        cuts = np.where(np.diff(ts) > max_gap)[0] + 1
        for gseg in np.split(np.arange(len(ts)), cuts):
            std_all = np.full(len(gseg), meas_std, float)
            if frame_conf is not None:
                std_all = np.array([meas_std * conf_noise(frame_conf[t] if t < len(frame_conf) else None)
                                    for t in ts[gseg]])
            pieces = [np.arange(len(gseg))]
            if speed_filter is not None and tid != ball_id:
                bad, jcuts = find_jumps(ts[gseg], arr[gseg, 1:], fps, **speed_filter)
                if bad.any():
                    std_all[bad] *= float(speed_filter.get('outlier_noise', 30.0))
                    # 끊은 구간의 앞뒤 끝에 걸린 이상값은 양쪽에서 잡아 줄 측정이 없어 필터가 직전 속도로
                    # 내달리게 되므로 내보내지 않는다 (화면 가장자리에 잘린 박스를 빼는 것과 같은 처리)
                    # 점프로 끊긴 조각이 0.4 초(창 2 개)도 안 되면 그 위치 자체를 믿을 수 없다
                    # (잠깐 다른 사람에게 갔다 온 구간인데 이력이 짧아 '돌아옴'으로 확인 못 한 경우) → 버림
                    min_piece = 2 * max(3, int(round(float(speed_filter.get('window_s', 0.2)) * fps)))
                    pieces = []
                    for piece in np.split(np.arange(len(gseg)), jcuts):
                        good = np.where(~bad[piece])[0]
                        if len(good) and (not jcuts or good[-1] - good[0] + 1 >= min_piece):
                            pieces.append(piece[good[0]:good[-1] + 1])
                    summary['dropped_points'] += int(
                        len(gseg) - sum(len(p) for p in pieces))
                    n_jumps = int(np.sum(np.diff(np.r_[0, bad.astype(int)]) == 1))
                    summary['jumps'] += n_jumps
                    summary['cuts'] += len(jcuts)
                    summary['outlier_points'] += int(bad.sum())
                    summary['tracks'].append({'id': tid, 'frames': [int(ts[gseg][i]) for i in np.where(bad)[0]],
                                              'cuts': [int(ts[gseg][c]) for c in jcuts]})
            for piece in pieces:
                seg = gseg[piece]
                if len(seg) == 1:
                    i = seg[0]
                    out[ts[i]].append([tid, round(arr[i, 1], 2), round(arr[i, 2], 2), 0.0])
                    continue
                pos, vel = rts_smooth(ts[seg], arr[seg, 1:], dt, ball_accel if tid == ball_id else player_accel,
                                      std_all[piece])
                for i, p, v in zip(seg, pos, vel):
                    out[ts[i]].append([tid, round(float(p[0]), 2), round(float(p[1]), 2),
                                       round(float(np.hypot(*v)), 2)])
    if report is not None:
        report.update(summary)
    return out


# ---------------------------------------------------------------------------
# 골키퍼
# ---------------------------------------------------------------------------
def assign_goalkeepers(world: List[Optional[List[list]]], teams: Dict[str, str], fps: float,
                       labels: Optional[Dict[str, str]] = None, min_seconds: float = 1.0) -> Dict[str, str]:
    """
    'other' 트랙 중 대부분의 시간을 한쪽 페널티 박스 안에서 보낸 트랙을 골키퍼로 판정하고,
    그 골문 쪽에 평균 위치(무게중심)가 더 가까운 팀에 배정 (roboflow/sports 방식).
    teams 를 제자리 수정하고 {track_id: 'goalkeeper' | 'referee'} 역할 표를 반환한다.

    labels: 트랙별 탐지 라벨 다수결 (축구 전용 탐지 모델). 'goalkeeper' 트랙은 페널티 박스 조건 없이
            경기장 어느 쪽 반에 있는지만으로 팀에 배정하고, 'referee' 트랙은 'other' 로 두고 역할만 기록한다.
    """
    labels = labels or {}
    pos: Dict[str, List[Tuple[float, float]]] = {}
    for entries in world:
        for e in entries or []:
            pos.setdefault(str(int(e[0])), []).append((e[1], e[2]))
    cent = {}
    for team in ('home', 'away'):
        xs = [x for tid, ps in pos.items() if teams.get(tid) == team for x, _ in ps]
        if xs:
            cent[team] = float(np.mean(xs))
    roles = {tid: 'referee' for tid, lab in labels.items() if lab == 'referee' and teams.get(tid, 'other') == 'other'}
    if len(cent) < 2:
        return roles
    for tid, ps in pos.items():
        if teams.get(tid) != 'other' or tid == '0' or tid in roles:
            continue
        detected_gk = labels.get(tid) == 'goalkeeper'
        if len(ps) < (0.3 if detected_gk else min_seconds) * fps:
            continue
        p = np.array(ps)
        side = np.sign(np.median(p[:, 0]))
        in_box = (side * p[:, 0] >= 52.5 - 16.5 - 2) & (np.abs(p[:, 1]) <= 20.16 + 2)
        if side == 0 or (not detected_gk and in_box.mean() < 0.7):
            continue
        goal_x = side * 52.5
        team = min(cent, key=lambda k: abs(cent[k] - goal_x))
        teams[tid] = team
        roles[tid] = 'goalkeeper'
    return roles
    for tid, ps in pos.items():
        if teams.get(tid) != 'other' or tid == '0' or len(ps) < min_seconds * fps:
            continue
        p = np.array(ps)
        side = np.sign(np.median(p[:, 0]))
        in_box = (side * p[:, 0] >= 52.5 - 16.5 - 2) & (np.abs(p[:, 1]) <= 20.16 + 2)
        if side == 0 or in_box.mean() < 0.7:
            continue
        goal_x = side * 52.5
        team = min(cent, key=lambda k: abs(cent[k] - goal_x))
        teams[tid] = team
        roles[tid] = 'goalkeeper'
    return roles
