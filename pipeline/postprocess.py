"""
정밀 분석 모드 오프라인 후처리

실시간 추적은 지금까지의 프레임만 보고 결정하지만, 업로드 영상 분석은 영상 전체를 이미 갖고 있다.
이 모듈은 전체 궤적을 보고 다시 판단해 정확도를 끌어올린다.

  - stitch_tracks : 가려짐 등으로 끊긴 트랙 조각을 카메라 움직임 보정 위치·속도·유니폼 색으로 이어 붙임 (ID 스위치 감소)
  - fill_gaps     : 같은 트랙의 짧은 공백을 카메라 움직임을 따라 보간 (선수·공)
  - clean_ball    : 궤적에서 튀는 공 오탐 제거
  - smooth_world  : 경기장 좌표를 등속 칼만 필터 + RTS 역방향 평활화 → 위치·속도 안정화
  - assign_goalkeepers : 'other' 중 골문 앞에 머무는 트랙을 골키퍼로, 그 골문을 지키는 팀에 배정

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
def rts_smooth(ts: np.ndarray, z: np.ndarray, dt: float, accel_std: float, meas_std: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    등속 모델 칼만 필터 + Rauch-Tung-Striebel 역방향 평활화

    Args:
        ts: 측정 프레임 번호 (증가, 공백 허용), z: (N, 2) 측정 위치
    Returns:
        (평활 위치 (N, 2), 평활 속도 (N, 2)) — 측정 프레임에서의 값
    """
    n = len(ts)
    I4 = np.eye(4)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    R = np.eye(2) * meas_std ** 2
    xs_f, Ps_f, xs_p, Ps_p, Fs = [], [], [], [], []
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P = np.diag([meas_std ** 2, meas_std ** 2, 25.0, 25.0])
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
        S = Hm @ P @ Hm.T + R
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


def smooth_world(world: List[Optional[List[list]]], fps: float, ball_id: int = 0,
                 player_accel: float = 5.0, ball_accel: float = 40.0,
                 meas_std: float = 0.4, max_gap_s: float = 1.0) -> List[Optional[List[list]]]:
    """
    world[t] = [[id, x, y], ...] → [[id, x, y, speed(m/s)], ...] (평활화된 위치와 속력)

    트랙마다 max_gap_s 보다 긴 공백에서 구간을 나눠 따로 평활화한다. 보정 안 된 프레임(None)은 그대로 둔다.
    """
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
        for seg in np.split(np.arange(len(ts)), cuts):
            if len(seg) == 1:
                i = seg[0]
                out[ts[i]].append([tid, round(arr[i, 1], 2), round(arr[i, 2], 2), 0.0])
                continue
            pos, vel = rts_smooth(ts[seg], arr[seg, 1:], dt, ball_accel if tid == ball_id else player_accel,
                                  meas_std)
            for i, p, v in zip(seg, pos, vel):
                out[ts[i]].append([tid, round(float(p[0]), 2), round(float(p[1]), 2),
                                   round(float(np.hypot(*v)), 2)])
    return out


# ---------------------------------------------------------------------------
# 골키퍼
# ---------------------------------------------------------------------------
def assign_goalkeepers(world: List[Optional[List[list]]], teams: Dict[str, str], fps: float,
                       min_seconds: float = 1.0) -> Dict[str, str]:
    """
    'other' 트랙 중 대부분의 시간을 한쪽 페널티 박스 안에서 보낸 트랙을 골키퍼로 판정하고,
    그 골문 쪽에 평균 위치(무게중심)가 더 가까운 팀에 배정 (roboflow/sports 방식).
    teams 를 제자리 수정하고 {track_id: 'goalkeeper'} 역할 표를 반환한다.
    """
    pos: Dict[str, List[Tuple[float, float]]] = {}
    for entries in world:
        for e in entries or []:
            pos.setdefault(str(int(e[0])), []).append((e[1], e[2]))
    cent = {}
    for team in ('home', 'away'):
        xs = [x for tid, ps in pos.items() if teams.get(tid) == team for x, _ in ps]
        if xs:
            cent[team] = float(np.mean(xs))
    roles = {}
    if len(cent) < 2:
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
