"""
경기 분석 지표 — 평활화된 경기장 좌표(world)에서 계산

선수별 이동 거리·최고/평균 속력·스프린트, 볼 점유(점유율·점유 구간), 패스·턴오버 이벤트,
팀 대형(무게중심·폭·길이·수비 라인), 팀 히트맵을 만든다.
3부 문서(22.4)의 VLM 도구(player_stats, get_possession, find_events, team_shape)가 그대로 쓰는 데이터다.

입력: 분석 결과 dict — fps, world[t] = [[id, x, y, speed], ...], teams {id: home|away|other}, roles
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

BALL_ID = 0
SPRINT_SPEED = 7.0          # m/s (약 25 km/h)
HIGH_SPEED = 5.5            # m/s (약 20 km/h) — 고강도 달리기
SPRINT_MIN_S = 1.0
CONTROL_RADIUS = 1.5        # m, 공과 이 거리 이내인 가장 가까운 선수를 점유 후보로
CONTROL_MIN_FRAMES = 3      # 이 프레임 수 이상 연속으로 후보여야 점유 확정 (스치는 경우 제외)
MAX_SPEED = 12.5            # m/s, 이보다 빠른 값은 보정 오차로 보고 거리 누적에서 제외
HEAT_CELL = 5.0             # m
SHAPE_EVERY_S = 0.5


def _series(world: List[Optional[List[list]]]) -> Dict[int, List[Tuple[int, float, float, float]]]:
    out: Dict[int, List[Tuple[int, float, float, float]]] = {}
    for t, entries in enumerate(world):
        for e in entries or []:
            speed = float(e[3]) if len(e) > 3 else float('nan')
            out.setdefault(int(e[0]), []).append((t, float(e[1]), float(e[2]), speed))
    return out


def player_stats(world, teams: Dict[str, str], roles: Dict[str, str], fps: float) -> Dict[str, Dict]:
    """선수(트랙)별 지표 — 공(ID 0)은 제외"""
    stats = {}
    max_step = max(1, int(round(0.5 * fps)))      # 0.5 초 넘게 끊긴 구간은 거리에 넣지 않음
    for tid, s in _series(world).items():
        if tid == BALL_ID:
            continue
        a = np.array(s)
        ts, xy, sp = a[:, 0].astype(int), a[:, 1:3], a[:, 3]
        if np.isnan(sp).all():                    # 평활 속력이 없으면 위치 차분으로
            sp = np.zeros(len(ts))
            if len(ts) > 1:
                v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / (np.diff(ts) / fps)
                sp[1:] = v
        sp = np.where(sp > MAX_SPEED, np.nan, sp)
        dist = 0.0
        hi = 0.0
        if len(ts) > 1:
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            dt = np.diff(ts)
            ok = (dt <= max_step) & (step / (dt / fps) <= MAX_SPEED)
            dist = float(step[ok].sum())
            hi = float(step[ok & (np.nan_to_num(sp[1:]) >= HIGH_SPEED)].sum())
        sprints = _sprints(ts, sp, fps)
        valid = sp[~np.isnan(sp)]
        stats[str(tid)] = {
            'team': teams.get(str(tid), 'other'),
            'role': roles.get(str(tid), 'player'),
            'frames': int(len(ts)),
            'seconds': round(len(ts) / fps, 2),
            'distance': round(dist, 1),
            'high_speed_distance': round(hi, 1),
            'max_speed': round(float(valid.max()), 2) if len(valid) else 0.0,
            'avg_speed': round(float(valid.mean()), 2) if len(valid) else 0.0,
            'sprints': sprints,
            'mean_position': [round(float(xy[:, 0].mean()), 1), round(float(xy[:, 1].mean()), 1)],
        }
    return stats


def _sprints(ts: np.ndarray, sp: np.ndarray, fps: float) -> List[Dict]:
    """SPRINT_SPEED 이상이 SPRINT_MIN_S 이상 지속된 구간"""
    fast = np.nan_to_num(sp) >= SPRINT_SPEED
    out, start = [], None
    for i in range(len(ts) + 1):
        on = i < len(ts) and fast[i] and (start is None or ts[i] - ts[i - 1] <= 2)
        if on and start is None:
            start = i
        elif not on and start is not None:
            end = i - 1
            if (ts[end] - ts[start] + 1) / fps >= SPRINT_MIN_S:
                out.append({'start': int(ts[start]), 'end': int(ts[end]),
                            'max_speed': round(float(np.nanmax(sp[start:end + 1])), 2)})
            start = i if (i < len(ts) and fast[i]) else None
    return out


def possession(world, teams: Dict[str, str], fps: float) -> Dict:
    """
    프레임별 공 소유자 판정 → 점유율, 점유 구간, 패스/턴오버 이벤트

    공에 CONTROL_RADIUS 이내로 가장 가까운 선수가 CONTROL_MIN_FRAMES 연속이면 소유자로 확정.
    소유자는 다른 선수가 확정될 때까지 유지(루즈볼 구간 포함). 같은 팀 선수로 바뀌면 패스, 다른 팀이면 턴오버.
    """
    owner, cand, streak = None, None, 0
    owners: List[Optional[int]] = []
    for entries in world:
        cur = None
        if entries:
            ball = next((e for e in entries if int(e[0]) == BALL_ID), None)
            if ball is not None:
                best, bd = None, CONTROL_RADIUS
                for e in entries:
                    if int(e[0]) == BALL_ID or teams.get(str(int(e[0]))) not in ('home', 'away'):
                        continue
                    d = float(np.hypot(e[1] - ball[1], e[2] - ball[2]))
                    if d <= bd:
                        best, bd = int(e[0]), d
                cur = best
        if cur is not None and cur == cand:
            streak += 1
        else:
            cand, streak = cur, (1 if cur is not None else 0)
        if cand is not None and streak >= CONTROL_MIN_FRAMES:
            owner = cand
        owners.append(owner)

    segments, events = [], []
    counts = {'home': 0, 'away': 0}
    for t, o in enumerate(owners):
        if o is None:
            continue
        team = teams.get(str(o), 'other')
        counts[team] = counts.get(team, 0) + 1
        if segments and segments[-1][3] == o and segments[-1][1] == t - 1:
            segments[-1][1] = t
        else:
            if segments and segments[-1][3] != o:
                prev = segments[-1]
                kind = 'pass' if prev[2] == team else 'turnover'
                events.append({'type': kind, 'frame': prev[1], 'end_frame': t, 'team': prev[2],
                               'from': prev[3], 'to': o})
            segments.append([t, t, team, o])
    total = counts['home'] + counts['away']
    share = {k: (round(counts[k] / total, 3) if total else 0.0) for k in ('home', 'away')}
    return {'share': share, 'frames': counts, 'segments': segments, 'events': events}


def team_sides(stats_players: Dict[str, Dict], world, teams: Dict[str, str]) -> Dict[str, int]:
    """팀이 지키는 골문 방향 (-1: 왼쪽 골문 x=-52.5, +1: 오른쪽). 골키퍼 위치 우선, 없으면 팀 무게중심"""
    sides = {}
    for tid, s in stats_players.items():
        if s['role'] == 'goalkeeper' and s['team'] in ('home', 'away'):
            sides[s['team']] = int(np.sign(s['mean_position'][0]) or -1)
    if len(sides) < 2:
        cx = {}
        for team in ('home', 'away'):
            xs = [e[1] for entries in world for e in entries or [] if teams.get(str(int(e[0]))) == team]
            if xs:
                cx[team] = float(np.mean(xs))
        if len(cx) == 2:
            left = min(cx, key=cx.get)
            sides = {left: -1, ('away' if left == 'home' else 'home'): 1}
        elif sides:
            only = next(iter(sides))
            sides[('away' if only == 'home' else 'home')] = -sides[only]
    return sides


def team_shape(world, teams: Dict[str, str], roles: Dict[str, str], sides: Dict[str, int], fps: float
               ) -> List[Dict]:
    """SHAPE_EVERY_S 간격으로 팀별 무게중심·폭·길이·수비 라인(골키퍼 제외 최후방 선수의 x)"""
    out = []
    every = max(1, int(round(SHAPE_EVERY_S * fps)))
    for t in range(0, len(world), every):
        entries = world[t]
        if not entries:
            continue
        row = {'frame': t}
        for team in ('home', 'away'):
            pts = np.array([[e[1], e[2]] for e in entries
                            if teams.get(str(int(e[0]))) == team and roles.get(str(int(e[0]))) != 'goalkeeper'])
            if len(pts) < 2:
                continue
            side = sides.get(team, -1)
            row[team] = {
                'centroid': [round(float(pts[:, 0].mean()), 1), round(float(pts[:, 1].mean()), 1)],
                'width': round(float(np.ptp(pts[:, 1])), 1),
                'length': round(float(np.ptp(pts[:, 0])), 1),
                'defensive_line': round(float(pts[:, 0].min() if side < 0 else pts[:, 0].max()), 1),
                'players': int(len(pts)),
            }
        if len(row) > 1:
            out.append(row)
    return out


def heatmaps(world, teams: Dict[str, str]) -> Dict[str, List[List[float]]]:
    """팀별 위치 밀도 (HEAT_CELL m 격자, 행 = y 아래→위, 열 = x 왼쪽→오른쪽, 최댓값 1 로 정규화)"""
    nx, ny = int(np.ceil(105 / HEAT_CELL)), int(np.ceil(68 / HEAT_CELL))
    grids = {team: np.zeros((ny, nx)) for team in ('home', 'away')}
    for entries in world:
        for e in entries or []:
            team = teams.get(str(int(e[0])))
            if team in grids:
                i = int(np.clip((e[2] + 34) // HEAT_CELL, 0, ny - 1))
                j = int(np.clip((e[1] + 52.5) // HEAT_CELL, 0, nx - 1))
                grids[team][i, j] += 1
    return {k: np.round(g / g.max(), 3).tolist() if g.max() > 0 else g.tolist() for k, g in grids.items()}


def compute(result: Dict) -> Dict:
    """분석 결과(보정 완료) → 지표 dict"""
    world = result.get('world') or []
    fps = float(result.get('fps') or 30.0)
    teams = result.get('teams') or {}
    roles = result.get('roles') or {}
    players = player_stats(world, teams, roles, fps)
    poss = possession(world, teams, fps)
    sides = team_sides(players, world, teams)
    team_totals = {}
    for team in ('home', 'away'):
        members = [s for s in players.values() if s['team'] == team]
        team_totals[team] = {
            'players': len(members),
            'distance': round(sum(s['distance'] for s in members), 1),
            'sprints': sum(len(s['sprints']) for s in members),
            'possession': poss['share'][team],
            'passes': sum(1 for e in poss['events'] if e['type'] == 'pass' and e['team'] == team),
            'turnovers': sum(1 for e in poss['events'] if e['type'] == 'turnover' and e['team'] == team),
            'defends': sides.get(team),
        }
    events = list(poss['events'])
    for tid, s in players.items():
        for sp in s['sprints']:
            events.append({'type': 'sprint', 'frame': sp['start'], 'end_frame': sp['end'], 'team': s['team'],
                           'player': int(tid), 'max_speed': sp['max_speed']})
    events.sort(key=lambda e: e['frame'])
    return {
        'version': 1,
        'fps': fps,
        'covered_frames': sum(w is not None for w in world),
        'players': players,
        'teams': team_totals,
        'possession': {'share': poss['share'], 'segments': poss['segments']},
        'events': events,
        'shape': team_shape(world, teams, roles, sides, fps),
        'heatmaps': heatmaps(world, teams),
    }
