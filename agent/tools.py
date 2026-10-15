"""
VLM 에이전트 도구 (3부 22.4절) — 분석 결과(JSON)와 원본 영상에서 수치·이미지를 꺼낸다

설계 원칙: 수치는 도구가, 해석은 VLM 이. 모든 도구는 JSON 으로 직렬화되는 dict 를 돌려주고,
근거가 된 프레임 번호를 'frames' 에 담아 프론트엔드가 그 장면으로 이동할 수 있게 한다.
get_frame / crop_player 는 이미지를 'image'(BGR ndarray)로 함께 돌려준다 (VLM 이 직접 '본다').
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

TEAM_ENUM = ['home', 'away', 'other']
BGR = {'home': (60, 60, 230), 'away': (230, 140, 40), 'other': (160, 160, 160)}


def _schema(name: str, description: str, properties: Dict[str, Dict], required: List[str] = ()) -> Dict:
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}


TIME_RANGE = {
    'start_s': {'type': 'number', 'description': '구간 시작 (초, 생략하면 영상 처음)'},
    'end_s': {'type': 'number', 'description': '구간 끝 (초, 생략하면 영상 끝)'},
}

SCHEMAS = [
    _schema('match_overview', '경기 영상 전체 요약: 길이, 팀 구성·색, 점유율, 팀별 이동 거리·스프린트·패스 수, 보정 상태', {}),
    _schema('player_stats', '선수(트랙 ID)별 이동 거리(m), 평균·최고 속력(m/s), 스프린트 횟수. '
            'player_id 를 주면 그 선수만, 아니면 sort_by 기준 상위 top 명 (모든 팀)', {
                'player_id': {'type': 'integer', 'description': '트랙 ID (화면의 #번호)'},
                'team': {'type': 'string', 'enum': TEAM_ENUM, 'description': '특정 팀만 볼 때만 지정'},
                'sort_by': {'type': 'string', 'enum': ['distance', 'max_speed', 'avg_speed', 'sprints'],
                            'description': "정렬 기준. '많이 뛴'=distance, '빠른'=max_speed (기본 distance)"},
                'top': {'type': 'integer', 'description': '상위 몇 명 (기본 5)'}}),
    _schema('get_possession', '구간 내 팀별 볼 점유율과 점유 구간(점유 선수 포함)', dict(TIME_RANGE)),
    _schema('find_events', '패스(pass)·턴오버(turnover)·스프린트(sprint) 이벤트 목록 (조건을 안 주면 전체)', {
        'type': {'type': 'string', 'enum': ['pass', 'turnover', 'sprint']},
        'team': {'type': 'string', 'enum': TEAM_ENUM, 'description': '특정 팀만 볼 때만 지정'},
        'player_id': {'type': 'integer'},
        **TIME_RANGE,
        'limit': {'type': 'integer', 'description': '최대 개수 (기본 15)'}}),
    _schema('team_shape', '특정 시점(또는 구간 평균)의 팀 대형: 무게중심, 폭, 길이, 수비 라인 x 좌표(m)', {
        'team': {'type': 'string', 'enum': ['home', 'away']},
        'time_s': {'type': 'number', 'description': '시점 (초). 생략하면 start_s~end_s 평균'},
        **TIME_RANGE}, ['team']),
    _schema('query_tracks', '선수 또는 공(ID 0)의 경기장 좌표 궤적 (x: 길이 방향, y: 너비 방향, m, 센터 스폿 원점)', {
        'player_id': {'type': 'integer', 'description': '트랙 ID, 공은 0'},
        **TIME_RANGE,
        'step_s': {'type': 'number', 'description': '샘플 간격 (초, 기본 1)'}}, ['player_id']),
    _schema('get_frame', '해당 시점의 영상 프레임 이미지 (탐지 박스·ID 표시). 장면을 직접 봐야 할 때만', {
        'time_s': {'type': 'number'}, 'frame': {'type': 'integer'}}),
    _schema('crop_player', '해당 시점 선수의 확대 이미지 (등번호·유니폼 확인용)', {
        'player_id': {'type': 'integer'}, 'time_s': {'type': 'number'}}, ['player_id']),
]


class MatchTools:
    """분석 결과 하나(영상 하나)에 대한 도구 모음"""

    def __init__(self, result: Dict, video_path: Optional[Path] = None):
        self.r = result
        self.video_path = Path(video_path) if video_path else None
        self.fps = float(result.get('fps') or 30.0)
        self.n = int(result.get('frame_count') or len(result.get('frames') or []))
        self.stats = result.get('stats') or {}
        self.teams = result.get('teams') or {}

    # -- 공통 도우미 ------------------------------------------------------
    def schemas(self) -> List[Dict]:
        return SCHEMAS

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict:
        fn: Optional[Callable] = getattr(self, name, None) if name in {s['function']['name'] for s in SCHEMAS} else None
        if fn is None:
            return {'error': f'알 수 없는 도구: {name}'}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {'error': f'인자 오류: {e}'}
        except Exception as e:  # 도구 실패는 모델에게 알려 다른 방법을 찾게 한다
            return {'error': str(e)}

    def _frame(self, time_s: Optional[float] = None, frame: Optional[int] = None) -> int:
        f = int(frame) if frame is not None else int(round(float(time_s or 0) * self.fps))
        return int(np.clip(f, 0, max(self.n - 1, 0)))

    def _range(self, start_s: Optional[float], end_s: Optional[float]):
        a = 0 if start_s is None else self._frame(start_s)
        b = self.n - 1 if end_s is None else self._frame(end_s)
        return min(a, b), max(a, b)

    def _t(self, f: int) -> float:
        return round(f / self.fps, 2)

    def _need_stats(self):
        if not self.stats:
            raise ValueError('경기 지표가 없습니다. 고정밀 분석 후 경기장 보정이 끝나야 계산됩니다.')

    # -- 도구 ------------------------------------------------------------
    def match_overview(self) -> Dict:
        teams = {}
        for name in ('home', 'away', 'other'):
            teams[name] = sum(1 for v in self.teams.values() if v == name)
        calib = self.r.get('calibration')
        out = {
            'video': self.r.get('video'), 'duration_s': self._t(self.n), 'fps': self.fps,
            'track_counts': teams, 'team_colors': self.r.get('team_colors'),
            'team_method': (self.r.get('team_method') or {}).get('method', 'color'),
            'calibrated': calib is not None,
            'calibrated_frames': sum(w is not None for w in (self.r.get('world') or [])),
            'frames': [0],
        }
        if self.stats:
            out['possession'] = (self.stats.get('possession') or {}).get('share')
            out['teams'] = self.stats.get('teams')
        return out

    def player_stats(self, player_id: Optional[int] = None, team: Optional[str] = None,
                     sort_by: str = 'distance', top: int = 5) -> Dict:
        self._need_stats()
        players = self.stats.get('players') or {}
        if player_id is not None:
            p = players.get(str(int(player_id)))
            if p is None:
                return {'error': f'#{player_id} 선수의 지표가 없습니다 (보정된 구간에 나오지 않았거나 없는 ID).'}
            frames = [s['start'] for s in p.get('sprints', [])[:3]]
            return {'player_id': int(player_id), **{k: v for k, v in p.items() if k != 'sprints'},
                    'sprints': [{'start_s': self._t(s['start']), 'end_s': self._t(s['end']),
                                 'max_speed': s['max_speed']} for s in p.get('sprints', [])],
                    'frames': frames}
        key = {'sprints': lambda p: len(p.get('sprints', []))}.get(sort_by, lambda p: p.get(sort_by) or 0)
        rows = [(int(k), p) for k, p in players.items() if team is None or p.get('team') == team]
        rows.sort(key=lambda kv: key(kv[1]), reverse=True)
        return {'sort_by': sort_by, 'players': [
            {'player_id': k, 'team': p.get('team'), 'role': p.get('role'), 'seconds': p.get('seconds'),
             'distance': p.get('distance'), 'avg_speed': p.get('avg_speed'), 'max_speed': p.get('max_speed'),
             'sprints': len(p.get('sprints', []))} for k, p in rows[:max(1, int(top))]], 'frames': []}

    def get_possession(self, start_s: Optional[float] = None, end_s: Optional[float] = None) -> Dict:
        self._need_stats()
        a, b = self._range(start_s, end_s)
        segs = (self.stats.get('possession') or {}).get('segments') or []
        total, share, out = 0, {}, []
        for s0, s1, team, player in segs:
            lo, hi = max(s0, a), min(s1, b)
            if hi < lo:
                continue
            n = hi - lo + 1
            total += n
            share[team] = share.get(team, 0) + n
            out.append({'start_s': self._t(lo), 'end_s': self._t(hi), 'team': team, 'player_id': player})
        return {'start_s': self._t(a), 'end_s': self._t(b),
                'share': {k: round(v / total, 3) for k, v in share.items()} if total else {},
                'segments': out[:30], 'frames': [self._frame(o['start_s']) for o in out[:5]]}

    def find_events(self, type: Optional[str] = None, team: Optional[str] = None,
                    player_id: Optional[int] = None, start_s: Optional[float] = None,
                    end_s: Optional[float] = None, limit: int = 15) -> Dict:
        self._need_stats()
        a, b = self._range(start_s, end_s)
        out = []
        for e in self.stats.get('events') or []:
            if type and e['type'] != type or team and e.get('team') != team or not a <= e['frame'] <= b:
                continue
            if player_id is not None and int(player_id) not in (e.get('player'), e.get('from'), e.get('to')):
                continue
            item = {k: v for k, v in e.items() if k not in ('frame', 'end_frame')}
            item.update(time_s=self._t(e['frame']), end_s=self._t(e.get('end_frame', e['frame'])), frame=e['frame'])
            out.append(item)
        return {'count': len(out), 'events': out[:max(1, int(limit))],
                'frames': [e['frame'] for e in out[:max(1, int(limit))]]}

    def team_shape(self, team: str, time_s: Optional[float] = None, start_s: Optional[float] = None,
                   end_s: Optional[float] = None) -> Dict:
        self._need_stats()
        shapes = [s for s in self.stats.get('shape') or [] if s.get(team)]
        if not shapes:
            return {'error': f'{team} 팀 대형 데이터가 없습니다.'}
        if time_s is not None:
            f = self._frame(time_s)
            s = min(shapes, key=lambda s: abs(s['frame'] - f))
            return {'team': team, 'time_s': self._t(s['frame']), **s[team], 'frames': [s['frame']]}
        a, b = self._range(start_s, end_s)
        sel = [s[team] for s in shapes if a <= s['frame'] <= b]
        if not sel:
            return {'error': '해당 구간의 대형 데이터가 없습니다.'}
        avg = {k: round(float(np.mean([s[k] for s in sel])), 2) for k in ('width', 'length', 'defensive_line')}
        avg['centroid'] = [round(float(np.mean([s['centroid'][i] for s in sel])), 2) for i in range(2)]
        return {'team': team, 'start_s': self._t(a), 'end_s': self._t(b), 'samples': len(sel), 'average': avg,
                'frames': [a]}

    def query_tracks(self, player_id: int, start_s: Optional[float] = None, end_s: Optional[float] = None,
                     step_s: float = 1.0) -> Dict:
        world = self.r.get('world')
        if not world:
            return {'error': '경기장 좌표가 없습니다 (경기장 보정 필요).'}
        a, b = self._range(start_s, end_s)
        step = max(1, int(round(float(step_s or 1.0) * self.fps)))
        pts = []
        for f in range(a, b + 1, step):
            for e in world[f] or []:
                if int(e[0]) == int(player_id):
                    pts.append({'time_s': self._t(f), 'x': e[1], 'y': e[2],
                                **({'speed': e[3]} if len(e) > 3 else {})})
                    break
        return {'player_id': int(player_id), 'team': self.teams.get(str(int(player_id)), 'ball' if int(player_id) == 0 else None),
                'points': pts[:120], 'frames': [a]}

    # -- 이미지 도구 -------------------------------------------------------
    def _read(self, f: int) -> np.ndarray:
        if self.video_path is None or not self.video_path.is_file():
            raise ValueError('원본 영상을 찾을 수 없습니다.')
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f'{f} 프레임을 읽지 못했습니다.')
        return img

    def get_frame(self, time_s: Optional[float] = None, frame: Optional[int] = None) -> Dict:
        f = self._frame(time_s, frame)
        img = self._read(f)
        for o in (self.r.get('frames') or [[]] * (f + 1))[f]:
            tid, kind, x1, y1, x2, y2 = o[:6]
            color = (0, 230, 255) if kind == 1 else BGR.get(self.teams.get(str(tid), 'other'), BGR['other'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, 'BALL' if kind == 1 else f'#{tid}', (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        h, w = img.shape[:2]
        scale = min(1.0, 1280 / max(w, 1))
        img = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
        return {'frame': f, 'time_s': self._t(f), 'image': img, 'frames': [f],
                'note': '박스 색: 빨강=home, 파랑=away, 회색=other, 노랑=공'}

    def crop_player(self, player_id: int, time_s: Optional[float] = None) -> Dict:
        frames = self.r.get('frames') or []
        f = self._frame(time_s) if time_s is not None else None
        cands = [t for t, objs in enumerate(frames) if any(o[0] == int(player_id) and o[1] == 0 for o in objs)]
        if not cands:
            return {'error': f'#{player_id} 선수가 영상에 없습니다.'}
        if f is None:  # 시점이 없으면 박스가 가장 큰 프레임
            f = max(cands, key=lambda t: next((o[5] - o[3]) for o in frames[t] if o[0] == int(player_id)))
        else:
            f = min(cands, key=lambda t: abs(t - f))
        o = next(o for o in frames[f] if o[0] == int(player_id))
        img = self._read(f)
        x1, y1, x2, y2 = o[2:6]
        pw, ph = int((x2 - x1) * 0.4), int((y2 - y1) * 0.2)
        crop = img[max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw]
        scale = 384 / max(crop.shape[0], 1)
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 384), interpolation=cv2.INTER_CUBIC)
        return {'player_id': int(player_id), 'team': self.teams.get(str(int(player_id))), 'frame': f,
                'time_s': self._t(f), 'image': crop, 'frames': [f]}
