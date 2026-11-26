"""패스 네트워크 (analysis/stats.pass_network) 테스트 — 합성 world 좌표만 사용

실행: python -m pytest tests/test_pass_network.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import stats

TEAMS = {'1': 'home', '2': 'home', '3': 'home', '9': 'away'}
POS = {1: (-20.0, 0.0), 2: (-5.0, 10.0), 3: (10.0, -10.0), 9: (30.0, 20.0)}


def world_with_ball_path(holders, hold=6):
    """holders 순서대로 공이 선수 발밑에 hold 프레임씩 머무는 경기 (선수는 고정 위치)"""
    world = []
    for pid in holders:
        bx, by = POS[pid]
        for _ in range(hold):
            world.append([[0, bx + 0.3, by, 0.0]] + [[i, x, y, 0.0] for i, (x, y) in POS.items()])
    return world


def compute(world, **extra):
    return stats.compute({'fps': 30.0, 'world': world, 'teams': TEAMS, 'roles': {}, **extra})


def test_pass_network_counts_same_team_passes_only():
    # 1 → 2 → 1 → 2 → 3 → (턴오버) 9
    out = compute(world_with_ball_path([1, 2, 1, 2, 3, 9]))
    net = out['pass_network']
    home = net['home']
    assert home['passes'] == 4
    assert {(e['a'], e['b']): e['count'] for e in home['edges']} == {(1, 2): 3, (2, 3): 1}
    e12 = next(e for e in home['edges'] if (e['a'], e['b']) == (1, 2))
    assert (e12['a_to_b'], e12['b_to_a']) == (2, 1)
    nodes = {n['id']: n for n in home['nodes']}
    assert set(nodes) == {1, 2, 3}                       # 9 는 턴오버로만 공을 받음 → 패스 노드 아님
    assert (nodes[1]['passes'], nodes[1]['received']) == (2, 1)
    assert (nodes[3]['passes'], nodes[3]['received']) == (0, 1)
    assert nodes[2]['position'] == pytest.approx(POS[2], abs=0.1)
    assert net['away'] == {'nodes': [], 'edges': [], 'passes': 0}
    # 팀 합계의 패스 수와 일치
    assert out['teams']['home']['passes'] == home['passes']


def test_pass_network_keeps_existing_keys_and_adds_jersey():
    out = compute(world_with_ball_path([1, 2]), identities={'2': {'jersey_number': 10}})
    for key in ('version', 'fps', 'covered_frames', 'players', 'teams', 'possession', 'events', 'shape', 'heatmaps'):
        assert key in out                               # 기존 키(D 섹터 도구가 읽음)는 그대로
    nodes = {n['id']: n for n in out['pass_network']['home']['nodes']}
    assert nodes[2]['jersey_number'] == 10 and nodes[1]['jersey_number'] is None


def test_pass_network_falls_back_to_mean_position_when_frame_missing():
    events = [{'type': 'pass', 'team': 'home', 'from': 1, 'to': 2, 'frame': 50, 'end_frame': 51}]
    players = {'1': {'mean_position': [-1.0, 2.0], 'role': 'player'}, '2': {'mean_position': [3.0, 4.0]}}
    net = stats.pass_network([], events, players)
    assert [n['position'] for n in net['home']['nodes']] == [[-1.0, 2.0], [3.0, 4.0]]
    assert net['home']['edges'] == [{'a': 1, 'b': 2, 'count': 1, 'a_to_b': 1, 'b_to_a': 0}]


def test_pass_network_empty_match():
    out = compute([None, [], None])
    assert out['pass_network'] == {t: {'nodes': [], 'edges': [], 'passes': 0} for t in ('home', 'away')}
