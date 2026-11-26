"""
선수 속력 이상값 처리 (11부) — pipeline/postprocess.find_jumps, smooth_world(speed_filter)

실제 영상(sports_video_1)에서 본 세 가지 모양을 합성 궤적으로 재현한다.
  - 정상 질주·가속: 프레임마다 발 위치가 0.5~2 m 흔들려도 건드리지 않아야 한다
  - ID 교체(가려진 선수 박스가 뒤 선수로 미끄러짐): 위치가 0.2 초 만에 수 m 옮겨 가고 돌아오지 않음 → 끊기
  - 잠깐 튐(박스가 몸통·뒤 선수로 갔다 돌아옴): 0.3 초 동안 4 m 옆 → 그 사이를 결측처럼 메우기
"""

import numpy as np

from pipeline import postprocess as pp

FPS = 30.0


def _noisy(xy, seed=0, std=0.5, spikes=0.03):
    """실제 발 위치처럼: 가우스 잡음 + 가끔 2 m 튀는 프레임"""
    rng = np.random.default_rng(seed)
    out = xy + rng.normal(0, std, xy.shape)
    hit = rng.random(len(xy)) < spikes
    out[hit] += rng.normal(0, 2.0, (hit.sum(), 2))
    return out


def _world(tracks, n):
    """{tid: (ts, xy)} → world[t] = [[id, x, y], ...]"""
    world = [[] for _ in range(n)]
    for tid, (ts, xy) in tracks.items():
        for t, (x, y) in zip(ts, xy):
            world[int(t)].append([tid, float(x), float(y)])
    return world


def _speeds(out, tid):
    return np.array([e[3] for ent in out for e in ent or [] if e[0] == tid])


def _sprint(n=90, v=8.0, seed=0):
    ts = np.arange(n)
    xy = np.c_[v * ts / FPS, np.zeros(n)]
    return ts, xy, _noisy(xy, seed)


def test_normal_sprint_and_acceleration_untouched():
    for seed in range(5):
        ts, _, z = _sprint(v=9.5, seed=seed)
        bad, cuts = pp.find_jumps(ts, z, FPS)
        assert not bad.any() and cuts == []
        # 0 → 9 m/s 를 1 초 만에 (가속 9 m/s²) — 정상 범위의 가장 센 출발
        t = np.arange(75) / FPS
        tau = np.clip(t - 0.5, 0, 1.0)
        x = 4.5 * tau ** 2 + 9.0 * np.clip(t - 1.5, 0, None)
        z = _noisy(np.c_[x, np.zeros(75)], seed)
        bad, cuts = pp.find_jumps(np.arange(75), z, FPS)
        assert not bad.any() and cuts == []


def test_real_sprint_speed_is_kept():
    ts, _, z = _sprint(n=120, v=9.0, seed=3)
    world = _world({5: (ts, z)}, len(ts))
    on = _speeds(pp.smooth_world(world, FPS, speed_filter=True), 5)
    off = _speeds(pp.smooth_world(world, FPS, speed_filter=False), 5)
    np.testing.assert_allclose(on, off)                      # 점프가 없으면 결과가 완전히 같다
    assert 8.5 < np.median(on[30:-30]) < 9.5


def test_id_switch_is_cut_instead_of_smoothed_into_a_sprint():
    # 3 m/s 로 걷던 트랙이 0.15 초 동안 박스가 미끄러지며 9 m 옆 선수에게 넘어가 그대로 이어짐 (실제 트랙 64)
    n = 90
    ts = np.arange(n)
    x = 3.0 * ts / FPS
    y = np.zeros(n)
    y[45:] = 9.0
    y[40:45] = np.linspace(0, 9.0, 7)[1:-1]
    z = _noisy(np.c_[x, y], seed=1, std=0.3, spikes=0)
    bad, cuts = pp.find_jumps(ts, z, FPS)
    assert len(cuts) == 1 and 38 <= cuts[0] <= 47
    assert bad[40:45].all() and not bad[:35].any() and not bad[52:].any()

    world = _world({7: (ts, z)}, n)
    off = _speeds(pp.smooth_world(world, FPS, speed_filter=False), 7)
    rep = {}
    on_out = pp.smooth_world(world, FPS, speed_filter=True, report=rep)
    on = _speeds(on_out, 7)
    assert off.max() > 11.0                                  # 처리 전: 순간 이동이 '질주'로 퍼짐
    assert on.max() < 6.0                                    # 처리 후: 걷는 속도 그대로
    assert rep['jumps'] == 1 and rep['cuts'] == 1 and rep['tracks'][0]['id'] == 7
    # 끊긴 자리의 이상값(구간 끝)은 내보내지 않음 — 나머지 프레임은 모두 남음
    kept = [t for t, ent in enumerate(on_out) if any(e[0] == 7 for e in ent)]
    assert rep['dropped_points'] == n - len(kept) > 0
    assert all(t in kept for t in range(0, 38)) and all(t in kept for t in range(50, n))


def test_short_excursion_is_bridged_without_cut():
    # 7.5 m/s 질주 중 0.3 초 동안 발 위치가 4 m 옆(뒤 선수)으로 갔다 돌아옴 (실제 트랙 53/54)
    n = 120
    ts = np.arange(n)
    xy = np.c_[7.5 * ts / FPS, np.zeros(n)]
    xy[60:70, 1] += 4.0
    z = _noisy(xy, seed=2, std=0.4, spikes=0)
    bad, cuts = pp.find_jumps(ts, z, FPS)
    assert cuts == []
    assert bad[60:70].all() and bad.sum() < 20

    world = _world({3: (ts, z)}, n)
    off_out = pp.smooth_world(world, FPS, speed_filter=False)
    on_out = pp.smooth_world(world, FPS, speed_filter=True)
    on = _speeds(on_out, 3)
    assert len(on) == n                                      # 메운 것이지 버린 것이 아님
    assert on.max() < 8.0
    ys = np.array([e[2] for ent in on_out for e in ent if e[0] == 3])
    ys_off = np.array([e[2] for ent in off_out for e in ent if e[0] == 3])
    assert np.abs(ys[60:70]).max() < 1.0 < np.abs(ys_off[60:70]).max()   # 메운 위치는 원래 궤적 위


def test_short_piece_between_jumps_is_dropped():
    # 이력이 짧은 트랙(0.25 초)이 곧바로 0.15 초 튀었다 돌아옴 → 사이 조각을 믿지 않음 (실제 트랙 90)
    rng = np.random.default_rng(4)
    n = 45
    ts = np.arange(n)
    xy = np.c_[np.full(n, 46.0), np.zeros(n)] + rng.normal(0, 0.6, (n, 2))
    xy[9:14] += [2.5, 3.5]
    world = _world({9: (ts, xy)}, n)
    out = pp.smooth_world(world, FPS, speed_filter=True)
    assert _speeds(out, 9).max() < 6.0
    assert _speeds(pp.smooth_world(world, FPS, speed_filter=False), 9).max() > _speeds(out, 9).max()


def test_ball_and_disabled_filter_keep_old_behaviour():
    n = 60
    ts = np.arange(n)
    ball = np.c_[20.0 * ts / FPS, np.zeros(n)]
    ball[30:] += [0, 8.0]                                    # 공은 튀어도(패스·슈팅) 그대로 둔다
    world = _world({0: (ts, ball)}, n)
    a = pp.smooth_world(world, FPS, speed_filter=True)
    b = pp.smooth_world(world, FPS, speed_filter=False)
    assert a == b
    assert pp.smooth_world(world, FPS, speed_filter={'enabled': False}) == b


def test_config_block_is_loaded_by_default(tmp_path):
    cfg = pp.load_speed_filter()
    assert cfg['enabled'] is True and cfg['max_speed'] == 11.0
    p = tmp_path / 'c.yaml'
    p.write_text('speed_filter:\n  enabled: false\n  margin: 2.0\n', encoding='utf-8')
    cfg = pp.load_speed_filter(str(p))
    assert cfg['enabled'] is False and cfg['margin'] == 2.0 and cfg['window_s'] == 0.2
    assert pp.load_speed_filter(str(tmp_path / 'missing.yaml')) == pp.SPEED_FILTER_DEFAULTS


def test_smooth_world_default_uses_filter():
    n = 90
    ts = np.arange(n)
    y = np.zeros(n)
    y[45:] = 9.0
    z = np.c_[3.0 * ts / FPS, y]
    world = _world({7: (ts, z)}, n)
    assert _speeds(pp.smooth_world(world, FPS), 7).max() < 6.0      # video_analysis.calibrate 기본 호출 경로
