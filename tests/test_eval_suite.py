"""평가 도구·성능 설정 테스트 (scripts/eval_suite.py, video_analysis.perf_settings, autocalib 후보 수 조정, 12부)

실행: python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import eval_suite as es  # noqa: E402
from pipeline import autocalib, video_analysis as va  # noqa: E402


@pytest.fixture
def restore_perf():
    saved = {s: va.perf_settings(s) for s in va.PERF_DEFAULTS}
    yield
    for s, v in saved.items():
        va._perf_config[s] = dict(v)


# ---------------------------------------------------------------------------
# 성능 설정
# ---------------------------------------------------------------------------
def test_perf_settings_have_defaults_for_every_section():
    for section, defaults in va.PERF_DEFAULTS.items():
        got = va.perf_settings(section)
        assert set(defaults) <= set(got)
    assert va.perf_settings('없는 구역') == {}


def test_set_perf_overrides_only_given_keys(restore_perf):
    before = va.perf_settings('autocalib')
    va.set_perf('autocalib', every_s=1.0)
    after = va.perf_settings('autocalib')
    assert after['every_s'] == 1.0 and after['top_k'] == before['top_k']


def test_auto_keyframes_passes_candidate_counts():
    calls = []

    class FakeCal:
        def calibrate(self, mask, valid, top_k=120, refine_k=8):
            calls.append((top_k, refine_k))
            return autocalib.AutoResult(None, 0, 0, 0, 1, 0, 0, '')

    mask = np.zeros((10, 10), np.uint8)
    out = autocalib.auto_keyframes([None] + [[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 59, lambda t: (mask, mask), (20, 20),
                                   fps=30.0, every_s=1.0, calibrator=FakeCal(), top_k=50, refine_k=3)
    assert out == [] and calls and set(calls) == {(50, 3)}
    assert len(calls) == 2                                     # 0, 30 프레임 (1 초 간격)


def test_calibrator_caches_are_consistent():
    cal = autocalib.AutoCalibrator(scale=0.5)
    assert cal._Ph.shape == (len(cal.P), 3) and np.all(cal._Ph[:, 2] == 1)
    assert len(cal._dense) == len(cal.polylines)
    # 행렬곱 채점 = einsum 채점
    G = np.random.default_rng(0).normal(size=(5, 3, 3))
    ref = np.einsum('kij,nj->kni', G, cal._Ph)
    assert np.allclose(cal._Ph @ G.transpose(0, 2, 1), ref)


# ---------------------------------------------------------------------------
# 평가 도구
# ---------------------------------------------------------------------------
def test_parse_variant():
    name, ov = es.parse_variant('fast:autocalib.every_s=1.0,sam_video.batch=false,autocalib.top_k=60')
    assert name == 'fast'
    assert ov == {'autocalib': {'every_s': 1.0, 'top_k': 60}, 'sam_video': {'batch': False}}
    assert es.parse_variant('base:') == ('base', {})
    with pytest.raises(ValueError):
        es.parse_variant('x:autocalib.nope=1')
    with pytest.raises(ValueError):
        es.parse_variant('x:every_s=1')


def test_apply_variant_resets_between_variants(restore_perf):
    base = {s: va.perf_settings(s) for s in va.PERF_DEFAULTS}
    es.apply_variant({'autocalib': {'every_s': 2.0}}, base)
    assert va.perf_settings('autocalib')['every_s'] == 2.0
    es.apply_variant({'calib_track': {'anchor_every_s': 0.3}}, base)
    assert va.perf_settings('autocalib')['every_s'] == base['autocalib']['every_s']
    assert va.perf_settings('calib_track')['anchor_every_s'] == 0.3


def synthetic_result():
    """선수 트랙 3 개 (그중 1 개는 화면 안에서 끊겼다 근처에서 새 ID 로 다시 시작) + 공 20 프레임"""
    frames = []
    for t in range(60):
        objs = [[1, 0, 400 + t, 500, 460 + t, 660, 0.9]] if t < 25 else [[2, 0, 400 + t, 500, 460 + t, 660, 0.9]]
        objs.append([3, 0, 1200, 500, 1260, 660, 0.9])
        if t % 3 == 0:
            objs.append([0, 1, 900, 600, 906, 606, 0.5])
        frames.append(objs)
    world = [[[1, 0.0, 0.0, 5.5], [3, 1.0, 1.0, 7.25], [0, 2.0, 2.0, 30.0]] for _ in range(60)]
    return {'video': 'clip.mp4', 'fps': 30.0, 'width': 1920, 'height': 1080, 'frame_count': 60, 'frames': frames,
            'world': world, 'identities': {'3': {'jersey_number': 7}, '1': {'jersey_number': None}},
            'timings': {'detect_track': 12.5, 'sam_video': 3.25},
            'auto_calibration': {'status': 'ok', 'auto': 4, 'seconds': {'keyframes': 2.0, 'tracking': 1.5}}}


def test_summarize_without_masks():
    row = es.summarize(synthetic_result())
    assert row['frames'] == 60
    assert row['ball_frames'] == 20 and row['ball_frames_pct'] == pytest.approx(33.3)
    assert row['tracks'] == 3 and row['breaks'] == 1
    assert row['jersey_confirmed'] == 1
    assert row['max_speed'] == 7.25                            # 공(ID 0) 제외
    assert row['calib_ok_pct'] is None and row['line_err_px'] is None
    assert (row['t_detect_track'], row['t_sam_video'], row['t_keyframes'], row['t_anchor']) == (12.5, 3.25, 2.0, 1.5)
    assert row['auto_keyframes'] == 4


def test_markdown_table_skips_empty_columns():
    rows = [{'video': 'a.mp4', 'variant': 'base', 'frames': 10, 'max_speed': 8.123, 'tracks': None},
            {'video': 'b.mp4', 'variant': 'base', 'frames': 20, 'max_speed': None, 'tracks': 4}]
    md = es.format_markdown(rows)
    lines = md.splitlines()
    assert lines[0].startswith('| 영상 | 설정 | 프레임 |') and '최고 속력 m/s' in lines[0] and 'VRAM' not in lines[0]
    assert '8.12' in lines[2] and lines[3].count('-') >= 1
    assert len(lines) == 4


def test_eval_mode_writes_json_and_markdown(tmp_path):
    src = tmp_path / 'clip.mp4.json'
    src.write_text(json.dumps(synthetic_result()), encoding='utf-8')
    out = tmp_path / 'out' / 'eval.json'
    rows = es.main(['eval', str(src), '--out', str(out)])
    assert len(rows) == 1 and rows[0]['video'] == 'clip.mp4'
    saved = json.loads(out.read_text(encoding='utf-8'))
    assert saved['rows'][0]['breaks'] == 1 and saved['meta']['mode'] == 'eval'
    assert out.with_suffix('.md').read_text(encoding='utf-8').count('|') > 10
