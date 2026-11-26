"""
여러 영상의 보고서 지표를 한 번에 재는 도구 (12부 성능 · 후반기 향후 계획 2·5순위)

지표 (영상 × 설정마다 한 행)
  - 보정 정상 프레임 비율·라인 오차(중앙값 px): scripts/eval_calibration.py 와 같은 판정 (1 초 간격 프레임)
  - 공 검출 프레임 수·비율, 선수 트랙 수, 화면 안 끊김 수, 등번호 확정 트랙 수: scripts/compare_tracking.py 와 같은 규칙
  - 최고 속력 (m/s, 공 제외)
  - 단계별 시간 (초): 탐지·추적, 후처리, SAM3 영상 추적, 팀 분류, 등번호, 자동 키프레임 탐색, 앵커 추적, 합계
  - 최대 VRAM (GiB, PyTorch 할당 기준 — run 모드에서만)
결과는 JSON(--out) 과 같은 이름의 Markdown 표(.md)로 저장하고 표를 화면에도 출력한다.

모드
  run     : 영상을 처음부터 분석 (백엔드와 같은 순서: 분석 → 자동 보정). GPU 사용. 결과 JSON·라인 마스크를 저장해
            나중에 eval/recalib 에 다시 쓸 수 있다
  recalib : 저장된 분석 결과(.json + .lines.npz)로 자동 보정만 다시 (CPU). 보정 설정 비교용
  eval    : 저장된 분석 결과의 지표만 계산

설정 비교: --variant 이름:구역.키=값,구역.키=값 (여러 번) — 구역은 video_analysis.PERF_DEFAULTS
           (sam_video.batch / sam_video.max_span / autocalib.every_s / autocalib.top_k / autocalib.refine_k /
            calib_track.anchor_every_s). 없으면 configs/model_config.yaml 의 perf 설정 그대로 한 번

예)
  # 하부리그 실증: 영상 여러 편 정밀 분석 (모델·영상이 있는 체크아웃에서 실행)
  python scripts/eval_suite.py run uploads/*.mp4 --out data/eval/precise.json --save-dir data/eval/results
  # SAM3 묶음 처리 전후 비교
  python scripts/eval_suite.py run uploads/sports_video_1.mp4 --out data/eval/sam_batch.json \\
      --variant one:sam_video.batch=false --variant batch:sam_video.batch=true
  # 자동 보정 설정 비교 (CPU)
  python scripts/eval_suite.py recalib uploads/analysis/sports_video_1.mp4.json --out data/eval/calib.json \\
      --variant base: --variant every1:autocalib.every_s=1.0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from pipeline import video_analysis as va  # noqa: E402

KIND_BALL = 1

# (키, 표 머리글, 형식)
COLUMNS: List[Tuple[str, str, str]] = [
    ('video', '영상', '{}'),
    ('variant', '설정', '{}'),
    ('frames', '프레임', '{}'),
    ('calib_ok_pct', '보정 정상 %', '{:.1f}'),
    ('line_err_px', '라인 오차 px', '{:.2f}'),
    ('ball_frames_pct', '공 검출 %', '{:.1f}'),
    ('tracks', '트랙', '{}'),
    ('breaks', '화면 안 끊김', '{}'),
    ('jersey_confirmed', '등번호 확정', '{}'),
    ('max_speed', '최고 속력 m/s', '{:.2f}'),
    ('t_detect_track', '탐지·추적 s', '{:.1f}'),
    ('t_sam_video', 'SAM3 s', '{:.1f}'),
    ('t_keyframes', '키프레임 s', '{:.1f}'),
    ('t_anchor', '앵커 s', '{:.1f}'),
    ('t_total', '합계 s', '{:.1f}'),
    ('peak_vram_gib', 'VRAM GiB', '{:.2f}'),
]


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def parse_value(text: str):
    low = text.strip().lower()
    if low in ('true', 'false'):
        return low == 'true'
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_variant(spec: str) -> Tuple[str, Dict[str, Dict]]:
    """'이름:구역.키=값,구역.키=값' → (이름, {구역: {키: 값}}). 값 없이 '이름:' 이면 기본 설정"""
    name, _, body = spec.partition(':')
    out: Dict[str, Dict] = {}
    for item in filter(None, (s.strip() for s in body.split(','))):
        key, eq, value = item.partition('=')
        section, dot, field = key.partition('.')
        if not eq or not dot:
            raise ValueError(f"설정 형식은 구역.키=값 입니다: {item}")
        if section not in va.PERF_DEFAULTS or field not in va.PERF_DEFAULTS[section]:
            raise ValueError(f"알 수 없는 설정: {key} (가능: "
                             + ', '.join(f'{s}.{k}' for s, d in va.PERF_DEFAULTS.items() for k in d) + ')')
        out.setdefault(section, {})[field] = parse_value(value)
    return name or 'default', out


def apply_variant(overrides: Dict[str, Dict], base: Dict[str, Dict]):
    """perf 설정을 base(시작 시점 설정)로 되돌린 뒤 overrides 적용"""
    for section in va.PERF_DEFAULTS:
        va.set_perf(section, **base[section])
    for section, values in overrides.items():
        va.set_perf(section, **values)


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def summarize(result: Dict, masks=None, every_s: float = 1.0) -> Dict:
    """분석 결과 (+ 라인 마스크) → 지표 dict. 보정이 없거나 마스크가 없으면 보정 지표는 None"""
    from compare_tracking import metrics as tracking_metrics
    from eval_calibration import evaluate as eval_calib, speed_summary

    frames = result.get('frames') or []
    n = len(frames)
    trk = tracking_metrics(result)
    ball = sum(any(o[1] == KIND_BALL for o in f or []) for f in frames)
    calib: Dict = {}
    if masks is not None and result.get('homographies'):
        calib, _ = eval_calib(result, masks, every_s)
    speed = speed_summary(result) if result.get('world') else {}
    timings = dict(result.get('timings') or {})
    auto = result.get('auto_calibration') or {}
    secs = auto.get('seconds') or {}
    row = {
        'frames': n,
        'fps': result.get('fps'),
        'calib_ok_pct': None if calib.get('ok_ratio') is None else round(100 * calib['ok_ratio'], 1),
        'line_err_px': calib.get('median_err_px'),
        'calib_judged_frames': calib.get('judged'),
        'auto_keyframes': auto.get('auto'),
        'ball_frames': ball,
        'ball_frames_pct': round(100 * ball / max(1, n), 1),
        'tracks': trk['player_tracks'],
        'tracks_all': trk['tracks'],
        'breaks': trk['breaks'],
        'overlap_scenes': trk['overlap_scenes'],
        'jersey_confirmed': trk['jersey_confirmed'],
        'max_speed': speed.get('max_speed'),
        'players_over_10': speed.get('players_over_10'),
        'sam_video': trk['sam_video'] or None,
        't_detect_track': timings.get('detect_track'),
        't_postprocess': timings.get('postprocess'),
        't_sam_video': timings.get('sam_video'),
        't_teams': timings.get('teams'),
        't_identity': timings.get('identity'),
        't_keyframes': secs.get('keyframes'),
        't_anchor': secs.get('tracking'),
    }
    return row


def format_markdown(rows: Sequence[Dict], columns: Sequence[Tuple[str, str, str]] = COLUMNS) -> str:
    cols = [c for c in columns if any(r.get(c[0]) is not None for r in rows)]
    lines = ['| ' + ' | '.join(h for _, h, _ in cols) + ' |',
             '|' + '|'.join('---' if k in ('video', 'variant') else '---:' for k, _, _ in cols) + '|']
    for r in rows:
        cells = []
        for k, _, fmt in cols:
            v = r.get(k)
            cells.append('-' if v is None else fmt.format(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_outputs(rows: List[Dict], out: Optional[Path], meta: Dict):
    table = format_markdown(rows)
    print('\n' + table)
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'meta': meta, 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    md = out.with_suffix('.md')
    md.write_text(f"# 평가 결과 ({meta.get('mode')}, {meta.get('created')})\n\n{table}\n", encoding='utf-8')
    print(f"\n저장: {out}, {md}")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def lines_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.name[:-len('.json')] + '.lines.npz')


def gpu_peak_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return True
    except ImportError:
        pass
    return False


def gpu_peak_gib() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    except ImportError:
        pass
    return None


def analyze(pipeline, video: Path, mode: str, max_frames: Optional[int]):
    """backend/server.py 의 분석 작업과 같은 순서: 프로필 → 분석(GPU 모델은 끝나면 내림) → 자동 보정"""
    from pipeline.pitch import LineMaskStore
    from tracking.tracker import BallTracker

    p = pipeline
    profile = p.analysis_profile(mode)
    stride = int(profile.get('stride', 1))
    det = p.detector
    for attr, key in (('imgsz', 'img_size'), ('ball_imgsz', 'ball_imgsz'), ('ball_tile', 'ball_tile')):
        if key in profile and hasattr(det, attr):
            setattr(det, attr, profile[key])
    tracker = p.make_tracker()
    tracker.ball_tracker = BallTracker(max_jump=60 * stride)
    lines = LineMaskStore((0, 0)) if profile.get('line_refine') else None
    embedder = p.make_team_embedder() if profile.get('team_classifier') == 'siglip' else None
    refiner = p.make_sam_refiner() if profile.get('sam_refine') else None
    jersey = p.make_jersey_reader() if profile.get('jersey_ocr') else None
    video_tracker = p.make_video_tracker() if profile.get('sam_video') else None
    t0 = time.perf_counter()
    try:
        result = va.analyze_video(video, det, tracker, stride=stride, mode=profile['mode'], line_store=lines,
                                  postprocess=profile.get('postprocess'), max_frames=max_frames,
                                  team_embedder=embedder, refiner=refiner, jersey_reader=jersey,
                                  video_tracker=video_tracker)
    finally:
        for m in (embedder, refiner, jersey, video_tracker):
            if m is not None:
                m.unload()
    t_analyze = time.perf_counter() - t0
    if lines is not None and len(lines) and profile.get('auto_calibrate'):
        try:
            result['auto_calibration'] = {'status': 'ok', **va.auto_calibrate(result, lines)}
        except ValueError as e:
            result['auto_calibration'] = {'status': 'failed', 'reason': str(e)}
    return result, lines, t_analyze, profile


def run_mode(args, variants, base) -> List[Dict]:
    from pipeline.main_pipeline import Soccer3DPipeline

    p = Soccer3DPipeline(config_path=str(args.config))
    p.initialize_detector()
    rows = []
    for video in args.inputs:
        for name, overrides in variants:
            apply_variant(overrides, base)
            gpu_peak_reset()
            t0 = time.perf_counter()
            result, lines, t_analyze, profile = analyze(p, video, args.analysis_mode, args.max_frames)
            total = time.perf_counter() - t0
            row = {'video': video.name, 'variant': name, **summarize(result, lines, args.every)}
            row.update(t_analyze=round(t_analyze, 1), t_total=round(total, 1), peak_vram_gib=gpu_peak_gib(),
                       profile={k: v for k, v in profile.items() if not isinstance(v, dict)})
            rows.append(row)
            if args.save_dir:
                dst = args.save_dir / name / f'{video.name}.json'
                va.save(result, dst)
                if lines is not None and len(lines):
                    lines.save(lines_path(dst))
            print(f"[{video.name} / {name}] {total:.1f}s")
    return rows


def recalib_mode(args, variants, base) -> List[Dict]:
    from pipeline.pitch import LineMaskStore

    rows = []
    for path in args.inputs:
        src = va.load(path)
        masks = LineMaskStore.load(args.lines or lines_path(path))
        for name, overrides in variants:
            apply_variant(overrides, base)
            result = json.loads(json.dumps(src))                 # 설정마다 원본에서 새로
            manual = [kf for kf in (result.get('calibration') or {}).get('keyframes', [])
                      if kf.get('source') != 'auto']
            result['calibration'] = {'keyframes': manual} if manual else None
            t0 = time.perf_counter()
            try:
                result['auto_calibration'] = {'status': 'ok',
                                              **va.auto_calibrate(result, masks, smooth=not args.no_smooth)}
            except ValueError as e:
                result['auto_calibration'] = {'status': 'failed', 'reason': str(e)}
            total = time.perf_counter() - t0
            row = {'video': result.get('video') or path.name, 'variant': name, **summarize(result, masks, args.every)}
            row['t_total'] = round(total, 1)
            row['perf'] = {s: va.perf_settings(s) for s in ('autocalib', 'calib_track')}
            rows.append(row)
            if args.save_dir:
                va.save(result, args.save_dir / name / path.name)
            print(f"[{path.name} / {name}] 자동 보정 {total:.1f}s, 키프레임 {row['auto_keyframes']}, "
                  f"정상 {row['calib_ok_pct']}%, 라인 오차 {row['line_err_px']} px")
    return rows


def eval_mode(args) -> List[Dict]:
    from pipeline.pitch import LineMaskStore

    rows = []
    for path in args.inputs:
        result = va.load(path)
        lp = args.lines or lines_path(path)
        masks = LineMaskStore.load(lp) if lp.is_file() else None
        rows.append({'video': result.get('video') or path.name, 'variant': path.parent.name,
                     **summarize(result, masks, args.every)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('run', 'recalib', 'eval'))
    ap.add_argument('inputs', nargs='+', type=Path, help='run: 영상 파일, recalib/eval: 분석 결과 JSON')
    ap.add_argument('--out', type=Path, default=None, help='결과 JSON (같은 이름 .md 로 표도 저장)')
    ap.add_argument('--variant', action='append', default=[], help='이름:구역.키=값,... (여러 번)')
    ap.add_argument('--every', type=float, default=1.0, help='보정 판정 프레임 간격 (초)')
    ap.add_argument('--analysis-mode', default='precise', choices=('precise', 'realtime'))
    ap.add_argument('--max-frames', type=int, default=None, help='run: 앞 N 프레임만 (스모크 실행)')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs' / 'model_config.yaml')
    ap.add_argument('--save-dir', type=Path, default=None, help='분석·재보정 결과 저장 폴더 (설정 이름별 하위 폴더)')
    ap.add_argument('--lines', type=Path, default=None, help='recalib/eval: 라인 마스크 (기본: 결과 옆 .lines.npz)')
    ap.add_argument('--no-smooth', action='store_true', help='recalib: 좌표 평활화·속력 계산 생략 (보정 지표만)')
    args = ap.parse_args(argv)
    mode = args.command

    variants = [parse_variant(v) for v in args.variant] or [('default', {})]
    base = {s: va.perf_settings(s) for s in va.PERF_DEFAULTS}
    meta = {'mode': mode, 'analysis_mode': args.analysis_mode, 'created': time.strftime('%Y-%m-%d %H:%M'),
            'perf_base': base, 'variants': {n: o for n, o in variants}}
    if mode == 'run':
        rows = run_mode(args, variants, base)
    elif mode == 'recalib':
        rows = recalib_mode(args, variants, base)
    else:
        rows = eval_mode(args)
    write_outputs(rows, args.out, meta)
    return rows


if __name__ == '__main__':
    main()
