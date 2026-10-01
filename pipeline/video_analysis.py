"""
업로드 영상 분석 모듈

영상 → 프레임별 탐지·추적 박스, 카메라 움직임, 팀 색상 → JSON 결과
경기장 보정(calibrate) 후에는 프레임별 경기장 좌표(미터)를 추가로 계산한다.

좌표 규칙
- 이미지: 원본 해상도 픽셀
- 경기장: 센터 스폿 원점, x = 길이 방향(오른쪽 +), y = 너비 방향(화면 먼 쪽 터치라인 +), 미터
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.tracker import SoccerTracker

KIND_PLAYER, KIND_BALL = 0, 1
MOTION_SCALE = 0.5          # 카메라 움직임 추정 해상도 배율 (1080p → 540p)
TEAM_SAMPLE_EVERY = 3       # 트랙별 유니폼 색 샘플 간격 (프레임)
TEAM_MAX_SAMPLES = 40
PITCH_MARGIN = 5.0          # m, 경기장 밖 이 거리까지는 유효 좌표로 인정 (라인 밖 스로인 등)
HALF_LENGTH, HALF_WIDTH = 52.5, 34.0


# ---------------------------------------------------------------------------
# 프레임 단위 도우미
# ---------------------------------------------------------------------------
def grass_mask(frame_small: np.ndarray) -> np.ndarray:
    """잔디(초록) 영역 마스크 — 관중석·광고판 위 오탐을 거르는 데 사용"""
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    # ponytail: 고정 HSV 범위. 조명/잔디색이 크게 다른 경기장은 범위 조정 필요
    return cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))


def on_grass(mask: np.ndarray, bbox: Tuple[int, int, int, int], scale: float, min_ratio: float = 0.15) -> bool:
    """
    발 주변(박스 하단을 좌우·아래로 넓힌 영역)에 잔디가 있는지

    다른 선수와 겹쳐 발밑이 가려져도 주변에 잔디가 보이도록 넓게 본다.
    관중석·광고판 위의 사람은 주변에 잔디가 거의 없어 걸러진다.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    h, w = mask.shape
    patch = mask[max(0, int((y2 - bh * 0.1) * scale)):min(h, int((y2 + bh * 0.15) * scale) + 1),
                 max(0, int((x1 - bw * 0.25) * scale)):min(w, int((x2 + bw * 0.25) * scale) + 1)]
    return patch.size > 0 and (patch > 0).mean() >= min_ratio


def plausible_ball(bbox: Tuple[int, int, int, int]) -> bool:
    """공 박스는 거의 정사각형 — 잔디 위 흰 줄·자국 같은 길쭉한 오탐 제거"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return h > 0 and 0.6 <= w / h <= 1.7


def camera_motion(prev_gray: np.ndarray, gray: np.ndarray, prev_boxes: List[Tuple[int, int, int, int]],
                  scale: float = MOTION_SCALE) -> Optional[np.ndarray]:
    """
    이전 프레임 → 현재 프레임 카메라 움직임 호모그래피 M (원본 해상도, p_t = M · p_{t-1})

    중계 카메라는 고정 위치에서 회전·줌하므로 배경 전체가 하나의 호모그래피로 움직인다.
    움직이는 선수 영역을 가리고 특징점 광류로 추정, 장면 전환 등 실패 시 None.
    """
    mask = np.full(prev_gray.shape, 255, np.uint8)
    for x1, y1, x2, y2 in prev_boxes:
        mask[int(y1 * scale):int(y2 * scale) + 1, int(x1 * scale):int(x2 * scale) + 1] = 0
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=8, mask=mask)
    if pts is None or len(pts) < 20:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 20:
        return None
    M, inliers = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
    if M is None or inliers is None or inliers.sum() < max(15, 0.3 * ok.sum()):
        return None
    S = np.diag([scale, scale, 1.0])
    return np.linalg.inv(S) @ M @ S


def torso_color(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """상체(유니폼) 영역의 평균 Lab 색 — 잔디 픽셀은 제외"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:
        return None
    crop = frame[y1 + int(h * 0.2):y1 + int(h * 0.5), x1 + int(w * 0.25):x2 - int(w * 0.25)]
    if crop.size == 0:
        return None
    keep = grass_mask(crop) == 0
    if keep.sum() < 10:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab[keep].mean(axis=0)


def cluster_teams(track_colors: Dict[int, List[np.ndarray]]) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    트랙별 대표 유니폼 색을 2 개로 군집 → 'home'/'away', 두 군집 모두에서 먼 트랙은 'other'(심판·골키퍼 등)

    ponytail: k=2 + 거리 기반 이상치. 골키퍼는 대개 'other' 로 분류됨.
    정확도가 필요하면 SigLIP 임베딩 + 위치 기반 골키퍼 보정으로 확장.
    """
    ids = [tid for tid, cs in track_colors.items() if len(cs) >= 3]
    if len(ids) < 2:
        return {tid: 'other' for tid in track_colors}, {}
    reps = np.float32([np.median(track_colors[tid], axis=0) for tid in ids])
    _, labels, centers = cv2.kmeans(reps, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5),
                                    5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    dist = np.linalg.norm(reps - centers[labels], axis=1)
    cutoff = max(3.0 * float(np.median(dist)), 18.0)  # Lab 거리
    order = np.argsort([-np.sum(labels == k) for k in range(2)])  # 트랙이 많은 군집을 home 으로
    names = {int(order[0]): 'home', int(order[1]): 'away'}
    teams = {tid: 'other' for tid in track_colors}
    for tid, lab, d in zip(ids, labels, dist):
        if d <= cutoff:
            teams[tid] = names[int(lab)]
    colors = {}
    for k, name in names.items():
        bgr = cv2.cvtColor(np.uint8([[centers[k]]]), cv2.COLOR_LAB2BGR)[0, 0]
        colors[name] = '#{:02x}{:02x}{:02x}'.format(int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return teams, colors


# ---------------------------------------------------------------------------
# 영상 분석
# ---------------------------------------------------------------------------
def interpolate_objects(a: List[list], b: List[list], alpha: float) -> List[list]:
    """두 탐지 프레임 사이 박스 선형 보간 — 양쪽 프레임에 모두 있는 트랙만"""
    by_id = {o[0]: o for o in b}
    out = []
    for o in a:
        other = by_id.get(o[0])
        if other is not None:
            box = [int(round(p + (q - p) * alpha)) for p, q in zip(o[2:6], other[2:6])]
            out.append([o[0], o[1], *box, min(o[6], other[6])])
    return out


def analyze_video(video_path: Path, detector, tracker: SoccerTracker,
                  progress: Callable[[int, int], None] = lambda done, total: None,
                  max_frames: Optional[int] = None, stride: int = 1, mode: str = 'realtime',
                  line_store=None, postprocess: Optional[bool] = None) -> Dict:
    """
    영상 전체를 분석해 결과 dict 반환

    Args:
        detector: detect(frame) -> List[Detection]
        tracker: SoccerTracker (공은 BallTracker 로 단일 ID 0)
        progress: (처리한 프레임 수, 전체 프레임 수) 콜백
        stride: N 프레임마다 탐지하고 사이 프레임은 트랙별 박스를 선형 보간
                (카메라 움직임은 모든 프레임에서 계산). CPU 에서는 3 정도 권장
        mode: 'realtime' | 'precise' — 결과에 기록되며 precise 는 기본으로 오프라인 후처리를 켠다
        line_store: pitch.LineMaskStore — 주면 프레임별 경기장 라인 마스크를 함께 만든다
                    (보정 시 라인 정렬로 누적 오차 제거에 사용)
        postprocess: 트랙 잇기·공백 보간·공 오탐 제거 (None 이면 precise 일 때만)
    """
    from . import pitch as pitch_lines

    stride = max(1, int(stride))
    if postprocess is None:
        postprocess = mode == 'precise'
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_frames:
        total = min(total, max_frames)

    frames: List[Optional[List[list]]] = []
    motion: List[Optional[list]] = []
    track_colors: Dict[int, List[np.ndarray]] = {}
    prev_gray, prev_boxes = None, []
    last_key = None  # 마지막 탐지 프레임 인덱스
    try:
        while len(frames) < total:
            ok, frame = cap.read()
            if not ok:
                break
            idx = len(frames)
            small = cv2.resize(frame, None, fx=MOTION_SCALE, fy=MOTION_SCALE, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if idx % stride == 0 or idx == total - 1:
                # 탐지 → 관중석 등 잔디 밖 사람·길쭉한 공 오탐 제거 → 추적
                grass = grass_mask(small)
                detections = [d for d in detector.detect(frame)
                              if (plausible_ball(d.bbox) if d.is_ball else on_grass(grass, d.bbox, MOTION_SCALE))]
                objs = []
                for o in tracker.update(detections, frame):
                    kind = KIND_BALL if o.track_id == tracker.BALL_TRACK_ID else KIND_PLAYER
                    objs.append([o.track_id, kind, *map(int, o.bbox), round(float(o.confidence), 2)])
                    if kind == KIND_PLAYER and idx % (TEAM_SAMPLE_EVERY * stride) == 0:
                        samples = track_colors.setdefault(o.track_id, [])
                        if len(samples) < TEAM_MAX_SAMPLES:
                            c = torso_color(frame, o.bbox)
                            if c is not None:
                                samples.append(c)
                frames.append(objs)
                if last_key is not None:  # 직전 탐지 프레임과의 사이를 보간으로 채움
                    for j in range(last_key + 1, idx):
                        frames[j] = interpolate_objects(frames[last_key], objs, (j - last_key) / (idx - last_key))
                last_key = idx
                prev_boxes = [o[2:6] for o in objs]
            else:
                frames.append(None)  # 다음 탐지 프레임에서 보간

            if line_store is not None:
                line_store.append(*pitch_lines.line_mask(small, prev_boxes, box_scale=MOTION_SCALE))
            M = camera_motion(prev_gray, gray, prev_boxes) if prev_gray is not None else None
            motion.append(None if M is None else np.round(M, 6).ravel().tolist())
            prev_gray = gray
            progress(idx + 1, total)
    finally:
        cap.release()
    frames = [f or [] for f in frames]  # 영상이 예상보다 일찍 끝난 경우 남은 빈칸 정리

    post = None
    if postprocess:
        from . import postprocess as pp
        stitched = pp.stitch_tracks(frames, motion, fps, track_colors)
        dropped = pp.clean_ball(frames, motion, fps)
        filled = pp.fill_gaps(frames, motion, int(round(1.5 * fps)), pp.KIND_PLAYER)
        filled_ball = pp.fill_gaps(frames, motion, int(round(0.7 * fps)), pp.KIND_BALL)
        post = {'stitched': len(stitched), 'ball_outliers': dropped,
                'filled_boxes': filled, 'filled_ball': filled_ball}

    teams, team_colors = cluster_teams(track_colors)
    return {
        'version': 2,
        'mode': mode,
        'postprocess': post,
        'video': video_path.name,
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': len(frames),
        'stride': stride,
        'frames': frames,
        'motion': motion,
        'teams': {str(k): v for k, v in teams.items()},
        'team_colors': team_colors,
        'calibration': None,
        'world': None,
    }


# ---------------------------------------------------------------------------
# 경기장 보정
# ---------------------------------------------------------------------------
def keyframe_homography(points: List[Dict]) -> np.ndarray:
    """[{image: [u, v], pitch: [x, y]}, ...] (4 개 이상) → 이미지 → 경기장 호모그래피"""
    if len(points) < 4:
        raise ValueError("보정점은 4 개 이상 필요합니다.")
    img = np.float32([p['image'] for p in points])
    pitch = np.float32([p['pitch'] for p in points])
    H, _ = cv2.findHomography(img, pitch, 0)
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("보정점이 한 직선 위에 있거나 중복되어 변환을 계산할 수 없습니다.")
    return H


def frame_homographies(motion: List[Optional[list]], keyframes: Dict[int, np.ndarray]) -> List[Optional[np.ndarray]]:
    """
    키프레임 보정을 카메라 움직임으로 전 프레임에 전파 (가장 가까운 키프레임 우선)

    H_t = H_k · M_k ··· M_{t+1}          (t < k)
    H_t = H_k · M_{k+1}⁻¹ ··· M_t⁻¹      (t > k)
    움직임 추정이 끊긴 구간(장면 전환 등) 너머로는 전파하지 않는다.
    """
    n = len(motion)
    Ms = [None if m is None else np.array(m).reshape(3, 3) for m in motion]
    H: List[Optional[np.ndarray]] = [None] * n
    steps = [np.inf] * n
    for k, Hk in keyframes.items():
        if not 0 <= k < n:
            continue
        H[k], steps[k] = Hk, 0
        cur = Hk
        for t in range(k + 1, n):                  # 앞으로
            if Ms[t] is None:
                break
            cur = cur @ np.linalg.inv(Ms[t])
            if t - k < steps[t]:
                H[t], steps[t] = cur, t - k
        cur = Hk
        for t in range(k - 1, -1, -1):             # 뒤로
            if Ms[t + 1] is None:
                break
            cur = cur @ Ms[t + 1]
            if k - t < steps[t]:
                H[t], steps[t] = cur, k - t
    return H


def ground_point(x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
    """경기장 평면에 닿는 지점: 선수는 발(박스 하단 중앙), 공도 하단 중앙(지면 가정)"""
    return (x1 + x2) / 2, float(y2)


def clipped_by_border(o: list, width: int, height: int, margin: int = 2) -> bool:
    """박스가 화면 좌·우·아래 가장자리에 닿아 잘렸는지 (위쪽은 머리가 잘려도 발 위치는 정확)"""
    if not width or not height:
        return False
    return o[2] <= margin or o[4] >= width - 1 - margin or o[5] >= height - 1 - margin


def calibrate(result: Dict, keyframe_points: List[Dict], line_masks=None, smooth: Optional[bool] = None) -> Dict:
    """
    keyframe_points: [{frame: int, points: [{image: [u, v], pitch: [x, y]}, ...]}, ...]
    → result['calibration'], result['world'] 갱신
    world[t] = [[track_id, x, y, speed], ...] (경기장 밖 PITCH_MARGIN 이상 벗어난 객체 제외), 보정 불가 프레임은 None

    Args:
        line_masks: pitch.LineMaskStore (또는 t → (라인, 유효 영역) 함수). 주면 매 프레임 경기장 라인에
                    정렬해 카메라 움직임 누적 오차를 제거한다 (정밀 분석 모드)
        smooth: 경기장 좌표 칼만+RTS 평활화, 골키퍼 판정, 경기 지표 계산 (None 이면 정밀 모드일 때만).
                끄면 world 는 [track_id, x, y] (속력 없음)
    """
    from . import pitch as pitch_lines

    if smooth is None:
        smooth = result.get('mode') == 'precise'
    keyframes = {int(kf['frame']): keyframe_homography(kf['points']) for kf in keyframe_points}
    if line_masks is not None:
        scale = line_masks.shape[1] / result['width'] if getattr(line_masks, 'shape', None) else MOTION_SCALE
        Hs, quality = pitch_lines.track_homographies(result['motion'], keyframes, line_masks,
                                                     pitch_lines.LineRefiner(scale=scale))
    else:
        Hs, quality = frame_homographies(result['motion'], keyframes), None
    world = []
    for objs, H in zip(result['frames'], Hs):
        if H is None:
            world.append(None)
            continue
        entries = []
        # 화면 가장자리에 잘린 선수 박스는 발 위치(하단 중앙)가 틀리므로 제외 — 속력이 튀는 주원인
        objs = [o for o in objs or [] if o[1] == KIND_BALL or not clipped_by_border(o, result.get('width'), result.get('height'))]
        if objs:
            pts = np.float32([ground_point(*o[2:6]) for o in objs]).reshape(-1, 1, 2)
            for o, (x, y) in zip(objs, cv2.perspectiveTransform(pts, H).reshape(-1, 2)):
                if abs(x) <= HALF_LENGTH + PITCH_MARGIN and abs(y) <= HALF_WIDTH + PITCH_MARGIN:
                    entries.append([o[0], round(float(x), 2), round(float(y), 2)])
        world.append(entries)
    calibration = {'keyframes': keyframe_points, 'refined': line_masks is not None}
    if quality is not None:
        calibration['line_fit'] = quality          # 프레임별 라인 정렬 비율 (정렬 안 됐으면 None)
        calibration['refined_frames'] = sum(q is not None for q in quality)
    result['calibration'] = calibration
    result['homographies'] = [None if H is None else np.round(H / H[2, 2], 8).ravel().tolist() for H in Hs]
    result['world'] = world
    if smooth:
        from . import postprocess as pp
        from analysis import stats as match_stats
        fps = result.get('fps') or 30.0
        result['world'] = pp.smooth_world(world, fps)
        # 골키퍼 배정은 자동 팀 분류 원본에서 매번 다시 계산 (보정을 다시 저장해도 결과가 누적되지 않도록)
        result.setdefault('teams_auto', dict(result.get('teams') or {}))
        teams = dict(result['teams_auto'])
        result['roles'] = pp.assign_goalkeepers(result['world'], teams, fps)
        result['teams'] = teams
        result['stats'] = match_stats.compute(result)
    return result


def save(result: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)  # 쓰는 도중 읽혀도 깨진 JSON 이 보이지 않도록 원자적 교체


def load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))
