"""
SAM3 영상 추적으로 트랙 끊김·ID 교체 바로잡기 (10부, 3부 24절 로드맵 7번, 정밀 모드 · GPU)

정밀 모드의 SAM3 는 프레임마다 따로(이미지 모드) 발 위치·유니폼 마스크만 구한다. 추적은 BoT-SORT 박스 매칭이라
선수가 겹치거나 가려지면 트랙이 끊기거나 두 선수의 ID 가 바뀐다. 이 모듈은 그런 **위험 구간만** 골라
SAM3 영상 추적기(transformers Sam3TrackerVideoModel, 메모리 기반 마스크 전파)로 다시 따라가 본다.

  1. 끊김   : 트랙 A 가 화면 안에서 끝나고 window 프레임 안에 근처에서 트랙 B 가 시작하면,
              A 의 마지막 박스를 프롬프트로 마스크를 전파해 B 가 시작하는 프레임에서 A 의 마스크와 B 의 박스가
              겹치면(IoU ≥ link_iou) B 를 A 로 잇는다 (가림 동안 SAM 메모리가 같은 선수를 유지)
  2. ID 교체: 두 트랙 박스가 겹친 뒤 떨어지는 장면에서, 겹치기 전 박스로 두 선수를 프롬프트해 전파하고
              떨어진 뒤 A 박스가 B 마스크와, B 박스가 A 마스크와 맞는 상태가 이어지면 그 시점부터 두 ID 를 바꾼다
  3. 비용   : 위험 구간·관련 선수만 전파 (객체 1~2 개, 30~40 프레임). 영상 전체 전파는 RTX 5070 에서
              10 명 기준 초당 1.5 프레임이라 정밀 분석 시간을 두 배 이상 늘린다
  4. 묶기   : 시간이 겹치는 위험 구간은 한 덩어리(merge_windows, 최대 max_span 프레임)로 묶어 영상을 한 번 읽고
              한 세션에서 처리한다 (track_many). 프레임 영상 특징(ViT, 전파 비용 대부분)은 덩어리 안에서 한 번만
              계산해 재사용하고, 같은 프레임에서 시작하는 사건의 선수들은 한 번의 전파로 함께 따라간다.
              사건마다 추적 상태(메모리)는 새로 시작하므로 결과는 사건을 하나씩 처리할 때와 같다
              (SAM 메모리는 객체별이라 객체끼리 섞이지 않는다). ID 교체를 바로잡아 뒤 사건의 프롬프트 박스가
              바뀌면 그 사건만 바뀐 프롬프트로 다시 전파한다

마스크 IoU 는 SAM 저해상도 출력(288×288, 화면 비율을 무시하고 정사각형으로 리사이즈된 좌표)을 원본 좌표로 늘려 잰다.
"""

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

KIND_PLAYER = 0
Box = Tuple[float, float, float, float]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def player_tracks(frames: List[List[list]]) -> Dict[int, Dict[int, Box]]:
    """선수 트랙 ID → {프레임: 박스} (보간 박스 포함)"""
    out: Dict[int, Dict[int, Box]] = {}
    for t, objs in enumerate(frames):
        for o in objs or []:
            if o[1] == KIND_PLAYER:
                out.setdefault(int(o[0]), {})[t] = tuple(float(v) for v in o[2:6])
    return out


def at_border(box: Box, width: int, height: int, margin: float = 8.0) -> bool:
    return box[0] <= margin or box[2] >= width - margin or box[3] >= height - margin


# ---------------------------------------------------------------------------
# 위험 구간 찾기
# ---------------------------------------------------------------------------
def break_events(tracks, n_frames: int, size: Tuple[int, int], window: int = 30, reach: float = 3.0
                 ) -> List[Tuple[int, int, List[int]]]:
    """
    [(A, A 의 끝 프레임, 후보 B 목록), ...] — A 가 화면 안에서 끝나고 window 안에 근처(A 박스 높이 × reach 이내)에서
    시작하는 트랙들. 화면 가장자리에서 끝난 트랙(화면 밖으로 나감)은 제외
    """
    w, h = size
    starts = {tid: min(fr) for tid, fr in tracks.items()}
    out = []
    for a, fr in tracks.items():
        e = max(fr)
        if e >= n_frames - 2 or at_border(fr[e], w, h):
            continue
        ax = (fr[e][0] + fr[e][2]) / 2, fr[e][3]
        ah = fr[e][3] - fr[e][1]
        cands = []
        for b, s in starts.items():
            if b == a or not e < s <= e + window:
                continue
            bb = tracks[b][s]
            if np.hypot((bb[0] + bb[2]) / 2 - ax[0], bb[3] - ax[1]) <= reach * max(ah, 20):
                cands.append(b)
        if cands:
            out.append((a, e, cands))
    return out


def overlap_events(tracks, min_iou: float = 0.3, min_after: int = 10, lead: int = 8
                   ) -> List[Tuple[int, int, int, int]]:
    """
    [(A, B, 겹침 시작, 겹침 끝), ...] — 두 트랙 박스가 IoU ≥ min_iou 로 겹쳤다가, 겹친 뒤에도 둘 다 min_after 프레임
    이상 이어지고, 겹치기 lead 프레임 전에 둘 다 있던 장면 (ID 가 바뀌었는지 확인할 수 있는 장면만)
    """
    ids = sorted(tracks)
    out = []
    for i, a in enumerate(ids):
        fa = tracks[a]
        for b in ids[i + 1:]:
            fb = tracks[b]
            common = sorted(set(fa) & set(fb))
            if not common:
                continue
            hit = [t for t in common if box_iou(fa[t], fb[t]) >= min_iou]
            if not hit:
                continue
            # 연속 구간으로 묶기 (5 프레임 이하 틈은 같은 겹침)
            groups, cur = [], [hit[0]]
            for t in hit[1:]:
                if t - cur[-1] <= 5:
                    cur.append(t)
                else:
                    groups.append(cur)
                    cur = [t]
            groups.append(cur)
            for g in groups:
                s, e = g[0], g[-1]
                if (s - lead in fa and s - lead in fb and e + min_after in fa and e + min_after in fb):
                    out.append((a, b, s, e))
    return out


def merge_windows(windows: Sequence[Tuple[int, int]], max_span: int = 150, max_gap: int = 0
                  ) -> List[Tuple[int, int, List[int]]]:
    """
    위험 구간 [(시작, 끝 포함), ...] → 덩어리 [(시작, 끝, [구간 번호, ...]), ...] (시작 순)

    시간이 겹치거나 max_gap 프레임 이하로 떨어진 구간을 묶되, 덩어리 길이(끝 - 시작 + 1)가 max_span 을 넘으면
    새 덩어리를 연다 (덩어리 프레임과 영상 특징을 메모리에 들고 있어야 하므로). max_span 보다 긴 구간은 혼자 한 덩어리.
    """
    order = sorted(range(len(windows)), key=lambda i: (windows[i][0], windows[i][1]))
    out: List[Tuple[int, int, List[int]]] = []
    for i in order:
        s, e = int(windows[i][0]), int(windows[i][1])
        if out:
            gs, ge, idx = out[-1]
            if s <= ge + 1 + max_gap and max(ge, e) - gs + 1 <= max_span:
                out[-1] = (gs, max(ge, e), idx + [i])
                continue
        out.append((s, e, [i]))
    return out


Job = Tuple[int, int, Dict[int, Box]]          # (덩어리 안 시작 프레임, 끝 프레임 포함, {객체: 시작 프레임 박스})


def track_jobs(tracker, clip: List[np.ndarray], jobs: Sequence[Job]) -> List[Optional[List[Dict[int, Optional[Box]]]]]:
    """
    한 덩어리 프레임(clip)에서 여러 사건을 추적. 사건 결과는 tracker.track(clip[s:e+1], prompts) 와 같은 모양
    (사건 시작 기준 프레임별 {객체: 박스}). 읽은 프레임이 2 개 미만인 사건은 None (하나씩 처리할 때와 같은 규칙)
    """
    out: List[Optional[List[Dict[int, Optional[Box]]]]] = [None] * len(jobs)
    run = [j for j, (s, e, _) in enumerate(jobs) if len(clip[s:e + 1]) >= 2]
    if not run:
        return out
    for j, r in zip(run, tracker.track_many(clip, [jobs[j] for j in run])):
        out[j] = r
    return out


def _same_prompts(a: Dict[int, Box], b: Dict[int, Box]) -> bool:
    return a.keys() == b.keys() and all(np.allclose(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# SAM3 영상 추적
# ---------------------------------------------------------------------------
class _FrameFeatureCache:
    """
    세션의 프레임 영상 특징 캐시를 대신한다 (transformers 기본은 최근 1 프레임만 보관):
    덩어리 안 모든 프레임 특징을 들고 있어 사건마다 ViT 를 다시 돌리지 않는다. 위치 임베딩은 해상도가 같으면
    프레임과 무관하게 같으므로 하나만 저장해 VRAM 을 아낀다 (값이 다르면 그 프레임 것을 따로 저장)
    """

    def __init__(self, device):
        self.device = device
        self._feats: Dict[int, list] = {}
        self._pos: Dict[int, list] = {}
        self._shared_pos: Optional[list] = None

    def cache_vision_features(self, frame_idx: int, features: dict):
        import torch
        feats = [v.to(self.device, non_blocking=True) for v in features['vision_feats']]
        pos = list(features['vision_pos_embeds'])
        if self._shared_pos is None:
            self._shared_pos = [p.to(self.device, non_blocking=True) for p in pos]
        elif not (len(pos) == len(self._shared_pos) and all(
                p.shape == q.shape and torch.equal(p.to(q.device), q) for p, q in zip(pos, self._shared_pos))):
            self._pos[frame_idx] = [p.to(self.device, non_blocking=True) for p in pos]
        self._feats[frame_idx] = feats

    def get_vision_features(self, frame_idx: int) -> Optional[dict]:
        if frame_idx not in self._feats:
            return None
        return {'vision_feats': self._feats[frame_idx],
                'vision_pos_embeds': self._pos.get(frame_idx, self._shared_pos)}

    def clear_all(self):
        self._feats.clear()
        self._pos.clear()
        self._shared_pos = None

    def __len__(self):
        return len(self._feats)


class Sam3VideoTracker:
    """
    Sam3TrackerVideoModel 래퍼: 짧은 구간 프레임 + 첫 프레임 박스 프롬프트 → 프레임별 객체 마스크 외접 박스
    track(frames_rgb, {obj_id: box}) -> [{obj_id: box 또는 None}, ...]  (원본 좌표)
    track_many(frames_rgb, [(시작, 끝, {obj_id: box}), ...]) -> 사건별 track 결과 (한 세션, 영상 특징 재사용)
    """

    def __init__(self, model_id: str = 'facebook/sam3', device: str = 'cuda', min_area: int = 6):
        self.model_id, self.device, self.min_area = model_id, device, min_area
        self.model = self.processor = None

    def _load(self):
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        self.dtype = torch.bfloat16 if str(self.device).startswith('cuda') else torch.float32
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_id)
        self.model = Sam3TrackerVideoModel.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()

    def track(self, frames_rgb: List[np.ndarray], prompts: Dict[int, Box]) -> List[Dict[int, Optional[Box]]]:
        import torch
        if self.model is None:
            self._load()
        H, W = frames_rgb[0].shape[:2]
        ids = list(prompts)
        sess = self.processor.init_video_session(video=frames_rgb, inference_device=self.device,
                                                 video_storage_device='cpu', dtype=self.dtype)
        self.processor.add_inputs_to_inference_session(
            sess, frame_idx=0, obj_ids=ids, input_boxes=[[[float(v) for v in prompts[i]] for i in ids]])
        out: List[Dict[int, Optional[Box]]] = [dict() for _ in frames_rgb]
        with torch.inference_mode():
            self.model(inference_session=sess, frame_idx=0)
            for res in self.model.propagate_in_video_iterator(sess, start_frame_idx=0):
                k = int(res.frame_idx)
                masks = (res.pred_masks[:, 0] > 0).cpu().numpy()          # (객체, 288, 288)
                for j, oid in enumerate(res.object_ids if hasattr(res, 'object_ids') else ids):
                    out[k][int(oid)] = self._mask_box(masks[j], W, H)
        return out

    def _mask_box(self, mask: np.ndarray, W: int, H: int) -> Optional[Box]:
        """SAM 저해상도 마스크 → 원본 좌표 외접 박스 (픽셀이 min_area 미만이면 None)"""
        mh, mw = mask.shape
        ys, xs = np.nonzero(mask)
        if len(ys) < self.min_area:
            return None
        return (xs.min() * W / mw, ys.min() * H / mh, (xs.max() + 1) * W / mw, (ys.max() + 1) * H / mh)

    def track_many(self, frames_rgb: List[np.ndarray], jobs: Sequence[Job]) -> List[List[Dict[int, Optional[Box]]]]:
        """
        한 덩어리 프레임에서 여러 사건 (시작, 끝, 프롬프트) 을 추적. 결과 [사건][사건 시작 기준 프레임] = {객체: 박스}

        영상 전처리와 프레임 영상 특징(ViT)은 덩어리에서 한 번만. 같은 시작 프레임의 사건들은 객체를 한 세션에
        함께 프롬프트해 한 번에 전파하고, 시작 프레임이 다르면 추적 상태만 지우고(reset_tracking_data) 다시 전파한다.
        """
        import torch
        if self.model is None:
            self._load()
        H, W = frames_rgb[0].shape[:2]
        L = len(frames_rgb)
        out: List[List[Dict[int, Optional[Box]]]] = [[dict() for _ in range(min(e, L - 1) - s + 1)]
                                                    for s, e, _ in jobs]
        sess = self.processor.init_video_session(video=frames_rgb, inference_device=self.device,
                                                 video_storage_device='cpu', dtype=self.dtype)
        sess.cache = _FrameFeatureCache(self.device)
        by_start: Dict[int, List[int]] = {}
        for j, (s, _, _) in enumerate(jobs):
            by_start.setdefault(int(s), []).append(j)
        try:
            with torch.inference_mode():
                for s, js in sorted(by_start.items()):
                    sess.reset_tracking_data()
                    owner: Dict[int, Tuple[int, int]] = {}       # 세션 객체 번호 → (사건, 원래 객체 ID)
                    boxes = []
                    for j in js:
                        for oid, b in jobs[j][2].items():
                            owner[len(owner) + 1] = (j, int(oid))
                            boxes.append([float(v) for v in b])
                    ids = list(owner)
                    end = max(min(int(jobs[j][1]), L - 1) for j in js)
                    self.processor.add_inputs_to_inference_session(sess, frame_idx=s, obj_ids=ids,
                                                                   input_boxes=[boxes])
                    self.model(inference_session=sess, frame_idx=s)
                    for res in self.model.propagate_in_video_iterator(sess, start_frame_idx=s,
                                                                      max_frame_num_to_track=end - s):
                        k = int(res.frame_idx)
                        masks = (res.pred_masks[:, 0] > 0).cpu().numpy()
                        for m, oid in enumerate(res.object_ids if hasattr(res, 'object_ids') else ids):
                            j, orig = owner[int(oid)]
                            if s <= k <= min(int(jobs[j][1]), L - 1):
                                out[j][k - s][orig] = self._mask_box(masks[m], W, H)
        finally:
            sess.cache.clear_all()
        return out

    def unload(self):
        self.model = self.processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def read_frames(video_path, start: int, end: int) -> List[np.ndarray]:
    """[start, end] 프레임 (RGB)"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    try:
        for _ in range(start, end + 1):
            ok, f = cap.read()
            if not ok:
                break
            out.append(np.ascontiguousarray(f[:, :, ::-1]))
    finally:
        cap.release()
    return out


# ---------------------------------------------------------------------------
# 바로잡기
# ---------------------------------------------------------------------------
def apply_swap(frames: List[List[list]], a: int, b: int, start: int):
    """start 프레임부터 끝까지 선수 트랙 a ↔ b ID 교체"""
    for objs in frames[start:]:
        for o in objs or []:
            if o[1] == KIND_PLAYER:
                if o[0] == a:
                    o[0] = b
                elif o[0] == b:
                    o[0] = a


def apply_links(frames: List[List[list]], parent: Dict[int, int]):
    for objs in frames:
        for o in objs or []:
            if o[1] == KIND_PLAYER and int(o[0]) in parent:
                o[0] = parent[int(o[0])]


def _root(parent: Dict[int, int], t: int) -> int:
    while t in parent:
        t = parent[t]
    return t


def _precompute(video_path, tracker, events: Sequence[Optional[Job]], max_span: int,
                report: Callable[[int], None] = lambda k: None) -> Tuple[List, int]:
    """
    사건 [(시작, 끝, 프롬프트) 또는 None, ...] (영상 프레임 번호) → 사건별 track 결과 (None: 건너뜀·프레임 부족)
    시간이 겹치는 사건을 덩어리로 묶어 덩어리마다 영상을 한 번 읽고 track_many 한 번. Returns: (결과, 덩어리 수)
    """
    idx = [i for i, ev in enumerate(events) if ev is not None]
    groups = merge_windows([events[i][:2] for i in idx], max_span)
    out: List = [None] * len(events)
    finished = 0
    for gs, ge, members in groups:
        clip = read_frames(video_path, gs, ge)
        jobs = [(events[idx[m]][0] - gs, events[idx[m]][1] - gs, events[idx[m]][2]) for m in members]
        for m, r in zip(members, track_jobs(tracker, clip, jobs)):
            out[idx[m]] = r
        finished += len(members)
        report(finished)
    return out, len(groups)


def repair_tracks(video_path, frames: List[List[list]], size: Tuple[int, int], tracker, fps: float = 30.0,
                  window_s: float = 1.0, link_iou: float = 0.5, swap_margin: float = 0.2, swap_frames: int = 5,
                  swap_ratio: float = 0.8,
                  max_events: int = 150, batch: bool = True, max_span: int = 120,
                  progress: Callable[[int, int], None] = lambda d, t: None
                  ) -> Tuple[Dict[int, int], List[Tuple[int, int, int]], Dict]:
    """
    frames 를 제자리에서 고친다 (끊긴 트랙 잇기 → ID 교체 바로잡기).

    Args:
        tracker: Sam3VideoTracker 또는 같은 인터페이스 (track(frames_rgb, prompts) -> 프레임별 {obj: box}).
                 track_many 가 있고 batch=True 면 시간이 겹치는 사건을 max_span 프레임 이하 덩어리로 묶어
                 한 세션에서 처리한다 (결과는 하나씩 처리할 때와 같음)
    Returns:
        ({child: root} 이은 트랙, [(A, B, 교체 시작 프레임), ...], 요약)
    """
    t0 = time.perf_counter()
    n = len(frames)
    window = max(5, int(round(window_s * fps)))
    lead = 8
    batch = bool(batch) and hasattr(tracker, 'track_many')
    tracks = player_tracks(frames)
    before = len(tracks)

    # 1. 끊김 ---------------------------------------------------------------
    breaks = break_events(tracks, n, size, window)
    overlaps = overlap_events(tracks)
    total = min(len(breaks) + len(overlaps), max_events)
    done = 0
    groups = recomputed = 0
    parent: Dict[int, int] = {}
    taken = set()
    scores = []
    todo = breaks[:max_events]
    if batch:
        pre, g = _precompute(video_path, tracker,
                             [(e, min(n - 1, e + window), {a: tracks[a][e]}) for a, e, _ in todo], max_span,
                             lambda k: progress(min(k, total), total))
        groups += g
    for i, (a, e, cands) in enumerate(todo):
        if batch:
            masks = pre[i]
            if masks is None:
                continue
        else:
            end = min(n - 1, e + window)
            clip = read_frames(video_path, e, end)
            if len(clip) < 2:
                continue
            masks = tracker.track(clip, {a: tracks[a][e]})
        best, best_iou = None, link_iou
        for b in cands:
            if b in taken:
                continue
            s = min(tracks[b])
            mb = masks[s - e].get(a) if s - e < len(masks) else None
            if mb is None:
                continue
            iou = box_iou(mb, tracks[b][s])
            if iou >= best_iou:
                best, best_iou = b, iou
        if best is not None:
            parent[best] = a
            taken.add(best)
            scores.append(round(best_iou, 3))
        done += 1
        if not batch:
            progress(done, total)
    # 사슬(A ← B ← C)을 대표 ID 로 정리
    parent = {c: _root(parent, p) for c, p in parent.items()}
    if parent:
        apply_links(frames, parent)
        tracks = player_tracks(frames)
        overlaps = overlap_events(tracks)

    # 2. ID 교체 -------------------------------------------------------------
    swaps: List[Tuple[int, int, int]] = []
    evidence: List[Dict] = []
    todo = overlaps[:max(0, max_events - done)]
    if batch:
        # 바로잡기 전 트랙 기준 프롬프트로 미리 전파. 앞 사건의 교체로 프롬프트가 바뀌면 아래에서 그 사건만 다시 전파
        planned: List[Optional[Job]] = []
        for a, b, s, e in todo:
            start, fa, fb = s - lead, tracks.get(a, {}), tracks.get(b, {})
            ok = start in fa and start in fb
            planned.append((start, min(n - 1, e + window // 2), {a: fa[start], b: fb[start]}) if ok else None)
        base = done
        pre, g = _precompute(video_path, tracker, planned, max_span,
                             lambda k: progress(min(base + k, total), total))
        groups += g
    for i, (a, b, s, e) in enumerate(todo):
        start, end = s - lead, min(n - 1, e + window // 2)
        fa, fb = tracks.get(a, {}), tracks.get(b, {})
        if start not in fa or start not in fb:
            continue
        prompts = {a: fa[start], b: fb[start]}
        if batch and planned[i] is not None and _same_prompts(prompts, planned[i][2]):
            masks = pre[i] or []
        else:
            if batch:
                recomputed += 1
            masks = tracker.track(read_frames(video_path, start, end), prompts)
        crossed, compared, diffs = 0, 0, []
        for t in range(e + 1, end + 1):
            m = masks[t - start] if t - start < len(masks) else {}
            ma, mb = m.get(a), m.get(b)
            if ma is None or mb is None or t not in fa or t not in fb:
                continue
            same = box_iou(fa[t], ma) + box_iou(fb[t], mb)
            cross = box_iou(fa[t], mb) + box_iou(fb[t], ma)
            compared += 1
            diffs.append(cross - same)
            crossed += cross > same + 2 * swap_margin
        # 떨어진 뒤 비교한 프레임 대부분(swap_ratio)에서, swap_frames 이상 엇갈려야 교체 (잠깐의 마스크 흔들림 제외)
        if compared and crossed >= swap_frames and crossed >= swap_ratio * compared:
            at = (s + e) // 2 + 1                          # 겹친 구간의 가운데부터
            apply_swap(frames, a, b, at)
            swaps.append((a, b, at))
            evidence.append({'pair': [a, b], 'frame': at, 'crossed': int(crossed), 'compared': compared,
                             'mean_diff': round(float(np.mean(diffs)), 2)})
            tracks = player_tracks(frames)
        done += 1
        if not batch:
            progress(done, total)
    if batch:
        progress(total, total)

    after = len(player_tracks(frames))
    info = {'break_events': len(breaks), 'overlap_events': len(overlaps), 'linked': len(parent),
            'swapped': len(swaps), 'tracks_before': before, 'tracks_after': after,
            'link_iou': scores, 'swap_evidence': evidence,
            'batched': batch, 'batches': groups, 'recomputed': recomputed,
            'seconds': round(time.perf_counter() - t0, 1)}
    return parent, swaps, info
