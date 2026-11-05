"""
등번호 인식 · 신원 부여 (9부, 3부 21.4절)

  1. 크롭 고르기 (분석 중)  : 트랙마다 일정 간격으로 선수 박스 상단(등·가슴, 높이 12~60 %)을 잘라
                             크기(박스 높이 ≥ 80 px)·선명도(라플라시안 분산)·다른 박스와 겹치지 않음을 만족하는 것만,
                             트랙당 점수가 높은 K 장만 보관 (JerseyCropBank)
  2. 읽기                  : GLM-OCR(zai-org/GLM-OCR, MIT) "Text Recognition:" → 숫자 1~99 만 인정,
                             생성 토큰 확률의 곱을 신뢰도로. '1' 은 골키퍼 트랙에서만 인정 (번호 없는 셔츠의 흔한 오독)
  3. 트랙별 투표            : 신뢰도 가중 투표. 1 등 번호가 2 번 이상·전체 판독의 25 % 이상 읽히고 유효 판독 점수의
                             50 % 이상이며 평균 신뢰도가 충분해야 확정, 아니면 미확정(None)
  4. 중복 정리              : 같은 팀·같은 번호 트랙이 같은 시간에 둘 이상이면 신뢰도 낮은 쪽을 미확정으로
  5. 트랙 재연결            : 같은 팀·같은 번호이고 시간상 겹치지 않는 트랙은 같은 선수 → 가장 먼저 나온 ID 로 합침

번호가 보이는 프레임은 와이드 중계 화면에서 드물다는 전제로, 프레임 하나의 판독은 믿지 않고 트랙 전체로 판단한다.
"""

import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

KIND_PLAYER = 0
DIGITS = re.compile(r'^\s*#?\s*(\d{1,2})\s*$')


# ---------------------------------------------------------------------------
# 1. 크롭
# ---------------------------------------------------------------------------
def jersey_crop(frame: np.ndarray, bbox: Sequence[int], others: Sequence[Sequence[int]] = (),
                min_height: int = 80, min_sharpness: float = 60.0) -> Optional[Tuple[np.ndarray, float]]:
    """
    선수 박스 → (등·가슴 크롭, 품질 점수) 또는 None

    - 박스 높이 min_height px 미만(멀리 있는 선수)은 번호가 안 읽히므로 버림
    - 다른 선수 박스와 조금이라도 겹치면 버림 (번호가 가려지거나 다른 선수 번호가 섞임)
    - 화면 가장자리에 잘린 박스, 흐린 크롭(라플라시안 분산 < min_sharpness) 버림
    품질 = 선명도 × 박스 높이 (클수록 잘 읽힘)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    H, W = frame.shape[:2]
    if h < min_height or w < 0.2 * h or x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2:
        return None
    for o in others:
        ox1, oy1, ox2, oy2 = o[:4]
        if (ox1, oy1, ox2, oy2) == (x1, y1, x2, y2):
            continue
        if ox1 < x2 and ox2 > x1 and oy1 < y2 and oy2 > y1:
            return None
    crop = frame[y1 + int(0.12 * h):y1 + int(0.6 * h), x1:x2]
    if crop.size == 0:
        return None
    sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharp < min_sharpness:
        return None
    return np.ascontiguousarray(crop), sharp * h


class JerseyCropBank:
    """트랙별로 품질이 높은 크롭 K 장만 보관 (메모리 상한: 트랙 수 × K)"""

    def __init__(self, per_track: int = 12):
        self.per_track = per_track
        self.items: Dict[int, List[Tuple[float, int, np.ndarray]]] = {}

    def add(self, track_id: int, frame_idx: int, crop: np.ndarray, quality: float):
        items = self.items.setdefault(track_id, [])
        items.append((quality, frame_idx, crop))
        if len(items) > self.per_track:
            items.sort(key=lambda x: -x[0])
            del items[self.per_track:]

    def merge(self, parent: Dict[int, int]):
        """트랙 잇기(child → root) 결과를 반영"""
        for child, root in parent.items():
            for q, t, c in self.items.pop(child, []):
                self.add(root, t, c, q)

    def __len__(self):
        return sum(len(v) for v in self.items.values())


# ---------------------------------------------------------------------------
# 2. OCR
# ---------------------------------------------------------------------------
def parse_number(text: str) -> Optional[int]:
    """OCR 출력 → 등번호 1~99 (숫자 1~2 자리만, 그 밖은 None)"""
    m = DIGITS.match(text or '')
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 99 else None


class GlmOcrReader:
    """
    GLM-OCR (Hugging Face transformers, GlmOcrForConditionalGeneration). 모델은 처음 호출할 때 로드.
    __call__(crops) -> [(텍스트, 신뢰도), ...]   신뢰도 = 생성 토큰 최대 확률의 곱
    """

    def __init__(self, model_id: str = 'zai-org/GLM-OCR', device: str = 'cuda', upscale: float = 3.0,
                 batch_size: int = 8, max_new_tokens: int = 6):
        self.model_id, self.device = model_id, device
        self.upscale, self.batch_size, self.max_new_tokens = upscale, batch_size, max_new_tokens
        self.model = self.processor = None

    def _load(self):
        import torch
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration
        dtype = torch.bfloat16 if str(self.device).startswith('cuda') else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = GlmOcrForConditionalGeneration.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()

    def _prepare(self, crop: np.ndarray):
        from PIL import Image
        big = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.ascontiguousarray(big[:, :, ::-1]))

    def _read_one(self, crop: np.ndarray) -> Tuple[str, float]:
        import torch
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': self._prepare(crop)},
                                             {'type': 'text', 'text': 'Text Recognition:'}]}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0][inputs['input_ids'].shape[1]:]
        text = self.processor.decode(seq, skip_special_tokens=True).strip()
        probs = [float(torch.softmax(s[0].float(), -1).max()) for s in out.scores]
        # 끝 토큰(EOS 등)까지 포함한 확률의 곱 — 짧고 확실한 출력일수록 높음
        return text, float(np.prod(probs)) if probs else 0.0

    def __call__(self, crops: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        if self.model is None:
            self._load()
        return [self._read_one(c) for c in crops]

    def unload(self):
        self.model = self.processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 3. 투표
# ---------------------------------------------------------------------------
def vote(readings: Sequence[Tuple[Optional[int], float]], min_count: int = 2, min_share: float = 0.5,
         min_conf: float = 0.5, min_frac: float = 0.25) -> Tuple[Optional[int], float, int]:
    """
    readings: [(번호 또는 None, 신뢰도), ...]  →  (확정 번호 또는 None, 신뢰도 0~1, 근거 판독 수)
    신뢰도 = (1 등 번호 점수 / 유효 판독 점수 합) × 1 등 판독 평균 신뢰도
    확정 조건: 1 등 번호 판독 min_count 회 이상, 유효 판독 점수의 min_share 이상, 평균 신뢰도 min_conf 이상,
              전체 판독(숫자가 아닌 것 포함)의 min_frac 이상 (크롭 대부분이 읽히지 않았는데 우연히 두 번 맞은 경우 제외)
    """
    score: Dict[int, float] = {}
    count: Dict[int, int] = {}
    for n, c in readings:
        if n is None:
            continue
        score[n] = score.get(n, 0.0) + c
        count[n] = count.get(n, 0) + 1
    if not score:
        return None, 0.0, 0
    best = max(score, key=score.get)
    share = score[best] / sum(score.values())
    mean = score[best] / count[best]
    conf = round(share * mean, 3)
    if count[best] < min_count or share < min_share or mean < min_conf or count[best] < min_frac * len(readings):
        return None, conf, count[best]
    return best, conf, count[best]


# ---------------------------------------------------------------------------
# 4·5. 중복 정리 · 재연결
# ---------------------------------------------------------------------------
def track_spans(frames: List[List[list]]) -> Dict[int, Tuple[int, int, set]]:
    """선수 트랙 ID → (첫 프레임, 마지막 프레임, 프레임 집합)"""
    spans: Dict[int, List[int]] = {}
    for t, objs in enumerate(frames):
        for o in objs or []:
            if o[1] == KIND_PLAYER:
                spans.setdefault(int(o[0]), []).append(t)
    return {tid: (ts[0], ts[-1], set(ts)) for tid, ts in spans.items()}


def resolve_duplicates(ids: Dict[int, dict], teams: Dict[int, str], spans) -> int:
    """같은 팀·번호 트랙이 같은 프레임에 함께 나오면 신뢰도 낮은 쪽을 미확정으로. 미확정으로 돌린 수"""
    changed = 0
    groups: Dict[Tuple[str, int], List[int]] = {}
    for tid, info in ids.items():
        if info['jersey_number'] is not None and teams.get(tid) in ('home', 'away'):
            groups.setdefault((teams[tid], info['jersey_number']), []).append(tid)
    for members in groups.values():
        members.sort(key=lambda t: -ids[t]['confidence'])
        kept: List[int] = []
        for tid in members:
            if any(spans[tid][2] & spans[k][2] for k in kept):
                ids[tid]['jersey_number'] = None
                ids[tid]['conflict'] = True
                changed += 1
            else:
                kept.append(tid)
    return changed


def relink(ids: Dict[int, dict], teams: Dict[int, str], spans) -> Dict[int, int]:
    """같은 팀·번호이고 시간상 겹치지 않는 트랙 → {child: root} (root = 가장 먼저 시작한 트랙)"""
    groups: Dict[Tuple[str, int], List[int]] = {}
    for tid, info in ids.items():
        if info['jersey_number'] is not None and teams.get(tid) in ('home', 'away'):
            groups.setdefault((teams[tid], info['jersey_number']), []).append(tid)
    parent = {}
    for members in groups.values():
        members.sort(key=lambda t: spans[t][0])
        root = members[0]
        for tid in members[1:]:
            parent[tid] = root
    return parent


def apply_relink(frames: List[List[list]], parent: Dict[int, int]):
    for objs in frames:
        for o in objs or []:
            if o[1] == KIND_PLAYER and int(o[0]) in parent:
                o[0] = parent[int(o[0])]


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------
def identify(bank: JerseyCropBank, reader: Callable[[Sequence[np.ndarray]], List[Tuple[str, float]]],
             frames: List[List[list]], teams: Dict[int, str], exclude: Sequence[int] = (),
             goalkeepers: Sequence[int] = (), relink_tracks: bool = True
             ) -> Tuple[Dict[int, dict], Dict[int, int], Dict]:
    """
    Args:
        bank: 분석 중 모은 크롭, reader: OCR (GlmOcrReader 또는 같은 인터페이스)
        frames: 분석 결과 frames (relink 하면 제자리에서 ID 를 바꾼다)
        teams: 트랙 ID → home | away | other (심판 등 other 는 번호를 읽어도 재연결하지 않음)
        exclude: 읽지 않을 트랙 (심판 등)
        goalkeepers: 골키퍼 트랙. 번호 없는 셔츠(심판·옆모습)를 OCR 이 '1' 로 읽는 일이 가장 흔하고, 1 번은 거의 항상
                     골키퍼이므로 골키퍼가 아닌 트랙의 '1' 판독은 버린다 (진짜 번호의 득표율을 깎지 않도록)
    Returns:
        (트랙 ID → {jersey_number, confidence, evidence, readings}, 재연결 {child: root}, 요약)
    """
    t0 = time.perf_counter()
    order, crops = [], []
    for tid, items in bank.items.items():
        if tid in exclude:
            continue
        for q, t, c in items:
            order.append((tid, t))
            crops.append(c)
    raw = reader(crops) if crops else []
    per: Dict[int, List[Tuple[Optional[int], float]]] = {}
    texts: Dict[int, List[Tuple[int, str, float]]] = {}
    keepers = set(goalkeepers)
    for (tid, t), (text, conf) in zip(order, raw):
        n = parse_number(text)
        if n == 1 and tid not in keepers:
            n = None
        per.setdefault(tid, []).append((n, conf))
        texts.setdefault(tid, []).append((t, text, round(conf, 3)))
    ids: Dict[int, dict] = {}
    for tid, readings in per.items():
        n, conf, evidence = vote(readings)
        ids[tid] = {'jersey_number': n, 'confidence': conf, 'evidence': evidence, 'readings': len(readings)}
    spans = track_spans(frames)
    ids = {tid: v for tid, v in ids.items() if tid in spans}
    conflicts = resolve_duplicates(ids, teams, spans)
    votes = {tid: dict(v) for tid, v in ids.items()}       # 재연결 전 트랙별 결과 (확인·평가용)
    parent = relink(ids, teams, spans) if relink_tracks else {}
    tracks_before = len(spans)
    if parent:
        apply_relink(frames, parent)
        for child, root in parent.items():
            c, r = ids.pop(child), ids[root]
            r['evidence'] += c['evidence']
            r['readings'] += c['readings']
            r['confidence'] = max(r['confidence'], c['confidence'])
            r.setdefault('merged', []).append(child)
    summary = {
        'crops': len(crops), 'read_tracks': len(per),
        'confirmed': sum(v['jersey_number'] is not None for v in ids.values()),
        'conflicts': conflicts, 'relinked': len(parent),
        'tracks_before': tracks_before, 'tracks_after': tracks_before - len(parent),
        'seconds': round(time.perf_counter() - t0, 1),
    }
    return ids, parent, {'summary': summary, 'texts': texts, 'votes': votes}
