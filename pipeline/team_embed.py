"""
SigLIP2 임베딩 기반 팀 분류 (3부 21.3절)

트랙별 선수 크롭 몇 장 → SigLIP2 이미지 임베딩 → UMAP 3차원 → K-Means(k=2) → 트랙 단위 다수결.
평균색 k-means(video_analysis.cluster_teams)보다 조명·그림자·비슷한 색 유니폼에 강하다.
결과 형식(teams, team_colors)은 cluster_teams 와 같아서 프론트엔드·지표 코드는 그대로 쓴다.

임베딩 모델(embedder)은 주입한다: 테스트는 가짜 임베더로, 실제 분석은 SiglipEmbedder(GPU 권장).
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CROP_SIZE = 96              # 저장할 크롭 크기(px). SigLIP 프로세서가 224 로 다시 키운다
MIN_CROPS = 3               # 이보다 적은 트랙은 판정하지 않음('other')
MIN_VOTE = 0.6              # 트랙 내 크롭의 이 비율 이상이 같은 군집이어야 팀 확정
OTHER_LABELS = ('referee',)  # 탐지기가 이 라벨로 낸 트랙은 군집에서 빼고 'other'


def jersey_crop(frame: np.ndarray, bbox: Sequence[int], mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    유니폼이 보이는 상반신~반바지 영역(박스 위 10%~65%) 크롭, 정사각으로 패딩해 CROP_SIZE 로 축소

    mask(박스 크기의 bool, SAM 마스크)를 주면 선수 밖 픽셀(잔디·다른 선수)을 회색으로 지운다.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    H, W = frame.shape[:2]
    top, bottom = y1 + int(h * 0.1), y1 + int(h * 0.65)
    xa, xb, ya, yb = max(0, x1), min(W, x2), max(0, top), min(H, bottom)
    if xb - xa < 4 or yb - ya < 4:
        return None
    crop = frame[ya:yb, xa:xb].copy()
    if mask is not None:
        m = mask[ya - y1:yb - y1, xa - x1:xb - x1]
        if m.shape == crop.shape[:2] and m.mean() > 0.15:
            crop[~m] = 128
    side = max(crop.shape[:2])
    pad = np.full((side, side, 3), 128, np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    pad[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return cv2.resize(pad, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


class SiglipEmbedder:
    """SigLIP2 이미지 임베딩 (Hugging Face transformers). 모델은 처음 호출할 때 로드"""

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str = "cuda",
                 batch_size: int = 64):
        self.model_id, self.device, self.batch_size = model_id, device, batch_size
        self.model = self.processor = None

    def _load(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def __call__(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        import torch
        if self.model is None:
            self._load()
        out = []
        for i in range(0, len(crops_bgr), self.batch_size):
            rgb = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops_bgr[i:i + self.batch_size]]
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            with torch.inference_mode():
                f = self.model.get_image_features(**inputs)
            f = getattr(f, "pooler_output", f)  # 버전에 따라 텐서 또는 출력 객체
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def unload(self):
        """다른 GPU 모델(VLM 등)에 메모리를 넘길 때"""
        self.model = self.processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _reduce(emb: np.ndarray, dims: int = 3, seed: int = 0) -> np.ndarray:
    """UMAP 3차원 축소. 샘플이 적거나 umap 이 없으면 PCA 로 대체"""
    if len(emb) >= 30:
        try:
            import umap
            return umap.UMAP(n_components=dims, n_neighbors=15, min_dist=0.0, metric="cosine",
                             random_state=seed).fit_transform(emb)
        except ImportError:
            pass
    from sklearn.decomposition import PCA
    return PCA(n_components=min(dims, emb.shape[1], len(emb))).fit_transform(emb)


def _team_clusters(low: np.ndarray, emb: np.ndarray, owner: np.ndarray, other_max: float = 0.25):
    """
    K-Means 로 두 팀 군집 찾기 → (크롭별 군집 번호, 군집 중심, {군집 번호: 'home'|'away'})

    심판·골키퍼는 유니폼이 달라 k=2 에서 한 팀으로 뽑혀 두 팀이 한 군집에 섞일 수 있다.
    그래서 k=3 을 먼저 해 보고, 가장 작은 군집이 트랙 기준 other_max 이하이면 그 군집을 'other'
    (심판·골키퍼)로 두고 큰 두 군집을 팀으로 쓴다. 아니면(세 번째 팀 같은 군집이 없으면) k=2.
    """
    from sklearn.cluster import KMeans
    n_tracks = len(set(owner.tolist()))
    for k in (3, 2):
        if len(low) < k * 3 or n_tracks < k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(low)
        labels = km.labels_
        # 군집 크기는 트랙 다수결 기준 (크롭 수가 많은 긴 트랙 하나에 끌려가지 않도록)
        track_label = {t: int(np.bincount(labels[owner == t], minlength=k).argmax()) for t in set(owner.tolist())}
        size = np.bincount(list(track_label.values()), minlength=k)
        order = np.argsort(-size)
        if k == 3 and size[order[2]] > other_max * n_tracks:
            continue
        # 트랙이 많은 군집을 home 으로 (평균색 분류와 같은 규칙)
        names = {int(order[0]): 'home', int(order[1]): 'away'}
        if k == 3:  # 작은 군집이 한 팀 중심에 가까우면(그늘진 같은 팀 등) 그 팀으로 합친다
            # 거리는 원래 임베딩 공간에서 잰다 (UMAP 은 군집 사이의 먼 거리를 보존하지 않음)
            c = np.array([emb[labels == j].mean(axis=0) for j in range(k)])
            small = int(order[2])
            d = [np.linalg.norm(c[small] - c[int(order[i])]) for i in (0, 1)]
            if min(d) < 0.5 * np.linalg.norm(c[int(order[0])] - c[int(order[1])]):
                names[small] = names[int(order[int(np.argmin(d))])]
        return labels, km.cluster_centers_, names
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(low)
    return km.labels_, km.cluster_centers_, {0: 'home', 1: 'away'}


def _hex(lab: np.ndarray) -> str:
    bgr = cv2.cvtColor(np.uint8([[np.clip(lab, 0, 255)]]), cv2.COLOR_LAB2BGR)[0, 0]
    return '#{:02x}{:02x}{:02x}'.format(int(bgr[2]), int(bgr[1]), int(bgr[0]))


def classify_teams(track_crops: Dict[int, List[np.ndarray]],
                   embedder: Callable[[List[np.ndarray]], np.ndarray],
                   track_colors: Optional[Dict[int, List[np.ndarray]]] = None,
                   track_labels: Optional[Dict[int, Dict[str, int]]] = None,
                   outlier_k: float = 3.0) -> Tuple[Dict[int, str], Dict[str, str], Dict]:
    """
    Args:
        track_crops: 트랙 ID → jersey_crop 크롭 목록
        embedder: 크롭 목록 → (N, D) 임베딩
        track_colors: 트랙 ID → 상체 Lab 색 샘플 (팀 대표색 표시용, 없으면 크롭 평균색)
        track_labels: 트랙 ID → {탐지 라벨: 횟수}. 'referee' 가 다수인 트랙은 'other'
        outlier_k: 트랙 중심이 군집 중심에서 (중앙값 × k) 이상 멀면 'other' (심판·골키퍼 등)

    Returns:
        (teams {트랙 ID: home|away|other}, team_colors {home|away: '#rrggbb'}, info)
    """
    teams = {tid: 'other' for tid in track_crops}
    referee = {tid for tid, votes in (track_labels or {}).items()
               if votes and max(votes, key=votes.get) in OTHER_LABELS}
    ids = [tid for tid, cs in track_crops.items() if len(cs) >= MIN_CROPS and tid not in referee]
    info = {'method': 'siglip', 'tracks': len(ids), 'referee_tracks': len(referee)}
    if len(ids) < 2:
        return teams, {}, info

    crops, owner = [], []
    for tid in ids:
        crops += track_crops[tid]
        owner += [tid] * len(track_crops[tid])
    owner = np.array(owner)
    emb = np.asarray(embedder(crops), np.float32)
    low = _reduce(emb)

    labels, centers, names = _team_clusters(low, emb, owner)

    # 크롭마다 팀 이름(합쳐진 군집은 같은 이름, 이름 없는 군집은 'other')으로 바꿔 트랙 단위 다수결
    crop_team = np.array([names.get(int(k), 'other') for k in labels])
    votes, dist = {}, {}
    for tid in ids:
        sel = owner == tid
        team, n = max(((t, int(np.sum(crop_team[sel] == t))) for t in ('home', 'away', 'other')), key=lambda x: x[1])
        votes[tid] = (team, n / sel.sum())
        own = [j for j in range(len(centers)) if names.get(j, 'other') == team]
        dist[tid] = float(min(np.linalg.norm(np.median(low[sel], axis=0) - centers[j]) for j in own))
    cutoff = outlier_k * float(np.median(list(dist.values()))) + 1e-6
    for tid in ids:
        team, frac = votes[tid]
        if team != 'other' and frac >= MIN_VOTE and dist[tid] <= cutoff:
            teams[tid] = team
    info.update(outliers=int(sum(dist[t] > cutoff for t in ids)), low_vote=int(sum(votes[t][1] < MIN_VOTE for t in ids)))

    colors = {}
    for name in ('home', 'away'):
        members = [t for t in ids if teams[t] == name]
        samples = [c for t in members for c in (track_colors or {}).get(t, [])]
        if samples:
            colors[name] = _hex(np.median(samples, axis=0))
        elif members:
            mean_bgr = np.mean([c.reshape(-1, 3).mean(axis=0) for t in members for c in track_crops[t]], axis=0)
            colors[name] = '#{:02x}{:02x}{:02x}'.format(int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
    return teams, colors, info
