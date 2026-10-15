"""
RF-DETR 탐지 백엔드 (3부 21.1절, GPU 권장)

YOLO 탐지기(RoboflowSoccerDetector)와 같은 인터페이스: detect(frame) -> List[Detection],
추론 설정 imgsz / ball_imgsz / ball_tile 은 분석 모드 프로필이 실행 중에 바꾼다.

- 사전학습(COCO) 가중치: person → 선수, sports ball → 공 (나머지 클래스는 버림)
- 파인튜닝 가중치(checkpoint): 클래스 이름이 player / goalkeeper / referee / ball 이면 그대로 라벨로 쓴다
  (scripts/train_rfdetr.py 로 Roboflow 축구 데이터셋을 학습)
한 모델이 선수와 공을 함께 내므로 YOLO 처럼 모델을 두 번 돌리지 않고, 공만 타일 추론을 추가한다.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import BALL_OFFSET, Detection, merge_detections, tiles

SIZES = ('nano', 'small', 'medium', 'base', 'large')
# 파인튜닝 클래스 이름 → (탐지 종류, 라벨). 그 밖의 이름은 버린다
FOOTBALL_CLASSES = {
    'person': ('player', 'person'), 'player': ('player', 'player'),
    'goalkeeper': ('player', 'goalkeeper'), 'referee': ('player', 'referee'),
    'sports ball': ('ball', 'ball'), 'ball': ('ball', 'ball'), 'football': ('ball', 'ball'),
}
PLAYER_LABEL_IDS = {'person': 0, 'player': 0, 'goalkeeper': 1, 'referee': 2}


def _coco_ids() -> Dict[int, str]:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
        return dict(COCO_CLASSES)
    except ImportError:
        return {}


def _round_to(v: int, m: int) -> int:
    return max(m, int(round(v / m)) * m)


class RFDETRDetector:
    """RF-DETR(rfdetr 패키지) 기반 축구 탐지기"""

    def __init__(self, size: str = 'medium', checkpoint: Optional[str] = None, device: str = 'cuda',
                 confidence_threshold: float = 0.4, ball_threshold: float = 0.25, iou_threshold: float = 0.45,
                 imgsz: Optional[int] = None, ball_imgsz: Optional[int] = None, ball_tile: Optional[int] = None,
                 tile_overlap: float = 0.2, model=None):
        """
        Args:
            size: 'nano' | 'small' | 'medium' | 'base' | 'large' (Apache 2.0 크기만)
            checkpoint: 파인튜닝한 가중치(.pth). None 이면 COCO 사전학습 가중치
            imgsz: 추론 해상도 (None 이면 모델 기본값). 패치 크기 배수로 맞춘다
            ball_imgsz: 공을 위해 한 번 더 돌릴 전체 화면 해상도 (None 이거나 imgsz 와 같으면 생략)
            ball_tile: 공 타일 추론 크기(px, 원본 해상도). None 이면 끔
            model: 테스트용 모델 주입 (predict(images, threshold, shape) -> supervision.Detections)
        """
        self.confidence_threshold = confidence_threshold
        self.ball_threshold = ball_threshold
        self.iou_threshold = iou_threshold
        self.imgsz, self.ball_imgsz, self.ball_tile, self.tile_overlap = imgsz, ball_imgsz, ball_tile, tile_overlap
        self.device = device
        self.model = model if model is not None else self._build(size, checkpoint, device)
        self.class_names = self._class_names()
        self.multiple = self._multiple()
        print(f"✅ RF-DETR 탐지기 초기화 완료 (크기 {size}, 클래스 {sorted(set(self.class_names.values()))})")

    @staticmethod
    def _build(size: str, checkpoint: Optional[str], device: str):
        import rfdetr
        if size not in SIZES:
            raise ValueError(f"RF-DETR 크기는 {SIZES} 중 하나입니다: {size}")
        cls = getattr(rfdetr, f"RFDETR{size.capitalize()}")
        kwargs = {'device': device}
        if checkpoint:
            from .detector import resolve_path
            kwargs['pretrain_weights'] = str(resolve_path(checkpoint))
        model = cls(**kwargs)
        try:
            model.optimize_for_inference()  # 가능하면 추론 그래프 최적화 (실패해도 그대로 사용)
        except Exception:
            pass
        return model

    def _class_names(self) -> Dict[int, str]:
        """모델 class_id → 축구 클래스 이름 (FOOTBALL_CLASSES 에 있는 것만)"""
        names = getattr(self.model, 'class_names', None) or {}
        if isinstance(names, (list, tuple)):
            coco = _coco_ids()
            # COCO 사전학습 모델은 class_names 가 80 개 목록이지만 예측 class_id 는 COCO 원래 ID(1=person, 37=sports ball)
            names = coco if coco and list(names) == list(coco.values()) else dict(enumerate(names))
        return {int(k): str(v).lower() for k, v in names.items() if str(v).lower() in FOOTBALL_CLASSES}

    def _multiple(self) -> int:
        cfg = getattr(self.model, 'model_config', None)
        return int(getattr(cfg, 'patch_size', 16) or 16) * int(getattr(cfg, 'num_windows', 2) or 2)

    def _shape(self, size: Optional[int]) -> Optional[Tuple[int, int]]:
        if not size:
            return None
        s = _round_to(int(size), self.multiple)
        return (s, s)

    def _predict(self, images: List[np.ndarray], size: Optional[int], threshold: float):
        rgb = [np.ascontiguousarray(im[:, :, ::-1]) for im in images]
        kwargs = {'threshold': threshold, 'include_source_image': False}
        shape = self._shape(size)
        if shape:
            kwargs['shape'] = shape
        res = self.model.predict(rgb if len(rgb) > 1 else rgb[0], **kwargs)
        return res if isinstance(res, list) else [res]

    def _to_detections(self, dets, x0: int = 0, y0: int = 0, want: Tuple[str, ...] = ('player', 'ball'),
                       min_conf: Optional[Dict[str, float]] = None) -> List[Detection]:
        out = []
        if dets is None or len(dets) == 0:
            return out
        for (x1, y1, x2, y2), conf, cid in zip(dets.xyxy.tolist(), dets.confidence.tolist(), dets.class_id.tolist()):
            name = self.class_names.get(int(cid))
            if name is None:
                continue
            kind, label = FOOTBALL_CLASSES[name]
            if kind not in want or (min_conf and conf < min_conf.get(kind, 0.0)):
                continue
            box = (int(x1) + x0, int(y1) + y0, int(x2) + x0, int(y2) + y0)
            if kind == 'ball':
                out.append(Detection(box, float(conf), BALL_OFFSET, 'ball_ball', 'ball'))
            else:
                out.append(Detection(box, float(conf), PLAYER_LABEL_IDS[label], f"player_{label}", 'player'))
        return out

    def detect(self, frame: np.ndarray, detect_players: bool = True, detect_ball: bool = True,
               detect_field: bool = False) -> List[Detection]:
        low = min(self.confidence_threshold, self.ball_threshold)
        thresholds = {'player': self.confidence_threshold, 'ball': self.ball_threshold}
        want = tuple(k for k, on in (('player', detect_players), ('ball', detect_ball)) if on)
        dets = self._to_detections(self._predict([frame], self.imgsz, low)[0], want=want, min_conf=thresholds)
        if not detect_ball:
            return dets
        players = [d for d in dets if not d.is_ball]
        balls = [d for d in dets if d.is_ball]
        if self.ball_imgsz and self.ball_imgsz != self.imgsz:
            balls += self._to_detections(self._predict([frame], self.ball_imgsz, self.ball_threshold)[0],
                                         want=('ball',))
        if self.ball_tile:
            boxes = tiles(frame.shape[1], frame.shape[0], self.ball_tile, self.tile_overlap)
            crops = [frame[y:y + t, x:x + t] for x, y, t in boxes]
            for (x, y, t), res in zip(boxes, self._predict(crops, boxes[0][2], self.ball_threshold)):
                balls += self._to_detections(res, x, y, want=('ball',))
        return players + merge_detections(balls, self.iou_threshold)

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        return self.detect(frame)
