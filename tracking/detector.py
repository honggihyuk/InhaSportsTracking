"""
Roboflow 스포츠 전용 YOLO 탐지기

Roboflow 에서 제공하는 축구 전문 데이터셋 (Players, Ball, Field) 을 사용하여
더 정확한 축구 객체 탐지를 수행합니다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

# 프로젝트 루트 — 설정 파일의 상대 경로(models/…, data/…)는 실행 위치와 관계없이 여기를 기준으로 찾는다
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAYER_MODEL = PROJECT_ROOT / "models" / "yolo11s.pt"
DEFAULT_BALL_MODEL = PROJECT_ROOT / "models" / "yolo11n.pt"


def resolve_path(path) -> Path:
    """상대 경로를 현재 작업 폴더에서 먼저 찾고, 없으면 프로젝트 루트 기준으로 해석"""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


# 모델별 class_id 가 겹치지 않도록 부여하는 오프셋
#   [0, 100): 선수 모델, [100, 200): 공 모델, [200, ...): 필드 모델
BALL_OFFSET = 100
FIELD_OFFSET = 200


@dataclass
class Detection:
    """탐지 결과 (detector → tracker → 좌표 변환 공통 형식)"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 픽셀
    confidence: float
    class_id: int                    # 오프셋이 적용된 ID
    class_name: str                  # 예: "player_goalkeeper", "ball_ball"
    type: str                        # 'player' | 'ball' | 'field' (탐지한 모델)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def is_ball(self) -> bool:
        """공 판정은 class_id 범위로만 한다 (선수 모델의 0번 클래스는 공이 아님)"""
        return BALL_OFFSET <= self.class_id < FIELD_OFFSET

    @property
    def label(self) -> str:
        """표시용 라벨: 공은 'ball', 그 외는 모델 클래스명에서 접두사 제거"""
        if self.is_ball:
            return 'ball'
        return self.class_name.removeprefix(f"{self.type}_")


def _coco_class_ids(model, name: str) -> Optional[List[int]]:
    """COCO 범용 모델이면 name 클래스 ID 목록, 전용 모델(Roboflow 등)이면 None(필터 없음)"""
    names = getattr(model, 'names', None) or {}
    if 'sports ball' not in names.values():
        return None
    return [i for i, n in names.items() if n == name]


def tiles(width: int, height: int, tile: int, overlap: float = 0.2) -> List[Tuple[int, int, int]]:
    """화면을 덮는 겹치는 정사각 타일 (x0, y0, 크기). 마지막 타일은 화면 끝에 맞춘다"""
    tile = int(min(tile, width, height))
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(width - tile, 0) + 1, step))
    ys = list(range(0, max(height - tile, 0) + 1, step))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    return [(x, y, tile) for y in ys for x in xs]


def merge_detections(dets: List[Detection], iou: float = 0.45) -> List[Detection]:
    """타일 경계·전체 화면 추론에서 중복된 박스를 NMS 로 하나만 남김"""
    if len(dets) < 2:
        return dets
    boxes = [[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]] for d in dets]
    keep = cv2.dnn.NMSBoxes(boxes, [d.confidence for d in dets], 0.0, iou)
    return [dets[i] for i in np.array(keep).ravel()]


def _yolo(path: str):
    # ultralytics(torch) 는 무거우므로 실제 모델을 만들 때만 import
    from ultralytics import YOLO
    return YOLO(path)


class RoboflowSoccerDetector:
    """Roboflow 축구 전문 모델 기반 객체 탐지기"""

    BALL_OFFSET = BALL_OFFSET
    FIELD_OFFSET = FIELD_OFFSET
    ball_tile: Optional[int] = None   # 공 타일 추론 크기 (None 이면 끔)
    tile_overlap: float = 0.2

    def __init__(
        self,
        players_model_path: Optional[str] = None,
        ball_model_path: Optional[str] = None,
        field_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",  # 'cuda' 또는 'cpu'
        imgsz: int = 640,
        ball_imgsz: int = 1280,
        ball_tile: Optional[int] = None,
        tile_overlap: float = 0.2
    ):
        """
        Args:
            players_model_path: 선수 탐지 모델 경로 (.pt 또는 data.yaml)
            ball_model_path: 공 탐지 모델 경로
            field_model_path: 필드 탐지 모델 경로
            confidence_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            device: 디바이스 ('cuda', 'cpu', 'mps')
            imgsz: 선수/필드 모델 추론 해상도
            ball_imgsz: 공 모델 추론 해상도 (중계 화면의 공은 수 픽셀이라 더 높게)
            ball_tile: 공 타일 추론 크기(px, 원본 해상도). 지정하면 전체 화면 추론에 더해
                       겹치는 타일을 원본 해상도로 추론해 작은 공을 찾는다 (SAHI 방식, 정밀 모드)
            tile_overlap: 타일 겹침 비율 (경계에 걸친 공을 놓치지 않도록)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        self.ball_imgsz = ball_imgsz
        self.ball_tile = ball_tile
        self.tile_overlap = tile_overlap
        
        # 모델 경로 설정
        base_path = PROJECT_ROOT / "data" / "roboflow_datasets"
        
        # 선수 탐지 모델
        if players_model_path:
            self.players_model = self._load_model(players_model_path)
        else:
            # 기본 경로: 다운로드된 Roboflow 데이터셋
            default_path = base_path / "football-players-detection-3zvbc" / "weights" / "best.pt"
            if default_path.exists():
                self.players_model = self._load_model(str(default_path))
            else:
                print("⚠️ 선수 탐지 모델이 없습니다. 기본 YOLO11s 를 사용합니다.")
                self.players_model = _yolo(str(DEFAULT_PLAYER_MODEL))
        
        # 공 탐지 모델
        if ball_model_path:
            self.ball_model = self._load_model(ball_model_path)
        else:
            default_path = base_path / "football-ball-detection-rejhg" / "weights" / "best.pt"
            if default_path.exists():
                self.ball_model = self._load_model(str(default_path))
            else:
                print("⚠️ 공 탐지 모델이 없습니다. 기본 YOLO11n 을 사용합니다.")
                self.ball_model = _yolo(str(DEFAULT_BALL_MODEL))
        
        # 필드 탐지 모델 (선택사항)
        self.field_model = None
        if field_model_path:
            self.field_model = self._load_model(field_model_path)
        else:
            default_path = base_path / "football-field-detection-f07vi" / "weights" / "best.pt"
            if default_path.exists():
                self.field_model = self._load_model(str(default_path))
        
        # 클래스 매핑
        self.class_names = self._get_class_names()

        # COCO 범용 모델이면 필요한 클래스만 추론 (선수 모델=person, 공 모델=sports ball).
        # 필터가 없으면 공 모델이 사람·의자 등 80 개 클래스를 모두 공 후보로 낸다.
        self.player_classes = _coco_class_ids(self.players_model, 'person')
        self.ball_classes = _coco_class_ids(self.ball_model, 'sports ball')
        
        print(f"✅ Roboflow Soccer Detector 초기화 완료")
        print(f"   Device: {device}")
        print(f"   Confidence: {confidence_threshold}")
        print(f"   Classes: {list(self.class_names.keys())}")
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        path = resolve_path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        
        # .pt 파일 직접 로드 또는 data.yaml 에서 학습된 모델 로드
        if path.suffix == '.pt':
            model = _yolo(str(path))
        elif path.name == 'data.yaml':
            # data.yaml 이 있는 경우, 해당 디렉토리에서 weights/best.pt 로드
            weights_path = path.parent / "weights" / "best.pt"
            if weights_path.exists():
                model = _yolo(str(weights_path))
            else:
                raise FileNotFoundError(f"weights/best.pt 를 찾을 수 없습니다: {weights_path}")
        else:
            model = _yolo(str(path))
        
        # 장치 설정
        model.to(self.device)
        return model
    
    def _get_class_names(self) -> Dict[int, str]:
        """클래스 이름 매핑 가져오기"""
        class_names = {}
        models = [(self.players_model, 0, 'player'),
                  (self.ball_model, self.BALL_OFFSET, 'ball'),
                  (self.field_model, self.FIELD_OFFSET, 'field')]
        for model, offset, prefix in models:
            if model is None:
                continue
            names = getattr(model, 'names', None)
            if isinstance(names, dict):
                class_names.update({offset + idx: f"{prefix}_{name}" for idx, name in names.items()})
            else:
                class_names[offset] = prefix
        return class_names
    
    def detect(
        self, 
        frame: np.ndarray, 
        detect_players: bool = True,
        detect_ball: bool = True,
        detect_field: bool = False
    ) -> List[Detection]:
        """
        프레임에서 객체 탐지

        Args:
            frame: BGR 이미지 (OpenCV 형식)
            detect_players: 선수 탐지 여부
            detect_ball: 공 탐지 여부
            detect_field: 필드 탐지 여부

        Returns:
            Detection 리스트
        """
        detections = []
        if detect_players:
            detections += self._predict(frame, self.players_model, 0, 'player',
                                        self.player_classes, self.imgsz)
        if detect_ball:
            balls = self._predict(frame, self.ball_model, self.BALL_OFFSET, 'ball',
                                  self.ball_classes, self.ball_imgsz)
            if self.ball_tile:
                balls = merge_detections(balls + self._predict_tiled(frame), self.iou_threshold)
            detections += balls
        if detect_field and self.field_model:
            detections += self._predict(frame, self.field_model, self.FIELD_OFFSET, 'field',
                                        None, self.imgsz)
        return detections

    def _predict(self, frame: np.ndarray, model, offset: int, det_type: str,
                 classes: Optional[List[int]] = None, imgsz: int = 640) -> List[Detection]:
        """단일 모델 추론 → Detection 으로 변환 (offset 으로 모델 간 class_id 충돌 방지)"""
        results = model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            imgsz=imgsz,
            verbose=False,
            device=self.device
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            # 박스별 인덱싱 대신 한 번에 리스트로 변환 (GPU→CPU 복사 1회)
            for (x1, y1, x2, y2), conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                cls_id = int(cls) + offset
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, det_type if offset else f"class_{cls_id}"),
                    type=det_type
                ))
        return detections

    def _predict_tiled(self, frame: np.ndarray) -> List[Detection]:
        """겹치는 타일마다 공 모델을 원본 해상도로 추론 → 프레임 좌표로 되돌림"""
        out = []
        for x0, y0, tile in tiles(frame.shape[1], frame.shape[0], self.ball_tile, self.tile_overlap):
            crop = frame[y0:y0 + tile, x0:x0 + tile]
            for d in self._predict(crop, self.ball_model, self.BALL_OFFSET, 'ball', self.ball_classes, tile):
                x1, y1, x2, y2 = d.bbox
                d.bbox = (x1 + x0, y1 + y0, x2 + x0, y2 + y0)
                out.append(d)
        return out

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """간소화된 탐지 메서드 (선수 + 공)"""
        return self.detect(frame, detect_players=True, detect_ball=True, detect_field=False)
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Detection],
        show_confidence: bool = True,
        show_class: bool = True
    ) -> np.ndarray:
        """
        탐지 결과를 프레임에 시각화
        
        Args:
            frame: BGR 이미지
            detections: 탐지 결과 리스트
            show_confidence: 신뢰도 표시 여부
            show_class: 클래스명 표시 여부
            
        Returns:
            시각화된 이미지
        """
        output = frame.copy()
        
        # 타입별 색상 정의
        colors = {
            'player': (0, 255, 0),      # 초록색
            'goalkeeper': (255, 0, 0),   # 파란색
            'referee': (0, 0, 255),      # 빨간색
            'ball': (255, 255, 0),       # 노란색
            'field': (255, 0, 255)       # 마젠타
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # 색상 선택: 세부 라벨(goalkeeper 등) → 모델 타입 → 기본 노랑
            color = colors.get(det.label, colors.get(det.type, (0, 255, 255)))

            # 바운딩 박스 그리기
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # 라벨 준비
            labels = []
            if show_class:
                labels.append(det.label)
            if show_confidence:
                labels.append(f"{det.confidence:.2f}")
            
            label_text = " | ".join(labels)
            
            # 라벨 배경
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            # 라벨 배경 사각형
            cv2.rectangle(
                output,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # 라벨 텍스트
            cv2.putText(
                output,
                label_text,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return output


# 하위 호환성을 위한 기존 SoccerDetector 클래스
class SoccerDetector(RoboflowSoccerDetector):
    """기존 SoccerDetector 와의 호환성을 위한 래퍼"""
    pass
