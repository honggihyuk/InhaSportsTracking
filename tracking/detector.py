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


def _yolo(path: str):
    # ultralytics(torch) 는 무거우므로 실제 모델을 만들 때만 import
    from ultralytics import YOLO
    return YOLO(path)


class RoboflowSoccerDetector:
    """Roboflow 축구 전문 모델 기반 객체 탐지기"""

    BALL_OFFSET = BALL_OFFSET
    FIELD_OFFSET = FIELD_OFFSET

    def __init__(
        self,
        players_model_path: Optional[str] = None,
        ball_model_path: Optional[str] = None,
        field_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",  # 'cuda' 또는 'cpu'
        imgsz: int = 640,
        ball_imgsz: int = 1280
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
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        self.ball_imgsz = ball_imgsz
        
        # 모델 경로 설정
        base_path = Path("data/roboflow_datasets")
        
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
                self.players_model = _yolo("yolo11s.pt")
        
        # 공 탐지 모델
        if ball_model_path:
            self.ball_model = self._load_model(ball_model_path)
        else:
            default_path = base_path / "football-ball-detection-rejhg" / "weights" / "best.pt"
            if default_path.exists():
                self.ball_model = self._load_model(str(default_path))
            else:
                print("⚠️ 공 탐지 모델이 없습니다. 기본 YOLO11n 을 사용합니다.")
                self.ball_model = _yolo("yolo11n.pt")
        
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
        path = Path(model_path)
        
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
            detections += self._predict(frame, self.ball_model, self.BALL_OFFSET, 'ball',
                                        self.ball_classes, self.ball_imgsz)
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
