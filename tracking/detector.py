"""
Roboflow 스포츠 전용 YOLO 탐지기

Roboflow 에서 제공하는 축구 전문 데이터셋 (Players, Ball, Field) 을 사용하여
더 정확한 축구 객체 탐지를 수행합니다.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO
import yaml


class RoboflowSoccerDetector:
    """Roboflow 축구 전문 모델 기반 객체 탐지기"""
    
    def __init__(
        self,
        players_model_path: Optional[str] = None,
        ball_model_path: Optional[str] = None,
        field_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"  # 'cuda' 또는 'cpu'
    ):
        """
        Args:
            players_model_path: 선수 탐지 모델 경로 (.pt 또는 data.yaml)
            ball_model_path: 공 탐지 모델 경로
            field_model_path: 필드 탐지 모델 경로
            confidence_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            device: 디바이스 ('cuda', 'cpu', 'mps')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # 모델 경로 설정
        base_path = Path("data/roboflow_datasets")
        
        # 선수 탐지 모델
        if players_model_path:
            self.players_model = self._load_model(players_model_path)
        else:
            # 기본 경로: 다운로드된 Roboflow 데이터셋
            default_path = base_path / "football-players-detection" / "weights" / "best.pt"
            if default_path.exists():
                self.players_model = self._load_model(str(default_path))
            else:
                print("⚠️ 선수 탐지 모델이 없습니다. 기본 YOLO11s 를 사용합니다.")
                self.players_model = YOLO("yolo11s.pt")
        
        # 공 탐지 모델
        if ball_model_path:
            self.ball_model = self._load_model(ball_model_path)
        else:
            default_path = base_path / "football-ball-detection" / "weights" / "best.pt"
            if default_path.exists():
                self.ball_model = self._load_model(str(default_path))
            else:
                print("⚠️ 공 탐지 모델이 없습니다. 기본 YOLO11n 을 사용합니다.")
                self.ball_model = YOLO("yolo11n.pt")
        
        # 필드 탐지 모델 (선택사항)
        self.field_model = None
        if field_model_path:
            self.field_model = self._load_model(field_model_path)
        else:
            default_path = base_path / "football-field-detection" / "weights" / "best.pt"
            if default_path.exists():
                self.field_model = self._load_model(str(default_path))
        
        # 클래스 매핑
        self.class_names = self._get_class_names()
        
        print(f"✅ Roboflow Soccer Detector 초기화 완료")
        print(f"   Device: {device}")
        print(f"   Confidence: {confidence_threshold}")
        print(f"   Classes: {list(self.class_names.keys())}")
    
    def _load_model(self, model_path: str) -> YOLO:
        """모델 로드"""
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        
        # .pt 파일 직접 로드 또는 data.yaml 에서 학습된 모델 로드
        if path.suffix == '.pt':
            model = YOLO(str(path))
        elif path.name == 'data.yaml':
            # data.yaml 이 있는 경우, 해당 디렉토리에서 weights/best.pt 로드
            weights_path = path.parent / "weights" / "best.pt"
            if weights_path.exists():
                model = YOLO(str(weights_path))
            else:
                raise FileNotFoundError(f"weights/best.pt 를 찾을 수 없습니다: {weights_path}")
        else:
            model = YOLO(str(path))
        
        # 장치 설정
        model.to(self.device)
        return model
    
    def _get_class_names(self) -> Dict[int, str]:
        """클래스 이름 매핑 가져오기"""
        class_names = {}
        
        # 선수 클래스
        try:
            players_classes = self.players_model.names
            for idx, name in players_classes.items():
                class_names[idx] = f"player_{name}"
        except:
            class_names[0] = "player"
        
        # 공 클래스 (별도 인덱스 사용)
        ball_offset = 100  # 공은 별도 인덱스 사용
        try:
            ball_classes = self.ball_model.names
            for idx, name in ball_classes.items():
                class_names[ball_offset + idx] = f"ball_{name}"
        except:
            class_names[ball_offset] = "ball"
        
        # 필드 클래스 (별도 인덱스 사용)
        field_offset = 200
        if self.field_model:
            try:
                field_classes = self.field_model.names
                for idx, name in field_classes.items():
                    class_names[field_offset + idx] = f"field_{name}"
            except:
                class_names[field_offset] = "field"
        
        return class_names
    
    def detect(
        self, 
        frame: np.ndarray, 
        detect_players: bool = True,
        detect_ball: bool = True,
        detect_field: bool = False
    ) -> List[Dict]:
        """
        프레임에서 객체 탐지
        
        Args:
            frame: BGR 이미지 (OpenCV 형식)
            detect_players: 선수 탐지 여부
            detect_ball: 공 탐지 여부
            detect_field: 필드 탐지 여부
            
        Returns:
            탐지 결과 리스트
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str,
                    'center': (cx, cy),
                    'area': int
                },
                ...
            ]
        """
        detections = []
        
        # 선수 탐지
        if detect_players:
            results = self.players_model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': self.class_names.get(cls_id, f"class_{cls_id}"),
                        'center': (cx, cy),
                        'area': area,
                        'type': 'player'
                    })
        
        # 공 탐지
        if detect_ball:
            results = self.ball_model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            ball_offset = 100
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0]) + ball_offset  # 오프셋 적용
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': self.class_names.get(cls_id, 'ball'),
                        'center': (cx, cy),
                        'area': area,
                        'type': 'ball'
                    })
        
        # 필드 탐지
        if detect_field and self.field_model:
            results = self.field_model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            field_offset = 200
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0]) + field_offset
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': self.class_names.get(cls_id, 'field'),
                        'center': (cx, cy),
                        'area': area,
                        'type': 'field'
                    })
        
        return detections
    
    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """간소화된 탐지 메서드 (선수 + 공)"""
        return self.detect(frame, detect_players=True, detect_ball=True, detect_field=False)
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
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
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            cls_name = det['class_name']
            det_type = det.get('type', 'player')
            
            # 색상 선택
            color = colors.get(det_type, (0, 255, 255))  # 기본: 노랑
            
            # 바운딩 박스 그리기
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 준비
            labels = []
            if show_class:
                # 클래스명 단순화
                simple_name = cls_name.replace('player_', '').replace('ball_', '')
                labels.append(simple_name)
            if show_confidence:
                labels.append(f"{conf:.2f}")
            
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
