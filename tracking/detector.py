"""
객체 탐지 모듈 - YOLO 기반 선수 및 공 탐지
Roboflow Sports 파이프라인을 활용한 객체 탐지 구현
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """탐지된 객체 정보"""
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """바운딩 박스 중심점"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> int:
        """바운딩 박스 면적"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class SoccerDetector:
    """
    축구 경기장 객체 탐지기
    YOLOv8/v10 모델을 사용하여 선수, 공, 심판 등 탐지
    """
    
    # 클래스 매핑 (Roboflow Sports 모델 기준)
    CLASS_MAP = {
        0: 'ball',
        1: 'player', 
        2: 'goalkeeper',
        3: 'referee',
        4: 'assistant_referee'
    }
    
    def __init__(self, model_path: str = 'models/yolo_soccer.pt', 
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = 'cuda'):
        """
        Args:
            model_path: YOLO 모델 파일 경로
            confidence_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            device: 연산 장치 ('cuda' 또는 'cpu')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        
    def load_model(self):
        """모델 로드"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"모델 로드 완료: {self.model_path}")
        except ImportError:
            print("ultralytics 패키지가 설치되어 있지 않습니다.")
            print("pip install ultralytics 로 설치해주세요.")
            raise
        except FileNotFoundError:
            print(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            print("Roboflow 에서 soccer detection 모델을 다운로드받아주세요.")
            raise
            
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        단일 프레임에서 객체 탐지
        
        Args:
            frame: 입력 이미지 (BGR 형식)
            
        Returns:
            Detection 객체 리스트
        """
        if self.model is None:
            self.load_model()
            
        # YOLO 추론
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        result = results[0]
        
        # 결과 처리
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                
                # 좌표 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # 클래스 레이블
                label = self.CLASS_MAP.get(class_id, f'class_{class_id}')
                
                detection = Detection(
                    class_id=class_id,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    label=label
                )
                detections.append(detection)
                
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        배치 단위 객체 탐지
        
        Args:
            frames: 입력 이미지 리스트
            
        Returns:
            프레임별 Detection 객체 리스트
        """
        if self.model is None:
            self.load_model()
            
        all_detections = []
        
        # 배치 추론
        results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        for result in results:
            detections = []
            boxes = result.boxes
            
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = self.CLASS_MAP.get(class_id, f'class_{class_id}')
                    
                    detection = Detection(
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        label=label
                    )
                    detections.append(detection)
                    
            all_detections.append(detections)
            
        return all_detections
    
    def visualize(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        탐지 결과 시각화
        
        Args:
            frame: 입력 이미지
            detections: 탐지 결과 리스트
            
        Returns:
            시각화된 이미지
        """
        vis_frame = frame.copy()
        
        # 색상 맵
        color_map = {
            'ball': (0, 255, 0),      # Green
            'player': (255, 0, 0),    # Blue
            'goalkeeper': (0, 0, 255), # Red
            'referee': (255, 255, 0),  # Cyan
            'assistant_referee': (255, 0, 255)  # Magenta
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.label, (128, 128, 128))
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 레이블 표시
            label_text = f"{det.label}: {det.confidence:.2f}"
            cv2.putText(vis_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        return vis_frame


if __name__ == '__main__':
    # 테스트 코드
    detector = SoccerDetector(confidence_threshold=0.3)
    
    # 테스트 이미지 로드
    test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 탐지 테스트 (모델이 없는 경우 예외 발생)
    try:
        detections = detector.detect(test_image)
        print(f"탐지된 객체 수: {len(detections)}")
        
        # 시각화
        vis_image = detector.visualize(test_image, detections)
        cv2.imwrite('test_detection.png', vis_image)
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        print("실제 사용시에는 학습된 YOLO 모델 파일이 필요합니다.")
