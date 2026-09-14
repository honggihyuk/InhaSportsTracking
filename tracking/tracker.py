"""
객체 추적 모듈 - ByteTrack/BoT-SORT 기반 선수 및 공 추적
탐지된 객체에 ID 를 할당하여 시간적 일관성 유지
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TrackedObject:
    """추적 중인 객체 정보"""
    track_id: int
    class_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[float, float]
    frame_detected: int
    frames_tracked: int = 1
    
    # 운동량 관련
    velocity: Optional[Tuple[float, float]] = None
    trajectory: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = [self.center]


class SoccerTracker:
    """
    축구 경기장 객체 추적기
    ByteTrack 또는 BoT-SORT 알고리즘을 사용하여 객체 추적
    """
    
    def __init__(self, 
                 tracker_type: str = 'bytetrack',
                 track_threshold: float = 0.3,
                 high_threshold: float = 0.6,
                 low_threshold: float = 0.1,
                 match_threshold: float = 0.8,
                 max_age: int = 30,
                 min_hits: int = 3):
        """
        Args:
            tracker_type: 추적기 타입 ('bytetrack' 또는 'botsort')
            track_threshold: 추적 신뢰도 임계값
            high_threshold: 높은 신뢰도 임계값 (ByteTrack 용)
            low_threshold: 낮은 신뢰도 임계값 (ByteTrack 용)
            match_threshold: 매칭 임계값
            max_age: 추적 손실 후 유지할 최대 프레임 수
            min_hits: 추적으로 인정하기 위한 최소 탐지 횟수
        """
        self.tracker_type = tracker_type
        self.track_threshold = track_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracker = None
        self.frame_count = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        
    def initialize_tracker(self, img_size: Tuple[int, int] = (720, 1280)):
        """
        추적기 초기화
        
        Args:
            img_size: 이미지 크기 (height, width)
        """
        try:
            if self.tracker_type == 'bytetrack':
                from boxmot import BYTETracker
                self.tracker = BYTETracker(
                    track_thresh=self.track_threshold,
                    match_thresh=self.match_threshold,
                    track_buffer=self.max_age,
                    frame_rate=30
                )
            elif self.tracker_type == 'botsort':
                from boxmot import BoTSORT
                self.tracker = BoTSORT(
                    track_thresh=self.track_threshold,
                    match_thresh=self.match_threshold,
                    track_buffer=self.max_age,
                    frame_rate=30
                )
            else:
                raise ValueError(f"지원하지 않는 추적기 타입: {self.tracker_type}")
                
            print(f"추적기 초기화 완료: {self.tracker_type}")
            
        except ImportError as e:
            print("boxmot 패키지가 설치되어 있지 않습니다.")
            print("pip install boxmot 로 설치해주세요.")
            raise
            
    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """
        추적기 업데이트
        
        Args:
            detections: 탐지 결과 리스트 (Detection 객체)
            frame: 현재 프레임 이미지 (선택사항)
            
        Returns:
            TrackedObject 리스트
        """
        if self.tracker is None:
            self.initialize_tracker()
            
        self.frame_count += 1
        
        # 탐지 결과를 tracker 입력 형식으로 변환
        if len(detections) > 0:
            # [x1, y1, x2, y2, confidence, class_id]
            det_array = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                det_array.append([
                    x1, y1, x2, y2,
                    det.confidence,
                    det.class_id
                ])
            det_array = np.array(det_array)
        else:
            det_array = np.empty((0, 6))
            
        # 추적기 업데이트
        tracks = self.tracker.update(det_array, frame)
        
        # 추적 결과 처리
        tracked_objects = []
        
        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            confidence = track[5] if len(track) > 5 else 1.0
            
            # 클래스 ID 추출 (det_array 에서)
            class_id = 1  # 기본값: player
            if len(det_array) > 0:
                # 가장 가까운 탐지의 클래스 ID 사용
                det_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                min_dist = float('inf')
                for det in detections:
                    dist = np.sqrt((det.center[0] - det_center[0])**2 + 
                                 (det.center[1] - det_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        class_id = det.class_id
                        
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # 기존 추적 객체인지 확인
            if track_id in self.tracked_objects:
                obj = self.tracked_objects[track_id]
                obj.bbox = (x1, y1, x2, y2)
                obj.center = center
                obj.frames_tracked += 1
                obj.frame_detected = self.frame_count
                
                # 속도 계산
                if len(obj.trajectory) > 1:
                    prev_center = obj.trajectory[-1]
                    obj.velocity = (
                        center[0] - prev_center[0],
                        center[1] - prev_center[1]
                    )
                    
                obj.trajectory.append(center)
            else:
                # 새로운 추적 객체 생성
                label = SoccerDetector.CLASS_MAP.get(class_id, f'class_{class_id}')
                obj = TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    center=center,
                    frame_detected=self.frame_count
                )
                self.tracked_objects[track_id] = obj
                
            tracked_objects.append(obj)
            
        # 오래된 추적 객체 제거
        self._remove_stale_tracks()
        
        return tracked_objects
    
    def _remove_stale_tracks(self):
        """오래된 추적 객체 제거"""
        stale_ids = []
        
        for track_id, obj in self.tracked_objects.items():
            if self.frame_count - obj.frame_detected > self.max_age:
                stale_ids.append(track_id)
                
        for track_id in stale_ids:
            del self.tracked_objects[track_id]
            
    def get_all_tracked_objects(self) -> Dict[int, TrackedObject]:
        """모든 추적 중인 객체 반환"""
        return self.tracked_objects.copy()
    
    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """
        특정 객체의 이동 경로 반환
        
        Args:
            track_id: 추적 ID
            
        Returns:
            중심점 좌표 리스트
        """
        if track_id in self.tracked_objects:
            return self.tracked_objects[track_id].trajectory.copy()
        return []
    
    def reset(self):
        """추적기 리셋"""
        self.tracker = None
        self.frame_count = 0
        self.tracked_objects.clear()
        print("추적기가 리셋되었습니다.")
        
    def visualize(self, frame: np.ndarray, 
                  tracked_objects: List[TrackedObject]) -> np.ndarray:
        """
        추적 결과 시각화
        
        Args:
            frame: 입력 이미지
            tracked_objects: 추적 객체 리스트
            
        Returns:
            시각화된 이미지
        """
        vis_frame = frame.copy()
        
        # 색상 맵 (track_id 기반으로 일관된 색상 할당)
        color_map = defaultdict(lambda: (128, 128, 128))
        
        for obj in tracked_objects:
            # track_id 기반으로 색상 생성
            if obj.track_id not in color_map or color_map[obj.track_id] == (128, 128, 128):
                np.random.seed(obj.track_id)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                color_map[obj.track_id] = color
                
            color = color_map[obj.track_id]
            
            x1, y1, x2, y2 = obj.bbox
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # ID 와 레이블 표시
            label_text = f"ID:{obj.track_id} {obj.label}"
            cv2.putText(vis_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
            # 이동 경로 그리기 (최근 10 프레임)
            if len(obj.trajectory) > 1:
                points = [tuple(map(int, p)) for p in obj.trajectory[-10:]]
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)
                    
        return vis_frame


# detector.py 의 CLASS_MAP 참조를 위한 임시 정의
class SoccerDetector:
    CLASS_MAP = {
        0: 'ball',
        1: 'player', 
        2: 'goalkeeper',
        3: 'referee',
        4: 'assistant_referee'
    }


if __name__ == '__main__':
    # 테스트 코드
    tracker = SoccerTracker(tracker_type='bytetrack')
    
    # 더미 탐지 데이터 생성
    from detector import Detection
    
    detections = [
        Detection(class_id=1, confidence=0.9, bbox=(100, 100, 150, 200), label='player'),
        Detection(class_id=0, confidence=0.8, bbox=(300, 300, 320, 320), label='ball'),
    ]
    
    # 추적 업데이트
    tracked_objects = tracker.update(detections)
    
    print(f"추적 중인 객체 수: {len(tracked_objects)}")
    for obj in tracked_objects:
        print(f"  ID: {obj.track_id}, Label: {obj.label}, Center: {obj.center}")
