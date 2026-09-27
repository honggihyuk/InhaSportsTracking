"""
객체 추적 모듈 - ByteTrack/BoT-SORT 기반 선수 및 공 추적
탐지된 객체에 ID 를 할당하여 시간적 일관성 유지
"""

import cv2
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
from dataclasses import dataclass

from .detector import Detection


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


class SimpleTracker:
    """
    간단한 추적기 (boxmot 설치되지 않았을 때 사용)
    IoU 기반 매칭으로 간단한 ID 유지
    """
    
    def __init__(self, match_threshold: float = 0.3, max_age: int = 30):
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.trackers = {}  # track_id -> {bbox, age, class_id, label}
        self.next_id = 1  # 0 은 공 전용 ID(SoccerTracker.BALL_TRACK_ID)로 예약, boxmot 도 1 부터 시작
        
    def update(self, det_array: np.ndarray, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        간단한 IoU 기반 추적 업데이트
        
        Args:
            det_array: [[x1, y1, x2, y2, conf, class_id], ...]
            
        Returns:
            [[x1, y1, x2, y2, track_id, conf, class_id], ...] (boxmot 출력과 동일한 열 순서)
        """
        tracks = []
        used = np.zeros(len(det_array), dtype=bool)
        tids = list(self.trackers)

        # IoU 행렬을 한 번에 계산 (tracker x detection)
        ious = (self._iou_matrix(np.array([self.trackers[t]['bbox'] for t in tids]), det_array[:, :4])
                if tids and len(det_array) else np.zeros((len(tids), 0)))

        for row, tid in enumerate(tids):
            # 기존과 동일한 greedy 매칭: 아직 쓰이지 않은 탐지 중 IoU 최대
            cand = np.where(used, -1.0, ious[row])
            i = int(cand.argmax()) if cand.size else -1
            if i >= 0 and cand[i] > self.match_threshold:
                det = det_array[i]
                self.trackers[tid].update(bbox=det[:4], age=0, conf=det[4])
                tracks.append([*det[:4], tid, det[4], self.trackers[tid]['class_id']])
                used[i] = True
            else:
                self.trackers[tid]['age'] += 1

        # 매칭되지 않은 탐지는 새 tracker 로 등록
        for det in det_array[~used]:
            self.trackers[self.next_id] = {'bbox': det[:4], 'age': 0, 'class_id': det[5], 'conf': det[4]}
            tracks.append([*det[:4], self.next_id, det[4], det[5]])
            self.next_id += 1

        # 오래된 tracker 제거
        self.trackers = {tid: t for tid, t in self.trackers.items() if t['age'] <= self.max_age}

        return np.array(tracks) if tracks else np.empty((0, 7))

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a(N,4) x b(M,4) IoU 행렬"""
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def make_tracker(tracker_type: str = 'bytetrack',
                 track_threshold: float = 0.3,
                 match_threshold: float = 0.8,
                 max_age: int = 30,
                 frame_rate: int = 30):
    """
    다중 객체 추적기 생성 (boxmot BYTETracker / BoTSORT / SimpleTracker)

    세 추적기는 모두 update(det_array, frame) → [[x1, y1, x2, y2, id, conf, cls, ...], ...]
    형태를 공유하므로 호출하는 쪽은 구현을 구분할 필요가 없다.
    boxmot 이 없거나 버전이 맞지 않으면 SimpleTracker 로 대체한다.

    Args:
        tracker_type: 'bytetrack' | 'botsort' | 'simple'
        match_threshold: ByteTrack 기준 매칭 비용(1 - IoU) 상한
    """
    # ByteTrack 의 match_thresh 는 매칭 비용(1 - IoU) 상한 → SimpleTracker 의 최소 IoU 로 변환
    simple = lambda: SimpleTracker(match_threshold=1 - match_threshold, max_age=max_age)
    if tracker_type == 'simple':
        return simple()
    if tracker_type not in ('bytetrack', 'botsort'):
        raise ValueError(f"지원하지 않는 추적기 타입: {tracker_type}")
    try:
        import boxmot
        cls = boxmot.BYTETracker if tracker_type == 'bytetrack' else boxmot.BoTSORT
        tracker = cls(track_thresh=track_threshold, match_thresh=match_threshold,
                      track_buffer=max_age, frame_rate=frame_rate)
        print(f"추적기 초기화 완료: {tracker_type}")
        return tracker
    except (ImportError, AttributeError, TypeError) as e:  # 미설치 / 버전별 클래스명·생성자 인자 불일치
        print(f"boxmot 을 사용할 수 없어 SimpleTracker 로 대체합니다. ({e})")
        print("pip install boxmot==10.0.84 를 실행하여 호환 버전을 설치해주세요.")
        return simple()


class BallTracker:
    """
    공 전용 단일 객체 추적기

    경기에는 공이 하나뿐이므로 다중 객체 추적기(ID 가 쪼개지기 쉬움) 대신
    매 프레임 공 후보 중 '하나만' 골라 항상 같은 ID(BALL_TRACK_ID)를 부여한다.
      1. 직전 위치 + 속도로 이번 위치를 예측 (등속 가정)
      2. 예측 위치에서 max_jump 픽셀 이내 후보 중 가장 가까운 것을 선택 (오탐 제거)
      3. 추적을 잃은 지 max_missing 프레임이 지나면 가장 신뢰도 높은 후보로 재획득
    """

    def __init__(self, max_jump: float = 100.0, max_missing: int = 10):
        # max_jump: 프레임당 허용 이동량(픽셀). 해상도/줌에 따라 튜닝 필요
        self.max_jump = max_jump
        self.max_missing = max_missing
        self.reset()

    def reset(self):
        self.last: Optional[Tuple[float, float]] = None
        self.velocity = (0.0, 0.0)
        self.missing = 0

    def update(self, balls: List[Detection]) -> Optional[Detection]:
        """공 후보 리스트 → 선택된 공 1개 (없으면 None)"""
        if self.last is not None and self.missing >= self.max_missing:
            self.reset()  # 너무 오래 놓침 → 재획득 모드

        if self.last is None:
            best = max(balls, key=lambda b: b.confidence, default=None)
        else:
            steps = self.missing + 1  # 놓친 프레임만큼 예측을 더 진행
            px = self.last[0] + self.velocity[0] * steps
            py = self.last[1] + self.velocity[1] * steps
            dist = lambda b: np.hypot(b.center[0] - px, b.center[1] - py)
            near = [b for b in balls if dist(b) <= self.max_jump * steps]
            best = min(near, key=dist, default=None)

        if best is None:
            self.missing += 1
            return None

        if self.last is not None:
            steps = self.missing + 1
            self.velocity = ((best.center[0] - self.last[0]) / steps,
                             (best.center[1] - self.last[1]) / steps)
        self.last = best.center
        self.missing = 0
        return best


class SoccerTracker:
    """
    축구 경기장 객체 추적기
    - 선수/심판 등: 다중 객체 추적기(ByteTrack/BoT-SORT/Simple)
    - 공: BallTracker 로 단일 ID(BALL_TRACK_ID) 유지
    """

    BALL_TRACK_ID = 0  # 공 전용 ID (다중 객체 추적기 ID 는 1 부터 시작하므로 겹치지 않음)

    def __init__(self,
                 tracker_type: str = 'bytetrack',
                 track_threshold: float = 0.3,
                 match_threshold: float = 0.8,
                 max_age: int = 30,
                 backend_factory: Optional[Callable[[], object]] = None,
                 ball_tracker: Optional[BallTracker] = None):
        """
        Args:
            tracker_type: 추적기 타입 ('bytetrack' | 'botsort' | 'simple')
            track_threshold: 추적 신뢰도 임계값
            match_threshold: 매칭 임계값 (ByteTrack 기준 1 - IoU)
            max_age: 추적 손실 후 유지할 최대 프레임 수
            backend_factory: 다중 객체 추적기 생성 함수 주입 (기본: make_tracker)
            ball_tracker: 공 추적기 주입 (기본: BallTracker())
        """
        self.max_age = max_age
        self.backend_factory = backend_factory or (
            lambda: make_tracker(tracker_type, track_threshold, match_threshold, max_age))
        self.ball_tracker = ball_tracker or BallTracker()

        self.tracker = None  # 첫 update 에서 생성 (reset 시 재생성)
        self.frame_count = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """
        추적기 업데이트

        Args:
            detections: Detection 리스트
            frame: 현재 프레임 이미지 (선택사항, BoT-SORT ReID 용)

        Returns:
            TrackedObject 리스트
        """
        if self.tracker is None:
            self.tracker = self.backend_factory()

        self.frame_count += 1

        balls = [d for d in detections if d.is_ball]
        others = [d for d in detections if not d.is_ball]

        # 다중 객체 추적기 입력: [x1, y1, x2, y2, confidence, class_id]
        det_array = np.array(
            [[*d.bbox, d.confidence, d.class_id] for d in others], dtype=float
        ).reshape(-1, 6)
        labels = {d.class_id: d.label for d in others}

        tracked_objects = []
        for track in self.tracker.update(det_array, frame):
            # boxmot / SimpleTracker 모두 7번째 열에 class_id 를 반환
            class_id = int(track[6]) if len(track) > 6 else 0
            tracked_objects.append(self._upsert(
                track_id=int(track[4]),
                bbox=tuple(map(int, track[:4])),
                confidence=float(track[5]) if len(track) > 5 else 1.0,
                class_id=class_id,
                label=labels.get(class_id, f'class_{class_id}'),
            ))

        ball = self.ball_tracker.update(balls)
        if ball is not None:
            tracked_objects.append(self._upsert(
                self.BALL_TRACK_ID, ball.bbox, ball.confidence, ball.class_id, ball.label))

        # 오래된 추적 객체 제거
        self._remove_stale_tracks()

        return tracked_objects

    def _upsert(self, track_id: int, bbox: Tuple[int, int, int, int], confidence: float,
                class_id: int, label: str) -> TrackedObject:
        """track_id 의 TrackedObject 를 갱신하거나 새로 만든다"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        obj = self.tracked_objects.get(track_id)
        if obj is None:
            obj = TrackedObject(track_id=track_id, class_id=class_id, label=label, bbox=bbox,
                                confidence=confidence, center=center, frame_detected=self.frame_count)
            self.tracked_objects[track_id] = obj
            return obj
        # 속도 (픽셀/프레임)
        obj.velocity = (center[0] - obj.center[0], center[1] - obj.center[1])
        obj.bbox = bbox
        obj.center = center
        obj.confidence = confidence
        obj.frames_tracked += 1
        obj.frame_detected = self.frame_count
        obj.trajectory.append(center)
        return obj

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
        self.ball_tracker.reset()
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

        for obj in tracked_objects:
            # track_id 기반의 일관된 색상 (전역 난수 상태를 건드리지 않음)
            color = tuple(map(int, np.random.default_rng(obj.track_id).integers(0, 255, 3)))

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


if __name__ == '__main__':
    # 테스트 코드: python -m tracking.tracker
    from .detector import BALL_OFFSET

    tracker = SoccerTracker(tracker_type='simple')
    detections = [
        Detection(bbox=(100, 100, 150, 200), confidence=0.9, class_id=0, class_name='player_player', type='player'),
        Detection(bbox=(300, 300, 320, 320), confidence=0.8, class_id=BALL_OFFSET, class_name='ball_ball', type='ball'),
    ]
    tracked_objects = tracker.update(detections)

    print(f"추적 중인 객체 수: {len(tracked_objects)}")
    for obj in tracked_objects:
        print(f"  ID: {obj.track_id}, Label: {obj.label}, Center: {obj.center}")
