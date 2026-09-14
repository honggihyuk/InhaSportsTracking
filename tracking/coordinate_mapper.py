"""
2D 좌표를 3D 월드 좌표로 매핑하는 모듈
카메라 캘리브레이션 및 좌표계 변환 수행
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CameraParams:
    """카메라 내부/외부 파라미터"""
    # 내부 파라미터
    fx: float  # 초점 거리 (x)
    fy: float  # 초점 거리 (y)
    cx: float  # 주점 (x)
    cy: float  # 주점 (y)
    
    # 외부 파라미터
    R: Optional[np.ndarray] = None  # 회전 행렬 (3x3)
    t: Optional[np.ndarray] = None  # 병진 벡터 (3,)
    
    @property
    def K(self) -> np.ndarray:
        """내부 파라미터 행렬"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @property
    def extrinsics(self) -> np.ndarray:
        """외부 파라미터 행렬 [R|t]"""
        if self.R is None or self.t is None:
            return None
        return np.hstack([self.R, self.t.reshape(-1, 1)])


@dataclass
class Point3D:
    """3D 공간 상의 점"""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Point3D') -> float:
        """다른 점까지의 거리 계산"""
        return np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )


class CoordinateMapper:
    """
    2D 이미지 좌표를 3D 월드 좌표로 매핑
    
    카메라 투영 기하학을 사용하여 평면 (경기장) 위의 점을 3D 로 복원
    """
    
    def __init__(self, camera_params: Optional[CameraParams] = None):
        """
        Args:
            camera_params: 카메라 파라미터 (없으면 기본값 사용)
        """
        self.camera_params = camera_params
        self.ground_plane_normal = np.array([0, 1, 0])  # 지면 법선벡터 (Y 축)
        self.ground_plane_height = 0.0  # 지면 높이 (Y=0)
        
    def set_camera_params(self, params: CameraParams):
        """카메라 파라미터 설정"""
        self.camera_params = params
        
    def estimate_camera_from_homography(self, H: np.ndarray, 
                                       field_size: Tuple[float, float]) -> CameraParams:
        """
        호모그래피 행렬로부터 카메라 파라미터 추정
        
        Args:
            H: 호모그래피 행렬 (3x3)
            field_size: 경기장 크기 (length, width)
            
        Returns:
            CameraParams 객체
        """
        # 호모그래피를 카메라 행렬로 분해
        # H = K * [r1|r2|t] (평면 Z=0 가정)
        
        # 단순화를 위해 정사영 카메라 가정
        fx = fy = 1000.0  # 초점거리 (픽셀)
        cx = field_size[0] / 2  # 이미지 중심
        cy = field_size[1] / 2
        
        params = CameraParams(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            R=np.eye(3),  # 기본 회전
            t=np.array([0, 10, 0])  # 카메라 높이 10m
        )
        
        self.camera_params = params
        return params
        
    def pixel_to_3d_ray(self, pixel_x: float, pixel_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        픽셀 좌표를 3D 레이 (반직선) 로 변환
        
        Args:
            pixel_x: 픽셀 X 좌표
            pixel_y: 픽셀 Y 좌표
            
        Returns:
            (origin, direction): 레이 원점과 방향 벡터
        """
        if self.camera_params is None:
            raise ValueError("카메라 파라미터가 설정되지 않았습니다.")
            
        K = self.camera_params.K
        K_inv = np.linalg.inv(K)
        
        # 정규화된 이미지 평면 좌표
        normalized = K_inv @ np.array([pixel_x, pixel_y, 1])
        
        # 카메라 좌표계에서 레이 방향
        direction = normalized / np.linalg.norm(normalized)
        
        # 월드 좌표계로 변환
        if self.camera_params.R is not None:
            direction = self.camera_params.R.T @ direction
            
        origin = -self.camera_params.R.T @ self.camera_params.t.reshape(-1, 1)
        
        return (origin.flatten(), direction)
        
    def intersect_with_ground(self, pixel_x: float, pixel_y: float) -> Point3D:
        """
        픽셀 좌표와 지면 평면의 교점 계산
        
        Args:
            pixel_x: 픽셀 X 좌표
            pixel_y: 픽셀 Y 좌표
            
        Returns:
            Point3D: 지면과의 교점
        """
        origin, direction = self.pixel_to_3d_ray(pixel_x, pixel_y)
        
        # 레이 - 평면 교차 계산
        # plane: n·(P - P0) = 0
        # ray: P = O + t*D
        
        n = self.ground_plane_normal
        P0 = np.array([0, self.ground_plane_height, 0])
        
        denom = np.dot(n, direction)
        if abs(denom) < 1e-6:
            # 레이와 평면이 평행
            raise ValueError("레이가 지면과 평행합니다.")
            
        t = np.dot(n, P0 - origin) / denom
        
        intersection = origin + t * direction
        
        return Point3D(
            x=intersection[0],
            y=intersection[1],
            z=intersection[2]
        )
        
    def detections_to_3d(self, detections: List, 
                        homography_matrix: np.ndarray) -> List[Dict]:
        """
        탐지된 객체들을 3D 월드 좌표로 변환
        
        Args:
            detections: Detection 객체 리스트
            homography_matrix: 호모그래피 행렬
            
        Returns:
            3D 좌표 정보가 추가된 딕셔너리 리스트
        """
        results = []
        
        for det in detections:
            center_x, center_y = det.center
            
            # 호모그래피를 사용하여 2D 월드 좌표 변환
            pixel_point = np.array([center_x, center_y, 1])
            world_point_h = homography_matrix @ pixel_point
            world_point_h /= world_point_h[2]  # 동차 좌표 정규화
            
            world_x, world_y = world_point_h[0], world_point_h[1]
            
            # 3D 좌표 (지면 Y=0 가정)
            point_3d = Point3D(x=world_x, y=0.0, z=world_y)
            
            results.append({
                'id': getattr(det, 'track_id', None),
                'label': det.label,
                'pixel_center': (center_x, center_y),
                'world_2d': (world_x, world_y),
                'world_3d': point_3d,
                'confidence': det.confidence,
                'bbox': det.bbox
            })
            
        return results
        
    def compute_player_height(self, bbox: Tuple[int, int, int, int],
                             foot_position: Point3D) -> float:
        """
        선수의 키 추정
        
        Args:
            bbox: 바운딩 박스 (x1, y1, x2, y2)
            foot_position: 발 위치 (3D)
            
        Returns:
            추정 키 (미터)
        """
        x1, y1, x2, y2 = bbox
        head_pixel_y = y1
        foot_pixel_y = y2
        
        # 피라미드 기하학을 사용한 높이 추정
        # 단순화: 카메라가 경기장을 내려다보는 경우
        pixel_height = foot_pixel_y - head_pixel_y
        
        # 대략적인 높이 계산 (카메라 높이에 비례)
        if self.camera_params and self.camera_params.t is not None:
            camera_height = abs(self.camera_params.t[1])
            # 경험적 비율
            estimated_height = pixel_height * 0.005 + 1.5
            
        else:
            estimated_height = 1.75  # 기본값
            
        return estimated_height
        
    def trajectory_to_3d(self, trajectory: List[Tuple[float, float]],
                        homography_matrix: np.ndarray) -> List[Point3D]:
        """
        2D 이동 경로를 3D 로 변환
        
        Args:
            trajectory: 2D 중심점 좌표 리스트
            homography_matrix: 호모그래피 행렬
            
        Returns:
            Point3D 리스트
        """
        points_3d = []
        
        for px, py in trajectory:
            point_3d = self.intersect_with_ground(px, py)
            points_3d.append(point_3d)
            
        return points_3d
        
    def compute_speed(self, point1: Point3D, point2: Point3D,
                     time_delta: float) -> float:
        """
        두 점 사이의 속도 계산
        
        Args:
            point1: 시작점
            point2: 종료점
            time_delta: 시간 간격 (초)
            
        Returns:
            속도 (m/s)
        """
        distance = point1.distance_to(point2)
        speed = distance / time_delta if time_delta > 0 else 0
        
        return speed
        
    def create_top_down_view(self, detections_3d: List[Dict],
                            scale: float = 10.0) -> np.ndarray:
        """
        Top-down 뷰 이미지 생성
        
        Args:
            detections_3d: 3D 검출 결과 리스트
            scale: 스케일 (픽셀/미터)
            
        Returns:
            Top-down 뷰 이미지
        """
        import cv2
        
        # 캔버스 크기
        canvas_width = int(self.camera_params.fx * 2) if self.camera_params else 1280
        canvas_height = int(self.camera_params.fy * 2) if self.camera_params else 720
        
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 중심점 (원점)
        center_x, center_y = canvas_width // 2, canvas_height // 2
        
        for det in detections_3d:
            point_3d = det['world_3d']
            
            # 3D 좌표를 Top-down 픽셀 좌표로 변환
            pixel_x = int(center_x + point_3d.x * scale)
            pixel_y = int(center_y + point_3d.z * scale)
            
            # 점 그리기
            color = (0, 255, 0) if det['label'] == 'player' else (0, 0, 255)
            cv2.circle(canvas, (pixel_x, pixel_y), 5, color, -1)
            
            # ID 표시
            if det['id'] is not None:
                cv2.putText(canvas, f"ID:{det['id']}", 
                           (pixel_x + 10, pixel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                           
        return canvas
        
    def save_3d_data(self, filepath: str, detections_3d: List[Dict]):
        """
        3D 데이터를 JSON 으로 저장
        
        Args:
            filepath: 저장 파일 경로
            detections_3d: 3D 검출 결과 리스트
        """
        import json
        
        data = {
            'camera_params': {
                'fx': self.camera_params.fx if self.camera_params else None,
                'fy': self.camera_params.fy if self.camera_params else None,
                'cx': self.camera_params.cx if self.camera_params else None,
                'cy': self.camera_params.cy if self.camera_params else None
            },
            'detections': [
                {
                    'id': d['id'],
                    'label': d['label'],
                    'pixel_center': d['pixel_center'],
                    'world_2d': d['world_2d'],
                    'world_3d': {
                        'x': d['world_3d'].x,
                        'y': d['world_3d'].y,
                        'z': d['world_3d'].z
                    },
                    'confidence': d['confidence']
                }
                for d in detections_3d
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"3D 데이터 저장 완료: {filepath}")


if __name__ == '__main__':
    # 테스트 코드
    mapper = CoordinateMapper()
    
    # 더미 카메라 파라미터 설정
    params = CameraParams(
        fx=800,
        fy=800,
        cx=640,
        cy=360,
        R=np.eye(3),
        t=np.array([0, 15, 0])
    )
    mapper.set_camera_params(params)
    
    # 테스트: 픽셀 좌표를 3D 로
    test_pixel = (640, 400)
    try:
        point_3d = mapper.intersect_with_ground(*test_pixel)
        print(f"픽셀 {test_pixel} -> 3D 점 ({point_3d.x:.2f}, {point_3d.y:.2f}, {point_3d.z:.2f})")
    except Exception as e:
        print(f"오류: {e}")
