"""
호모그래피 변환 모듈
2D 이미지 좌표를 경기장 3D 좌표로 변환
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FieldPoint:
    """경기장 기준점"""
    name: str
    x: float  # 경기장 X 좌표 (미터)
    y: float  # 경기장 Y 좌표 (미터)
    pixel_x: Optional[float] = None  # 이미지 X 좌표 (픽셀)
    pixel_y: Optional[float] = None  # 이미지 Y 좌표 (픽셀)


class HomographyTransformer:
    """
    호모그래피 변환기
    2D 이미지 좌표 <-> 3D 경기장 좌표 변환 수행
    """
    
    # 표준 축구장 크기 (FIFA 규격, 미터 단위)
    FIELD_LENGTH = 105.0  # 길이
    FIELD_WIDTH = 68.0    # 너비
    
    def __init__(self):
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.field_points: List[FieldPoint] = []
        self.pixel_points: List[Tuple[float, float]] = []
        
    def set_field_points(self, points: List[FieldPoint]):
        """
        경기장 기준점 설정
        
        Args:
            points: FieldPoint 객체 리스트
        """
        self.field_points = points
        
    def load_pixel_correspondences(self, pixel_coords: Dict[str, Tuple[float, float]]):
        """
        이미지 상의 대응점 로드
        
        Args:
            pixel_coords: {point_name: (pixel_x, pixel_y)} 딕셔너리
        """
        self.pixel_points = []
        matched_points = []
        
        for field_point in self.field_points:
            if field_point.name in pixel_coords:
                px, py = pixel_coords[field_point.name]
                self.pixel_points.append((px, py))
                matched_points.append(field_point)
                
        if len(matched_points) != len(self.field_points):
            print(f"경고: 일부 대응점을 찾지 못했습니다. ({len(matched_points)}/{len(self.field_points)})")
            
        self.field_points = matched_points
        
    def compute_homography(self, 
                          pixel_points: List[Tuple[float, float]],
                          world_points: List[Tuple[float, float]]) -> bool:
        """
        호모그래피 행렬 계산
        
        Args:
            pixel_points: 이미지 좌표 리스트 [(x1, y1), (x2, y2), ...]
            world_points: 월드 좌표 리스트 [(x1, y1), (x2, y2), ...]
            
        Returns:
            성공 여부
        """
        if len(pixel_points) < 4:
            print("호모그래피 계산에 최소 4 개의 대응점이 필요합니다.")
            return False
            
        pixel_array = np.array(pixel_points, dtype=np.float32)
        world_array = np.array(world_points, dtype=np.float32)
        
        # 호모그래피 행렬 계산
        H, _ = cv2.findHomography(pixel_array, world_array)
        
        if H is None:
            print("호모그래피 행렬을 계산할 수 없습니다.")
            return False
            
        self.homography_matrix = H
        self.inverse_homography = np.linalg.inv(H)
        
        print("호모그래피 행렬 계산 완료")
        print(f"H = \n{H}")
        
        return True
        
    def compute_from_field_lines(self, image_corners: List[Tuple[float, float]]) -> bool:
        """
        경기장 코너 기반 호모그래피 자동 계산
        
        Args:
            image_corners: 이미지 상의 경기장 4 개 코너
                          [(top_left), (top_right), (bottom_right), (bottom_left)]
                          
        Returns:
            성공 여부
        """
        if len(image_corners) != 4:
            print("4 개의 코너 점이 필요합니다.")
            return False
            
        # 월드 좌표 설정 (경기장 중심이 원점)
        half_length = self.FIELD_LENGTH / 2
        half_width = self.FIELD_WIDTH / 2
        
        world_corners = [
            (-half_length, -half_width),   # top_left
            (half_length, -half_width),    # top_right
            (half_length, half_width),     # bottom_right
            (-half_length, half_width)     # bottom_left
        ]
        
        return self.compute_homography(image_corners, world_corners)
        
    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        픽셀 좌표를 월드 좌표로 변환
        
        Args:
            pixel_x: 이미지 X 좌표
            pixel_y: 이미지 Y 좌표
            
        Returns:
            (world_x, world_y) 미터 단위
        """
        if self.homography_matrix is None:
            raise ValueError("호모그래피 행렬이 설정되지 않았습니다.")
            
        pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        pixel_point = np.expand_dims(pixel_point, axis=1)
        
        world_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
        
        return (world_point[0, 0, 0], world_point[0, 0, 1])
        
    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        월드 좌표를 픽셀 좌표로 변환
        
        Args:
            world_x: 월드 X 좌표 (미터)
            world_y: 월드 Y 좌표 (미터)
            
        Returns:
            (pixel_x, pixel_y)
        """
        if self.inverse_homography is None:
            raise ValueError("호모그래피 행렬이 설정되지 않았습니다.")
            
        world_point = np.array([[world_x, world_y]], dtype=np.float32)
        world_point = np.expand_dims(world_point, axis=1)
        
        pixel_point = cv2.perspectiveTransform(world_point, self.inverse_homography)
        
        return (pixel_point[0, 0, 0], pixel_point[0, 0, 1])
        
    def transform_detections(self, detections: List) -> List[Tuple]:
        """
        탐지된 객체들의 좌표를 월드 좌표로 변환
        
        Args:
            detections: Detection 객체 리스트
            
        Returns:
            [(track_id, world_x, world_y, confidence)] 리스트
        """
        transformed = []
        
        for det in detections:
            center_x, center_y = det.center
            world_x, world_y = self.pixel_to_world(center_x, center_y)
            
            transformed.append({
                'id': getattr(det, 'track_id', None),
                'label': det.label,
                'world_x': world_x,
                'world_y': world_y,
                'confidence': det.confidence,
                'pixel_bbox': det.bbox
            })
            
        return transformed
        
    def validate_transformation(self, test_points: List[Tuple[float, float]]) -> Dict:
        """
        변환 정확도 검증
        
        Args:
            test_points: 검증용 픽셀 좌표 리스트
            
        Returns:
            오차 통계
        """
        errors = []
        
        for px, py in test_points:
            # 픽셀 -> 월드 -> 픽셀
            wx, wy = self.pixel_to_world(px, py)
            px_recon, py_recon = self.world_to_pixel(wx, wy)
            
            error = np.sqrt((px - px_recon)**2 + **(py - py_recon)2)
            errors.append(error)
            
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors)
        }
        
    def draw_field_overlay(self, frame: np.ndarray, 
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        경기장 라인 오버레이 그리기
        
        Args:
            frame: 입력 이미지
            color: 라인 색상 (BGR)
            thickness: 라인 두께
            
        Returns:
            오버레이가 그려진 이미지
        """
        if self.inverse_homography is None:
            return frame
            
        overlay = frame.copy()
        
        # 경기장 주요 포인트 정의 (월드 좌표)
        half_length = self.FIELD_LENGTH / 2
        half_width = self.FIELD_WIDTH / 2
        
        # 경기장 외곽선
        field_outline = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
            (-half_length, -half_width)  # 닫기
        ]
        
        # 월드 좌표를 픽셀 좌표로 변환
        pixel_points = []
        for wx, wy in field_outline:
            px, py = self.world_to_pixel(wx, wy)
            pixel_points.append((int(px), int(py)))
            
        # 라인 그리기
        pixel_points_np = np.array(pixel_points, dtype=np.int32)
        cv2.polylines(overlay, [pixel_points_np], False, color, thickness)
        
        # 센터 서클
        center_pixel = self.world_to_pixel(0, 0)
        # 반지름 9.15m 를 픽셀로 변환 (대략적)
        radius_pixel = self.world_to_pixel(9.15, 0)[0] - center_pixel[0]
        cv2.circle(overlay, (int(center_pixel[0]), int(center_pixel[1])), 
                  int(abs(radius_pixel)), color, thickness)
                  
        return overlay
        
    def save_calibration(self, filepath: str):
        """
        캘리브레이션 데이터 저장
        
        Args:
            filepath: 저장 파일 경로
        """
        calibration_data = {
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'field_length': self.FIELD_LENGTH,
            'field_width': self.FIELD_WIDTH,
            'field_points': [
                {
                    'name': fp.name,
                    'x': fp.x,
                    'y': fp.y,
                    'pixel_x': fp.pixel_x,
                    'pixel_y': fp.pixel_y
                }
                for fp in self.field_points
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
            
        print(f"캘리브레이션 데이터 저장 완료: {filepath}")
        
    def load_calibration(self, filepath: str):
        """
        캘리브레이션 데이터 로드
        
        Args:
            filepath: 로드 파일 경로
        """
        import json
        
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)
            
        if calibration_data['homography_matrix'] is not None:
            self.homography_matrix = np.array(calibration_data['homography_matrix'])
            self.inverse_homography = np.linalg.inv(self.homography_matrix)
            
        self.FIELD_LENGTH = calibration_data.get('field_length', self.FIELD_LENGTH)
        self.FIELD_WIDTH = calibration_data.get('field_width', self.FIELD_WIDTH)
        
        print(f"캘리브레이션 데이터 로드 완료: {filepath}")


if __name__ == '__main__':
    # 테스트 코드
    transformer = HomographyTransformer()
    
    # 더미 데이터로 테스트
    pixel_corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
    
    success = transformer.compute_from_field_lines(pixel_corners)
    
    if success:
        # 테스트 변환
        test_pixel = (640, 360)  # 이미지 중심
        world_coord = transformer.pixel_to_world(*test_pixel)
        print(f"픽셀 {test_pixel} -> 월드 {world_coord} 미터")
        
        # 역변환 테스트
        pixel_recon = transformer.world_to_pixel(*world_coord)
        print(f"월드 {world_coord} -> 픽셀 {pixel_recon}")
