"""
통합 파이프라인 모듈
축구 영상 → 탐지/추적 → 3D 매핑 → Dynamic 3DGS 렌더링
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml

from tracking.detector import SoccerDetector, Detection
from tracking.tracker import SoccerTracker, TrackedObject
from tracking.homography import HomographyTransformer
from tracking.coordinate_mapper import CoordinateMapper
from gs_model.canonical_gs import CanonicalGaussianSpace, HumanGaussianTemplate
from gs_model.deformation_mlp import DeformationMLP, DynamicGaussianRenderer


@dataclass
class FrameData:
    """프레임별 데이터"""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    tracked_objects: List[TrackedObject]
    world_coordinates: List[Dict]
    

class Soccer3DPipeline:
    """
    축구 3D Digital Twin 파이프라인
    
    1. 영상 입력
    2. 객체 탐지 (YOLO)
    3. 객체 추적 (ByteTrack)
    4. 호모그래피 변환 (2D → 3D 월드)
    5. Dynamic 3DGS 변형 및 렌더링
    """
    
    def __init__(self, config_path: str = 'configs/model_config.yaml'):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        
        # 컴포넌트 초기화
        self.detector = None
        self.tracker = None
        self.homography = None
        self.coord_mapper = None
        self.canonical_space = None
        self.deformation_mlp = None
        self.renderer = None
        
        # 상태 변수
        self.frame_count = 0
        self.device = self.config.get('device', 'cuda')
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"설정 파일 로드 완료: {config_path}")
        else:
            print(f"설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            config = {
                'detector': {
                    'confidence_threshold': 0.5,
                    'model_path': 'models/yolo_soccer.pt'
                },
                'tracker': {
                    'type': 'bytetrack',
                    'track_threshold': 0.3
                },
                'gaussian': {
                    'num_gaussians': 10000,
                    'sh_degree': 3
                }
            }
        return config
        
    def initialize_components(self):
        """모든 컴포넌트 초기화"""
        print("\n=== 컴포넌트 초기화 ===")
        
        # 1. 객체 탐지기
        det_config = self.config.get('detector', {})
        self.detector = SoccerDetector(
            model_path=det_config.get('model_path', 'models/yolo_soccer.pt'),
            confidence_threshold=det_config.get('confidence_threshold', 0.5),
            device=self.device
        )
        print("✓ SoccerDetector 초기화 완료")
        
        # 2. 객체 추적기
        track_config = self.config.get('tracker', {})
        self.tracker = SoccerTracker(
            tracker_type=track_config.get('type', 'bytetrack'),
            track_threshold=track_config.get('track_threshold', 0.3)
        )
        print("✓ SoccerTracker 초기화 완료")
        
        # 3. 호모그래피 변환기
        self.homography = HomographyTransformer()
        print("✓ HomographyTransformer 초기화 완료")
        
        # 4. 3D 좌표 매퍼
        self.coord_mapper = CoordinateMapper()
        print("✓ CoordinateMapper 초기화 완료")
        
        # 5. Canonical Gaussian Space
        gs_config = self.config.get('gaussian', {})
        self.canonical_space = CanonicalGaussianSpace(
            num_gaussians=gs_config.get('num_gaussians', 10000),
            sh_degree=gs_config.get('sh_degree', 3),
            device=self.device
        )
        print("✓ CanonicalGaussianSpace 초기화 완료")
        
        # 6. Deformation MLP
        self.deformation_mlp = DeformationMLP(
            input_dim=20,
            feature_dim=256,
            num_layers=8
        ).to(self.device)
        print("✓ DeformationMLP 초기화 완료")
        
        # 7. Dynamic Renderer
        self.renderer = DynamicGaussianRenderer(
            canonical_space=self.canonical_space,
            deformation_mlp=self.deformation_mlp,
            device=self.device
        )
        print("✓ DynamicGaussianRenderer 초기화 완료")
        
        print("\n=== 모든 컴포넌트 초기화 완료 ===\n")
        
    def load_calibration(self, calibration_file: str):
        """
        캘리브레이션 데이터 로드
        
        Args:
            calibration_file: 호모그래피 행렬 파일
        """
        self.homography.load_calibration(calibration_file)
        print(f"캘리브레이션 데이터 로드: {calibration_file}")
        
    def process_frame(self, frame: np.ndarray) -> FrameData:
        """
        단일 프레임 처리
        
        Args:
            frame: 입력 프레임 (BGR)
            
        Returns:
            FrameData
        """
        self.frame_count += 1
        
        # 1. 객체 탐지
        detections = self.detector.detect(frame)
        
        # 2. 객체 추적
        tracked_objects = self.tracker.update(detections, frame)
        
        # 3. 호모그래피 변환 (2D → 월드 좌표)
        world_coords = []
        if self.homography.homography_matrix is not None:
            for obj in tracked_objects:
                world_x, world_y = self.homography.pixel_to_world(*obj.center)
                world_coords.append({
                    'track_id': obj.track_id,
                    'label': obj.label,
                    'world_x': world_x,
                    'world_y': world_y,
                    'pixel_x': obj.center[0],
                    'pixel_y': obj.center[1]
                })
                
        # 4. 프레임 데이터 생성
        frame_data = FrameData(
            frame_number=self.frame_count,
            timestamp=self.frame_count / 30.0,  # 30 FPS 가정
            detections=detections,
            tracked_objects=tracked_objects,
            world_coordinates=world_coords
        )
        
        return frame_data
        
    def process_video(self, 
                     video_path: str,
                     output_dir: str = 'data/processed',
                     max_frames: Optional[int] = None):
        """
        비디오 처리
        
        Args:
            video_path: 입력 비디오 경로
            output_dir: 출력 디렉토리
            max_frames: 최대 처리 프레임 수 (None 이면 전체)
        """
        print(f"\n비디오 처리 시작: {video_path}")
        
        # 비디오 캡처
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오를 열 수 없습니다: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"FPS: {fps}, 총 프레임: {total_frames}")
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_data = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 최대 프레임 수 체크
            if max_frames is not None and frame_count >= max_frames:
                break
                
            # 프레임 처리
            frame_data = self.process_frame(frame)
            processed_data.append(frame_data)
            
            # 진행 상황 표시
            if frame_count % 100 == 0:
                print(f"처리 중... {frame_count}/{total_frames if max_frames is None else max_frames}")
                
            frame_count += 1
            
        cap.release()
        
        # 결과 저장
        result_file = output_path / 'processed_data.pkl'
        import pickle
        with open(result_file, 'wb') as f:
            pickle.dump(processed_data, f)
            
        print(f"\n처리 완료: {frame_count} 프레임")
        print(f"결과 저장: {result_file}")
        
        return processed_data
        
    def visualize_tracking(self, frame: np.ndarray, frame_data: FrameData) -> np.ndarray:
        """
        추적 결과 시각화
        
        Args:
            frame: 입력 프레임
            frame_data: 프레임 데이터
            
        Returns:
            시각화된 프레임
        """
        vis_frame = frame.copy()
        
        # 경기장 라인 오버레이
        if self.homography.homography_matrix is not None:
            vis_frame = self.homography.draw_field_overlay(vis_frame)
            
        # 추적 객체 표시
        color_map = {}
        for obj in frame_data.tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            
            # ID 기반으로 색상 생성
            if obj.track_id not in color_map:
                np.random.seed(obj.track_id * 42)
                color_map[obj.track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
                
            color = color_map[obj.track_id]
            
            # 바운딩 박스
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 레이블
            label = f"ID:{obj.track_id} {obj.label}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        return vis_frame
        
    def export_trajectory(self, 
                         processed_data: List[FrameData],
                         output_file: str = 'trajectories.csv'):
        """
        이동 경로 CSV 내보내기
        
        Args:
            processed_data: 처리된 프레임 데이터 리스트
            output_file: 출력 CSV 파일
        """
        import csv
        
        trajectories = {}
        
        # 트랙 ID 별 궤적 수집
        for frame_data in processed_data:
            for coord in frame_data.world_coordinates:
                track_id = coord['track_id']
                if track_id not in trajectories:
                    trajectories[track_id] = []
                    
                trajectories[track_id].append({
                    'frame': frame_data.frame_number,
                    'timestamp': frame_data.timestamp,
                    'world_x': coord['world_x'],
                    'world_y': coord['world_y'],
                    'label': coord['label']
                })
                
        # CSV 작성
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id', 'frame', 'timestamp', 'world_x', 'world_y', 'label'])
            
            for track_id, traj in trajectories.items():
                for point in traj:
                    writer.writerow([
                        track_id,
                        point['frame'],
                        point['timestamp'],
                        point['world_x'],
                        point['world_y'],
                        point['label']
                    ])
                    
        print(f"궤적 데이터 저장: {output_file}")
        print(f"총 {len(trajectories)}개 객체, {sum(len(t) for t in trajectories.values())}개 포인트")


if __name__ == '__main__':
    # 파이프라인 테스트
    pipeline = Soccer3DPipeline()
    
    # 컴포넌트 초기화 (실제 사용시에는 모델 파일 필요)
    try:
        pipeline.initialize_components()
        print("\n파이프라인 초기화 성공!")
    except Exception as e:
        print(f"\n초기화 중 오류 발생: {e}")
        print("실제 사용시에는 YOLO 모델 파일과 캘리브레이션 데이터가 필요합니다.")
        
    # 더미 데이터로 테스트
    print("\n=== 더미 데이터 테스트 ===")
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 호모그래피 설정 (더미)
    pixel_corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
    pipeline.homography.compute_from_field_lines(pixel_corners)
    
    print("테스트 완료!")
