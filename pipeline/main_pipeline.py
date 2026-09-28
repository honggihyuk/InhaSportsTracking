"""
통합 파이프라인 모듈
축구 영상 → 탐지/추적 → 3D 매핑 → Dynamic 3DGS 렌더링
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml

from tracking.detector import Detection, RoboflowSoccerDetector, SoccerDetector
from tracking.tracker import SoccerTracker, TrackedObject
from tracking.homography import HomographyTransformer
from tracking.coordinate_mapper import CoordinateMapper


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
    
    def __init__(self,
                 config_path: str = 'configs/model_config.yaml',
                 detector=None,
                 tracker: Optional[SoccerTracker] = None,
                 homography: Optional[HomographyTransformer] = None,
                 coord_mapper: Optional[CoordinateMapper] = None):
        """
        Args:
            config_path: 설정 파일 경로
            detector: detect(frame) -> List[Detection] 를 가진 객체 주입
                      (None 이면 initialize_components() 에서 설정대로 YOLO 로드)
            tracker / homography / coord_mapper: 컴포넌트 주입 (None 이면 설정 기반 기본값)
        """
        self.config = self._load_config(config_path)

        # 가벼운 컴포넌트는 즉시 생성, 무거운 탐지기/3DGS 는 initialize_components() 에서 생성
        self.detector = detector
        self.tracker = tracker or self.make_tracker()
        self.homography = homography or HomographyTransformer()
        self.coord_mapper = coord_mapper or CoordinateMapper()
        self.canonical_space = None
        self.deformation_mlp = None
        self.renderer = None
        
        # 상태 변수
        self.frame_count = 0
        # 설정 파일의 detection.device 를 참조하거나, 명시적 device 설정 사용
        det_config = self.config.get('detection', {})
        self.device = det_config.get('device', 'cpu')  # GPU 가 없을 경우 cpu 로 기본값 변경
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"설정 파일 로드 완료: {config_path}")
        else:
            print(f"설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            config = {
                'detector': {
                    'confidence_threshold': 0.5,
                    'model_path': 'models/yolo_soccer.pt'
                },
                'tracking': {
                    'type': 'bytetrack',
                    'track_threshold': 0.3
                },
                'gaussian': {
                    'num_gaussians': 10000,
                    'sh_degree': 3
                }
            }
        return config
        
    def make_tracker(self) -> SoccerTracker:
        """설정(tracking 섹션) 기반 새 추적기 — 영상마다 ID 를 새로 시작할 때 사용"""
        # 설정 파일은 'tracking' 키를 사용 (이전 기본 설정의 'tracker' 도 호환)
        track_config = self.config.get('tracking', self.config.get('tracker', {}))
        return SoccerTracker(
            tracker_type=track_config.get('type', 'bytetrack'),
            track_threshold=track_config.get('track_threshold', 0.3),
            match_threshold=track_config.get('match_threshold', 0.8),
            max_age=track_config.get('max_age', 30),
        )

    def initialize_components(self):
        """주입되지 않은 무거운 컴포넌트(탐지기, 3DGS) 초기화"""
        print("\n=== 컴포넌트 초기화 ===")

        self.initialize_detector()
        self._initialize_gaussian()

    def initialize_detector(self):
        """탐지기 초기화 (주입되었으면 그대로 사용) - 설정에 따라 Roboflow 또는 기본 YOLO"""
        det_config = self.config.get('detection', {})
        use_roboflow = det_config.get('use_roboflow_models', False)

        if self.detector is not None:
            print("✓ 주입된 탐지기 사용")
        elif use_roboflow:
            roboflow_models = det_config.get('roboflow_models', {})
            self.detector = RoboflowSoccerDetector(
                players_model_path=roboflow_models.get('players'),
                ball_model_path=roboflow_models.get('ball'),
                field_model_path=roboflow_models.get('field'),
                confidence_threshold=det_config.get('confidence_threshold', 0.5),
                device=self.device,
                imgsz=det_config.get('img_size', 640),
                ball_imgsz=self.config.get('analysis', {}).get('ball_imgsz', 1280)
            )
            print("✓ RoboflowSoccerDetector 초기화 완료 (전문 모델)")
        else:
            # 기본 YOLO11 모델 사용
            model_path = det_config.get('model_path', 'models/yolo11s.pt')
            self.detector = SoccerDetector(
                players_model_path=model_path,
                confidence_threshold=det_config.get('confidence_threshold', 0.5),
                device=self.device,
                imgsz=det_config.get('img_size', 640),
                ball_imgsz=self.config.get('analysis', {}).get('ball_imgsz', 1280)
            )
            print("✓ SoccerDetector 초기화 완료 (기본 YOLO11)")
        
    def _initialize_gaussian(self):
        """3DGS 컴포넌트 초기화 (추적기 / 호모그래피 / 좌표 매퍼는 __init__ 에서 생성 또는 주입됨)"""
        # 3DGS: torch 가 필요하므로 여기서만 import (추적만 쓸 때는 torch 불필요)
        from gs_model.canonical_gs import CanonicalGaussianSpace
        from gs_model.deformation_mlp import DeformationMLP, DynamicGaussianRenderer

        # 5. Canonical Gaussian Space
        # 설정 파일은 gaussian_splatting.canonical_space 를 사용 (이전 기본 설정의 'gaussian' 도 호환)
        gs_config = self.config.get('gaussian_splatting', {}).get('canonical_space') or self.config.get('gaussian', {})
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
            world = self.homography.pixels_to_world([obj.center for obj in tracked_objects])
            for obj, (world_x, world_y) in zip(tracked_objects, world):
                world_coords.append({
                    'track_id': obj.track_id,
                    'label': obj.label,
                    'world_x': float(world_x),
                    'world_y': float(world_y),
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

        try:
            # 최대 프레임 수에 도달하면 읽기 전에 중단 (불필요한 디코딩 방지)
            while max_frames is None or frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_data.append(self.process_frame(frame))

                # 진행 상황 표시
                if frame_count % 100 == 0:
                    print(f"처리 중... {frame_count}/{total_frames if max_frames is None else max_frames}")

                frame_count += 1
        finally:
            cap.release()  # 처리 중 예외가 나도 비디오 핸들 해제
        
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
        for obj in frame_data.tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            # ID 기반 일관된 색상 (전역 난수 상태를 건드리지 않음)
            color = tuple(map(int, np.random.default_rng(obj.track_id * 42).integers(0, 255, 3)))
            
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Soccer 3D Digital Twin Pipeline')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--video', type=str, default=None,
                        help='처리할 비디오 파일 경로')
    args = parser.parse_args()
    
    # 파이프라인 초기화
    print("=== Soccer 3D Digital Twin Pipeline ===\n")
    pipeline = Soccer3DPipeline(config_path=args.config)
    
    # 컴포넌트 초기화 (실제 사용시에는 모델 파일 필요)
    try:
        pipeline.initialize_components()
        print("\n✅ 파이프라인 초기화 성공!")
        
        # 비디오 처리
        if args.video:
            print(f"\n🎬 비디오 처리 시작: {args.video}")
            processed_data = pipeline.process_video(args.video, max_frames=50)  # 테스트용 50 프레임만
            
            # 궤적 내보내기
            pipeline.export_trajectory(processed_data, 'trajectories.csv')
            print("\n✅ 처리 완료!")
        else:
            # 더미 데이터로 테스트
            print("\n=== 더미 데이터 테스트 ===")
            dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # 호모그래피 설정 (더미)
            pixel_corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
            pipeline.homography.compute_from_field_lines(pixel_corners)
            
            # 프레임 처리 테스트
            frame_data = pipeline.process_frame(dummy_frame)
            print(f"프레임 {frame_data.frame_number} 처리 완료")
            print(f"탐지된 객체: {len(frame_data.detections)}개")
            print(f"추적된 객체: {len(frame_data.tracked_objects)}개")
            
            print("\n✅ 테스트 완료!")
            
    except Exception as e:
        print(f"\n⚠️ 초기화 중 오류 발생: {e}")
        print("\n💡 실제 사용시에는 다음이 필요합니다:")
        print("   1. Roboflow API 키 설정 (.env 파일)")
        print("   2. python setup_roboflow.py 실행하여 데이터셋 다운로드")
        print("   3. 또는 configs/model_config.yaml 에서 모델 경로 수정")
        print("\n📝 빠른 시작 가이드:")
        print("   1. .env 파일에 ROBOFLOW_API_KEY 설정")
        print("   2. python setup_roboflow.py 실행")
        print("   3. python -m pipeline.main_pipeline --config configs/model_config.yaml --video <영상경로>")
