"""
샘플 테스트 스크립트
실제 축구 영상이 없을 경우 더미 데이터로 파이프라인 테스트
"""

import numpy as np
import cv2
from pathlib import Path

# PYTHONPATH 설정
import sys
sys.path.insert(0, '/workspace')

from tracking.homography import HomographyTransformer
from gs_model.canonical_gs import CanonicalGaussianSpace
from gs_model.deformation_mlp import DeformationMLP, DynamicGaussianRenderer

def test_homography():
    """호모그래피 변환 테스트"""
    print("\n=== 호모그래피 변환 테스트 ===")
    
    transformer = HomographyTransformer()
    
    # 더미 코너 포인트 (1280x720 이미지 가정)
    pixel_corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
    success = transformer.compute_from_field_lines(pixel_corners)
    
    if success:
        # 테스트 변환
        test_points = [
            (640, 360),  # 이미지 중심
            (100, 100),  # top_left
            (1180, 620)  # bottom_right
        ]
        
        for px, py in test_points:
            world_x, world_y = transformer.pixel_to_world(px, py)
            print(f"픽셀 ({px:4d}, {py:4d}) -> 월드 ({world_x:7.2f}, {world_y:7.2f}) 미터")
            
        # 역변환 테스트
        print("\n역변환 테스트:")
        world_coord = (0.0, 0.0)  # 경기장 중심
        px, py = transformer.world_to_pixel(*world_coord)
        print(f"월드 ({world_coord[0]:5.1f}, {world_coord[1]:5.1f}) -> 픽셀 ({px:.1f}, {py:.1f})")
        
    return transformer


def test_gaussian_space():
    """Canonical Gaussian Space 테스트"""
    print("\n=== Canonical Gaussian Space 테스트 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    canonical_space = CanonicalGaussianSpace(
        num_gaussians=1000,
        sh_degree=3,
        device=device
    )
    
    params = canonical_space.get_params()
    print(f"가우시안 개수: {len(params)}")
    print(f"위치 범위: [{params.means.min():.3f}, {params.means.max():.3f}]")
    print(f"크기 범위: [{torch.exp(params.scales).min():.3f}, {torch.exp(params.scales).max():.3f}]")
    
    return canonical_space


def test_deformation_mlp():
    """Deformation MLP 테스트"""
    print("\n=== Deformation MLP 테스트 ===")
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mlp = DeformationMLP(
        input_dim=20,
        feature_dim=256,
        num_layers=8,
        use_positional_encoding=True
    ).to(device)
    
    # 더미 입력 생성
    N = 1000
    positions = torch.randn(N, 3, device=device)
    time = torch.tensor([0.5], device=device).expand(N, 1)
    pose_features = torch.randn(N, 16, device=device)
    
    # 순전파
    with torch.no_grad():
        output = mlp(positions, time, pose_features)
    
    print(f"입력 위치: {positions.shape}")
    print(f"출력 변형량:")
    print(f"  - Δmeans: {output.delta_means.shape}, 범위: [{output.delta_means.min():.3f}, {output.delta_means.max():.3f}]")
    print(f"  - Δrotations: {output.delta_rotations.shape}")
    print(f"  - Δscales: {output.delta_scales.shape}")
    print(f"  - Δopacities: {output.delta_opacities.shape}")
    
    return mlp


def create_sample_video(output_path='data/raw/sample_video.mp4', duration=5):
    """샘플 비디오 생성 (더미)"""
    print(f"\n=== 샘플 비디오 생성 ===")
    
    # 1280x720, 30 FPS 비디오
    width, height = 1280, 720
    fps = 30
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("비디오 작성기를 열 수 없습니다.")
        return False
    
    print(f"비디오 생성 중... {total_frames} 프레임")
    
    for frame_idx in range(total_frames):
        # 초록색 배경 (축구장)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Forest Green
        
        # 경기장 라인 그리기 (흰색)
        margin = 100
        cv2.rectangle(frame, (margin, margin), (width-margin, height-margin), (255, 255, 255), 2)
        cv2.line(frame, (width//2, margin), (width//2, height-margin), (255, 255, 255), 2)
        
        # 센터 서클
        center = (width//2, height//2)
        cv2.circle(frame, center, 60, (255, 255, 255), 2)
        
        # 움직이는 공 (노란색)
        t = frame_idx / fps
        ball_x = int(width//2 + 200 * np.sin(t * 2))
        ball_y = int(height//2 + 100 * np.cos(t * 3))
        cv2.circle(frame, (ball_x, ball_y), 15, (0, 255, 255), -1)
        
        # 선수들 (파란색, 빨간색)
        np.random.seed(frame_idx % 100)
        for i in range(10):
            # 파란 팀
            bx = int(margin + np.random.rand() * (width - 2*margin))
            by = int(margin + np.random.rand() * (height//2 - 2*margin))
            cv2.circle(frame, (bx, by), 10, (255, 0, 0), -1)
            
            # 빨간 팀
            rx = int(margin + np.random.rand() * (width - 2*margin))
            ry = int(height//2 + margin + np.random.rand() * (height//2 - 2*margin))
            cv2.circle(frame, (rx, ry), 10, (0, 0, 255), -1)
        
        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"  프레임 {frame_idx}/{total_frames}")
    
    out.release()
    print(f"샘플 비디오 저장 완료: {output_path}")
    return True


if __name__ == '__main__':
    import torch
    
    print("=" * 60)
    print("Soccer 3D Digital Twin - 모듈 테스트")
    print("=" * 60)
    
    # 1. 호모그래피 테스트
    transformer = test_homography()
    
    # 2. Gaussian Space 테스트
    canonical_space = test_gaussian_space()
    
    # 3. Deformation MLP 테스트
    deformation_mlp = test_deformation_mlp()
    
    # 4. 샘플 비디오 생성
    create_sample_video()
    
    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)
