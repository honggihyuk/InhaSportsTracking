"""
Deformation MLP 모듈
시간 및 포즈 정보를 기반으로 Canonical Gaussian 을 변형하는 네트워크
Dynamic 3DGS 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DeformationOutput:
    """MLP 변형 출력"""
    delta_means: torch.Tensor      # 위치 오프셋 (N, 3)
    delta_rotations: torch.Tensor  # 회전 델타 (N, 4)
    delta_scales: torch.Tensor     # 크기 델타 (N, 3)
    delta_opacities: torch.Tensor  # 불투명도 델타 (N, 1)
    

class PositionalEncoding(nn.Module):
    """위치 인코딩 - 고주파수 특징 표현"""
    
    def __init__(self, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (..., D)
            
        Returns:
            위치 인코딩된 텐서 (..., D * 2 * num_freqs)
        """
        encoded = []
        for freq in self.freq_bands.to(x.device):
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class DeformationMLP(nn.Module):
    """
    변형 네트워크 (Deformation Network)
    
    입력: 표준 가우시안 위치, 시간, 포즈 정보
    출력: 가우시안 변형량 (위치, 회전, 크기, 불투명도)
    """
    
    def __init__(self,
                 input_dim: int = 3 + 1 + 16,  # position + time + pose
                 feature_dim: int = 256,
                 num_layers: int = 8,
                 output_dim: int = 11,  # 3 + 4 + 3 + 1
                 use_positional_encoding: bool = True,
                 num_freqs: int = 10):
        """
        Args:
            input_dim: 입력 특징 차원
            feature_dim: 은닉층 차원
            num_layers: 레이어 개수
            output_dim: 출력 차원
            use_positional_encoding: 위치 인코딩 사용 여부
            num_freqs: 위치 인코딩 주파수 대역 개수
        """
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(num_freqs)
            enc_dim = input_dim * 2 * num_freqs
        else:
            enc_dim = input_dim
            
        # MLP 레이어 구성
        layers = []
        layers.append(nn.Linear(enc_dim, feature_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(feature_dim, feature_dim))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Linear(feature_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 회전은 쿼터니언으로 정규화 필요
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, 
                positions: torch.Tensor,
                time: torch.Tensor,
                pose_features: Optional[torch.Tensor] = None) -> DeformationOutput:
        """
        순전파
        
        Args:
            positions: 표준 가우시안 위치 (N, 3)
            time: 시간 정보 (N, 1) 또는 스칼라
            pose_features: 포즈 특징 (N, 16) - SMPL 파라미터 등
            
        Returns:
            DeformationOutput
        """
        N = len(positions)
        
        # 시간 정보 확장
        if time.dim() == 0:
            time = time.unsqueeze(0).expand(N, 1)
        elif time.dim() == 1:
            time = time.unsqueeze(1)
            
        # 입력 통합
        inputs = [positions, time]
        if pose_features is not None:
            inputs.append(pose_features)
        else:
            # 포즈 정보가 없으면 0 으로 채움
            inputs.append(torch.zeros(N, 16, device=positions.device))
            
        x = torch.cat(inputs, dim=-1)
        
        # 위치 인코딩 적용
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
            
        # MLP 통과
        output = self.mlp(x)
        
        # 출력 분할
        delta_means = output[:, :3]
        delta_rotations = output[:, 3:7]
        delta_scales = output[:, 7:10]
        delta_opacities = output[:, 10:11]
        
        # 회전 쿼터니언 정규화
        delta_rotations = F.normalize(delta_rotations, dim=-1)
        
        return DeformationOutput(
            delta_means=delta_means,
            delta_rotations=delta_rotations,
            delta_scales=delta_scales,
            delta_opacities=delta_opacities
        )


class PoseConditionedDeformer(nn.Module):
    """
    포즈 조건부 변형기
    
    선수의 3D 포즈 (SMPL 파라미터) 를 조건으로 사용하여
    더 정확한 인간 동작 변형 생성
    """
    
    def __init__(self,
                 smpl_dim: int = 76,  # SMPL pose + shape 파라미터
                 canonical_gaussians: int = 10000,
                 feature_dim: int = 512,
                 num_heads: int = 8):
        """
        Args:
            smpl_dim: SMPL 파라미터 차원
            canonical_gaussians: 표준 가우시안 개수
            feature_dim: 특징 차원
            num_heads: 어텐션 헤드 개수
        """
        super().__init__()
        
        self.canonical_gaussians = canonical_gaussians
        
        # SMPL 인코더
        self.smpl_encoder = nn.Sequential(
            nn.Linear(smpl_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 가우시안 임베딩
        self.gaussian_embedding = nn.Embedding(canonical_gaussians, feature_dim)
        
        # 크로스 어텐션
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 변형 디코더
        self.deformation_decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 11)  # 3 + 4 + 3 + 1
        )
        
    def forward(self,
                smpl_params: torch.Tensor,
                gaussian_indices: Optional[torch.LongTensor] = None) -> DeformationOutput:
        """
        Args:
            smpl_params: SMPL 파라미터 (B, 76)
            gaussian_indices: 샘플링할 가우시안 인덱스 (선택사항)
            
        Returns:
            DeformationOutput
        """
        B = len(smpl_params)
        N = self.canonical_gaussians if gaussian_indices is None else len(gaussian_indices)
        
        # SMPL 인코딩
        smpl_features = self.smpl_encoder(smpl_params)  # (B, feature_dim)
        smpl_features = smpl_features.unsqueeze(1)  # (B, 1, feature_dim)
        
        # 가우시안 임베딩 조회
        if gaussian_indices is None:
            indices = torch.arange(N, device=smpl_params.device).unsqueeze(0).expand(B, -1)
        else:
            indices = gaussian_indices.unsqueeze(0).expand(B, -1)
            
        gaussian_features = self.gaussian_embedding(indices)  # (B, N, feature_dim)
        
        # 크로스 어텐션
        attended_features, _ = self.cross_attention(
            query=gaussian_features,
            key=smpl_features,
            value=smpl_features
        )  # (B, N, feature_dim)
        
        # 변형 예측
        deformation = self.deformation_decoder(attended_features)  # (B, N, 11)
        
        # 배치 평균 (단일 프레임용)
        if B == 1:
            deformation = deformation.squeeze(0)
        else:
            deformation = deformation.mean(dim=0)
            
        # 출력 분할
        delta_means = deformation[:, :3]
        delta_rotations = F.normalize(deformation[:, 3:7], dim=-1)
        delta_scales = deformation[:, 7:10]
        delta_opacities = deformation[:, 10:11]
        
        return DeformationOutput(
            delta_means=delta_means,
            delta_rotations=delta_rotations,
            delta_scales=delta_scales,
            delta_opacities=delta_opacities
        )


class DynamicGaussianRenderer(nn.Module):
    """
    동적 가우시안 렌더러
    
    Canonical Gaussian + Deformation MLP 를 결합하여
    시간에 따른 동적 3D 장면 렌더링
    """
    
    def __init__(self,
                 canonical_space,
                 deformation_mlp: DeformationMLP,
                 device: str = 'cuda'):
        """
        Args:
            canonical_space: CanonicalGaussianSpace 인스턴스
            deformation_mlp: DeformationMLP 인스턴스
            device: 연산 장치
        """
        super().__init__()
        
        self.canonical_space = canonical_space
        self.deformation_mlp = deformation_mlp
        self.device = device
        
        # 렌더링 함수 (diff-gaussian-rasterization 연동용)
        self.rasterizer = None
        
    def get_deformed_gaussians(self,
                               time: torch.Tensor,
                               pose_features: Optional[torch.Tensor] = None) -> 'GaussianParams':
        """
        변형된 가우시안 파라미터 계산
        
        Args:
            time: 시간 정보
            pose_features: 포즈 특징
            
        Returns:
            변형된 GaussianParams
        """
        # 표준 가우시안 조회
        canonical_params = self.canonical_space.get_params()
        
        # 변형 예측
        deformation = self.deformation_mlp(
            positions=canonical_params.means,
            time=time,
            pose_features=pose_features
        )
        
        # 변형 적용
        deformed_means = canonical_params.means + deformation.delta_means
        deformed_rotations = self._quaternion_multiply(
            canonical_params.rotations,
            deformation.delta_rotations
        )
        deformed_scales = canonical_params.scales + deformation.delta_scales
        deformed_opacities = canonical_params.opacities + deformation.delta_opacities
        
        from gs_model.canonical_gs import GaussianParams
        
        return GaussianParams(
            means=deformed_means,
            scales=deformed_scales,
            rotations=deformed_rotations,
            opacities=deformed_opacities,
            sh_coeffs=canonical_params.sh_coeffs
        )
        
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """쿼터니언 곱셈"""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)
        
    def set_rasterizer(self, rasterizer):
        """렌더러 설정"""
        self.rasterizer = rasterizer
        
    def render(self,
               time: torch.Tensor,
               viewpoint_camera,
               pose_features: Optional[torch.Tensor] = None):
        """
        동적 장면 렌더링
        
        Args:
            time: 시간 정보
            viewpoint_camera: 시점 카메라 파라미터
            pose_features: 포즈 특징
            
        Returns:
            rendered_image, depth_map, etc.
        """
        if self.rasterizer is None:
            raise RuntimeError("래스터라이저가 설정되지 않았습니다.")
            
        # 변형된 가우시안 획득
        deformed_params = self.get_deformed_gaussians(time, pose_features)
        
        # 렌더링 수행 (구현은 diff-gaussian-rasterization 연동 필요)
        # rendered = self.rasterizer(deformed_params, viewpoint_camera)
        
        raise NotImplementedError("렌더링 구현은 diff-gaussian-rasterization 연동 필요")


if __name__ == '__main__':
    # 테스트 코드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Deformation MLP 테스트
    mlp = DeformationMLP(
        input_dim=20,
        feature_dim=256,
        num_layers=8
    ).to(device)
    
    # 더미 입력
    N = 1000
    positions = torch.randn(N, 3, device=device)
    time = torch.tensor([0.5], device=device).expand(N, 1)  # 모든 가우시안에 동일한 시간
    pose_features = torch.randn(N, 16, device=device)
    
    # 순전파
    output = mlp(positions, time, pose_features)
    
    print(f"입력: {positions.shape}")
    print(f"출력 delta_means: {output.delta_means.shape}")
    print(f"출력 delta_rotations: {output.delta_rotations.shape}")
    print(f"출력 delta_scales: {output.delta_scales.shape}")
    print(f"출력 delta_opacities: {output.delta_opacities.shape}")
    
    # PoseConditionedDeformer 테스트
    deformer = PoseConditionedDeformer(
        smpl_dim=76,
        canonical_gaussians=1000,
        feature_dim=256
    ).to(device)
    
    smpl_params = torch.randn(1, 76, device=device)
    output2 = deformer(smpl_params)
    
    print(f"\nPoseConditionedDeformer 출력:")
    print(f"delta_means: {output2.delta_means.shape}")
