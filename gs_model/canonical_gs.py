"""
3D Gaussian Splatting canonical space 정의 모듈
표준 공간에서의 가우시안 표현 및 관리
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GaussianParams:
    """3D 가우시안 파라미터"""
    means: torch.Tensor      # 위치 (N, 3)
    scales: torch.Tensor     # 크기 (N, 3)
    rotations: torch.Tensor  # 회전 (N, 4) - 쿼터니언
    opacities: torch.Tensor  # 불투명도 (N, 1)
    sh_coeffs: torch.Tensor  # 구면조화 계수 (N, K, 3)
    
    def to(self, device: str):
        """장치를 이동"""
        return GaussianParams(
            means=self.means.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            opacities=self.opacities.to(device),
            sh_coeffs=self.sh_coeffs.to(device)
        )
        
    def __len__(self):
        return len(self.means)


class CanonicalGaussianSpace:
    """
    표준 공간 (Canonical Space) 의 3D 가우시안 모델
    
    선수 및 경기장의 기준 3DGS 모델을 정의하고 관리
    """
    
    def __init__(self, 
                 num_gaussians: int = 10000,
                 sh_degree: int = 3,
                 feature_dim: int = 256,
                 device: str = 'cuda'):
        """
        Args:
            num_gaussians: 가우시안 개수
            sh_degree: 구면조화 차수
            feature_dim: 특징 벡터 차원
            device: 연산 장치
        """
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        self.feature_dim = feature_dim
        self.device = device
        
        # 구면조화 계수 개수
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        
        # 가우시안 파라미터 초기화
        self.gaussian_params: Optional[GaussianParams] = None
        self._initialize_gaussians()
        
    def _initialize_gaussians(self):
        """가우시안 파라미터 초기화"""
        # 위치: 정규 분포로 초기화
        means = torch.randn(self.num_gaussians, 3, device=self.device) * 0.5
        
        # 크기: 로그 정규 분포로 초기화
        scales = torch.rand(self.num_gaussians, 3, device=self.device) * 0.1 + 0.01
        scales = torch.log(scales)
        
        # 회전: 단위 쿼터니언으로 초기화
        rotations = torch.zeros(self.num_gaussians, 4, device=self.device)
        rotations[:, 0] = 1.0  # w 성분만 1
        
        # 불투명도: 시그모이드 역함수로 초기화
        opacities = torch.rand(self.num_gaussians, 1, device=self.device) * 0.5 + 0.3
        opacities = torch.log(opacities / (1 - opacities))  # inverse sigmoid
        
        # 구면조화 계수: 작은 값으로 초기화
        sh_coeffs = torch.zeros(
            self.num_gaussians, 
            self.num_sh_coeffs, 
            3, 
            device=self.device
        )
        sh_coeffs[:, 0, :] = 0.5  # DC 성분만 약간 설정
        
        self.gaussian_params = GaussianParams(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs
        )
        
        print(f"Canonical Gaussian Space 초기화 완료: {self.num_gaussians} 가우시안")
        
    def get_params(self) -> GaussianParams:
        """가우시안 파라미터 반환"""
        return self.gaussian_params
        
    def update_params(self, new_params: GaussianParams):
        """가우시안 파라미터 업데이트"""
        self.gaussian_params = new_params
        
    def sample_gaussians(self, indices: torch.LongTensor) -> GaussianParams:
        """
        특정 인덱스의 가우시안 샘플링
        
        Args:
            indices: 샘플링할 인덱스
            
        Returns:
            샘플링된 GaussianParams
        """
        return GaussianParams(
            means=self.gaussian_params.means[indices],
            scales=self.gaussian_params.scales[indices],
            rotations=self.gaussian_params.rotations[indices],
            opacities=self.gaussian_params.opacities[indices],
            sh_coeffs=self.gaussian_params.sh_coeffs[indices]
        )
        
    def add_gaussians(self, new_means: torch.Tensor,
                     new_scales: torch.Tensor,
                     new_rotations: torch.Tensor,
                     new_opacities: torch.Tensor,
                     new_sh_coeffs: torch.Tensor):
        """
        새로운 가우시안 추가
        
        Args:
            new_means: 새로운 위치 (M, 3)
            new_scales: 새로운 크기 (M, 3)
            new_rotations: 새로운 회전 (M, 4)
            new_opacities: 새로운 불투명도 (M, 1)
            new_sh_coeffs: 새로운 SH 계수 (M, K, 3)
        """
        M = len(new_means)
        
        # 기존 파라미터와 연결
        self.gaussian_params = GaussianParams(
            means=torch.cat([self.gaussian_params.means, new_means], dim=0),
            scales=torch.cat([self.gaussian_params.scales, new_scales], dim=0),
            rotations=torch.cat([self.gaussian_params.rotations, new_rotations], dim=0),
            opacities=torch.cat([self.gaussian_params.opacities, new_opacities], dim=0),
            sh_coeffs=torch.cat([self.gaussian_params.sh_coeffs, new_sh_coeffs], dim=0)
        )
        
        self.num_gaussians = len(self.gaussian_params.means)
        print(f"가우시안 추가 완료: 총 {self.num_gaussians}개")
        
    def prune_gaussians(self, opacity_threshold: float = 0.005,
                       scale_threshold: float = 0.01):
        """
        불필요한 가우시안 제거
        
        Args:
            opacity_threshold: 불투명도 임계값
            scale_threshold: 크기 임계값
        """
        opacities = torch.sigmoid(self.gaussian_params.opacities)
        scales = torch.exp(self.gaussian_params.scales)
        
        # 제거 조건
        mask = (opacities.squeeze() > opacity_threshold) & \
               (scales.max(dim=1).values < scale_threshold)
               
        if mask.sum() < len(mask):
            self.gaussian_params = GaussianParams(
                means=self.gaussian_params.means[mask],
                scales=self.gaussian_params.scales[mask],
                rotations=self.gaussian_params.rotations[mask],
                opacities=self.gaussian_params.opacities[mask],
                sh_coeffs=self.gaussian_params.sh_coeffs[mask]
            )
            self.num_gaussians = len(self.gaussian_params.means)
            print(f"가우시안 가지치기 완료: {self.num_gaussians}개 남음")
            
    def densify_gaussians(self, grad_threshold: float = 0.0002):
        """
        가우시안 밀집화 (복제 및 분할)
        
        Args:
            grad_threshold: 그래디언트 임계값
        """
        # 구현은 학습 루프에서 수행
        pass
        
    def save(self, filepath: str):
        """모델 저장"""
        checkpoint = {
            'num_gaussians': self.num_gaussians,
            'sh_degree': self.sh_degree,
            'feature_dim': self.feature_dim,
            'gaussian_params': {
                'means': self.gaussian_params.means.cpu().numpy(),
                'scales': self.gaussian_params.scales.cpu().numpy(),
                'rotations': self.gaussian_params.rotations.cpu().numpy(),
                'opacities': self.gaussian_params.opacities.cpu().numpy(),
                'sh_coeffs': self.gaussian_params.sh_coeffs.cpu().numpy()
            }
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        print(f"Canonical Gaussian Space 저장 완료: {filepath}")
        
    def load(self, filepath: str):
        """모델 로드"""
        import pickle
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
            
        self.num_gaussians = checkpoint['num_gaussians']
        self.sh_degree = checkpoint['sh_degree']
        self.feature_dim = checkpoint['feature_dim']
        
        params = checkpoint['gaussian_params']
        self.gaussian_params = GaussianParams(
            means=torch.from_numpy(params['means']).to(self.device),
            scales=torch.from_numpy(params['scales']).to(self.device),
            rotations=torch.from_numpy(params['rotations']).to(self.device),
            opacities=torch.from_numpy(params['opacities']).to(self.device),
            sh_coeffs=torch.from_numpy(params['sh_coeffs']).to(self.device)
        )
        
        print(f"Canonical Gaussian Space 로드 완료: {filepath}")


class HumanGaussianTemplate(CanonicalGaussianSpace):
    """
    인간 형태 템플릿 기반 Canonical Gaussian Space
    
    SMPL 메쉬 표면에 가우시안을 바인딩하여 인간 형태 표현
    """
    
    def __init__(self, 
                 smpl_vertices: Optional[np.ndarray] = None,
                 gaussians_per_vertex: int = 5,
                 **kwargs):
        """
        Args:
            smpl_vertices: SMPL 메쉬 정점 (V, 3)
            gaussians_per_vertex: 정점당 가우시안 개수
        """
        super().__init__(**kwargs)
        
        self.gaussians_per_vertex = gaussians_per_vertex
        self.smpl_template = smpl_vertices
        
        if smpl_vertices is not None:
            self._initialize_from_smpl(smpl_vertices)
            
    def _initialize_from_smpl(self, vertices: np.ndarray):
        """SMPL 메쉬 기반으로 가우시안 초기화"""
        V = len(vertices)
        total_gaussians = V * self.gaussians_per_vertex
        
        # 각 정점에 가우시안 할당
        means = []
        for i, vertex in enumerate(vertices):
            for j in range(self.gaussians_per_vertex):
                # 정점 주변에 약간의 오프셋을 주어 가우시안 배치
                offset = np.random.randn(3) * 0.05
                means.append(vertex + offset)
                
        means = np.array(means, dtype=np.float32)
        
        # 나머지 파라미터는 부모 클래스 방식대로 초기화
        self.num_gaussians = total_gaussians
        
        self.gaussian_params = GaussianParams(
            means=torch.from_numpy(means).to(self.device),
            scales=torch.rand(total_gaussians, 3, device=self.device) * 0.02 + 0.01,
            rotations=torch.eye(4, device=self.device).unsqueeze(0).expand(total_gaussians, -1),
            opacities=torch.ones(total_gaussians, 1, device=self.device) * 0.8,
            sh_coeffs=torch.zeros(total_gaussians, self.num_sh_coeffs, 3, device=self.device)
        )
        
        print(f"Human Gaussian Template 초기화 완료: {total_gaussians} 가우시안")


if __name__ == '__main__':
    # 테스트 코드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    canonical_space = CanonicalGaussianSpace(
        num_gaussians=1000,
        sh_degree=3,
        device=device
    )
    
    params = canonical_space.get_params()
    print(f"가우시안 개수: {len(params)}")
    print(f"위치.shape: {params.means.shape}")
    print(f"크기.shape: {params.scales.shape}")
    print(f"회전.shape: {params.rotations.shape}")
