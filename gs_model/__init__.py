"""
3D Gaussian Splatting 모델 패키지
"""

from .canonical_gs import CanonicalGaussianSpace, HumanGaussianTemplate, GaussianParams
# from .deformation_mlp import DeformationMLP, DynamicGaussianRenderer
# from .renderer import GaussianSplatRenderer

__all__ = [
    'CanonicalGaussianSpace',
    'HumanGaussianTemplate',
    'GaussianParams'
]
