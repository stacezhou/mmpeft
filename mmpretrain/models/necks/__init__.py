# Copyright (c) OpenMMLab. All rights reserved.
from .cae_neck import CAENeck
from .densecl_neck import DenseCLNeck
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .linear_neck import LinearNeck
from .mae_neck import ClsBatchNormNeck, MAEPretrainDecoder
from .milan_neck import MILANPretrainDecoder
from .mixmim_neck import MixMIMPretrainDecoder
from .nonlinear_neck import NonLinearNeck

__all__ = [
    'GlobalAveragePooling',
    'GeneralizedMeanPooling',
    'LinearNeck',
    'CAENeck',
    'DenseCLNeck',
    'MAEPretrainDecoder',
    'ClsBatchNormNeck',
    'MILANPretrainDecoder',
    'MixMIMPretrainDecoder',
    'NonLinearNeck',
]
