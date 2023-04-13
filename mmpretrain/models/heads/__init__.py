# Copyright (c) OpenMMLab. All rights reserved.
from .linear_head import LinearClsHead
from .mae_head import MAEPretrainHead
from .margin_head import ArcFaceClsHead
from .mim_head import MIMHead
from .mixmim_head import MixMIMPretrainHead
from .multi_label_cls_head import MultiLabelClsHead
from .multi_label_csra_head import CSRAClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .multi_task_head import MultiTaskHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    'LinearClsHead',
    'MAEPretrainHead',
    'ArcFaceClsHead',
    'MIMHead',
    'MixMIMPretrainHead',
    'MultiLabelClsHead',
    'CSRAClsHead',
    'MultiLabelLinearClsHead',
    'MultiTaskHead',
    'VisionTransformerClsHead'
]
