# Copyright (c) OpenMMLab. All rights reserved.
from .class_num_check_hook import ClassNumCheckHook
from .densecl_hook import DenseCLHook
from .ema_hook import EMAHook
from .margin_head_hooks import SetAdaptiveMarginsHook
from .precise_bn_hook import PreciseBNHook
from .retriever_hooks import PrepareProtoBeforeValLoopHook
from .simsiam_hook import SimSiamHook
from .swav_hook import SwAVHook
from .switch_recipe_hook import SwitchRecipeHook
from .visualization_hook import VisualizationHook
from .clip_set_prompts import ClipSetPrompts
from .nni_hook import NNIHook

__all__ = [
    'ClassNumCheckHook', 'PreciseBNHook', 'VisualizationHook',
    'SwitchRecipeHook', 'PrepareProtoBeforeValLoopHook',
    'SetAdaptiveMarginsHook', 'EMAHook', 'SimSiamHook', 'DenseCLHook',
    'ClipSetPrompts',
    'NNIHook',
    'SwAVHook'
]
