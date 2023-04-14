# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook

from mmpretrain.registry import HOOKS
from mmpretrain.utils import get_ori_model


@HOOKS.register_module()
class ClipSetPrompts(Hook):
    def before_train_epoch(self, runner) -> None:
        get_ori_model(runner.model).set_prompts(runner.train_dataloader.dataset.prompts)
    def before_val_epoch(self, runner) -> None:
        get_ori_model(runner.model).set_prompts(runner.val_dataloader.dataset.prompts)
    def before_test_epoch(self, runner) -> None:
        get_ori_model(runner.model).set_prompts(runner.test_dataloader.dataset.prompts)
