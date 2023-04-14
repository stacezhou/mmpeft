from mmengine.registry import MODELS
from ..builder import build_peft
import types

def build(self, cfg: dict, *args, **kwargs):
    """Build an instance.

    Build an instance by calling :attr:`build_func`.

    Args:
        cfg (dict): Config dict needs to be built.

    Returns:
        Any: The constructed object.

    Examples:
        >>> from mmengine import Registry
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     def __init__(self, depth, stages=4):
        >>>         self.depth = depth
        >>>         self.stages = stages
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = MODELS.build(cfg)
    """
    peft_cfg = None
    if 'peft' in cfg:
        peft_cfg = cfg.pop('peft')
        peft = build_peft(peft_cfg)

    model =  self.build_func(cfg, *args, **kwargs, registry=self)

    if peft_cfg is not None:
        model = peft(model)

    return model

## wrap MODELS.build to support custom peft methods
MODELS.build = types.MethodType(build, MODELS)