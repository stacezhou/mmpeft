from typing import Any, Optional
from mmengine.visualization.vis_backend import BaseVisBackend,force_init_env
from mmengine.registry import VISBACKENDS
from mmengine.logging import MMLogger
# logger = MMLogger.get_current_instance()

@VISBACKENDS.register_module()
class NNIVisBackend(BaseVisBackend):
    def __init__(self, report_key='accuracy/top1', **kw):
        super().__init__(**kw)
        self.report_key = report_key

    def _init_env(self):
        try:
            import nni
            self._nni = nni
        except ImportError:
            MMLogger.get_current_instance().error('Please install nni to use NNIVisBackend')
            raise
        pass

    def experiment(self):
        return self._nni

    @force_init_env
    def add_scalar(self, name, value, strp, **kwargs):
        if self.report_key == name:
            self._nni.report_intermediate_result(value)
            self._final_report = value
    
    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Useless parameter. Wandb does not
                need this parameter. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        if self.report_key in scalar_dict:
            value = scalar_dict[self.report_key]
            self._nni.report_intermediate_result(value)
            self._final_report = value

    def close(self) -> None:
        self._nni.report_final_result(self._final_report)
        return super().close()