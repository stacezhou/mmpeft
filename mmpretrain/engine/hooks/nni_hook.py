from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.dist import master_only

@HOOKS.register_module()
class NNIHook(Hook):
    @master_only
    def after_run(self, runner):
        if 'NNIVisBackend' in runner.visualizer._vis_backends:
            vis = runner.visualizer._vis_backends['NNIVisBackend']
            vis._nni.report_final_result(vis._final_report)
