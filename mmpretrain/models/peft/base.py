## 原则是，保留原有模型结构以及参数；但是使用新的 forward 函数
class BasePEFT:
    def __init__(self, freeze_other_params=True):
        self.freeze_other_params = freeze_other_params

    def __call__(self, module):
        if self.freeze_other_params:
            for param in module.parameters():
                param.requires_grad = False

        self._recur_add_parameter(module)
        self._recur_change_forward(module)
        self.count_train_params(module)
        return module

    def count_train_params(self, module):
        params_to_update = []
        for param in module.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        total_params = sum(p.numel() for p in module.parameters())
        logger.info(f"Training params count: {total_params}")
        module._num_trainable_params = total_params
    
    def add_parameter(self, module, child, path):
        # Add parameters to the module
        raise NotImplementedError
    
    def change_forward(self, module, child, path):
        # Change the forward function of the module
        raise NotImplementedError

    def _recur_add_parameter(self, module, path='.'):
        # Recursively add peft parameters to the module
        for name, child in list(module.named_children()):
            child_path = path + '.' + name
            self._recur_add_parameter(child, child_path)
            self.add_parameter(module, child, child_path)
            
    def _recur_change_forward(self, module, path='.'):
        for name, child in module.named_children():
            child_path = path + '.' + name
            self._recur_change_forward(child, child_path)
            self.change_forward(module, child, child_path)