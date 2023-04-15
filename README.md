第一次运行时在项目目录执行
mim install --no-cache-dir -e .
pip install ftfy regex wandb
pip install nni==2.10
pip install "typeguard<3"
修改了 mmengine.hooks.checkpoint_hook.py:255 行     self.out_dir = runner._log_dir
