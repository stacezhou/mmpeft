# 配置环境
conda create -n mmpeft python=3.8
conda activate mmpeft
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
pip install openmim
mim install --no-cache-dir -e .
pip install ftfy regex wandb

# 如果需要 nni
pip install nni==2.10
pip install "typeguard<3"

# todo
修改了 mmengine.hooks.checkpoint_hook.py:255 行     self.out_dir = runner._log_dir

# convert weight
python tools/convert_weight.py --arch 'ViT-B/16' --download_root 'data' --save_path vit-b-16.pth

# test
python tools/test.py myconfigs/clip_vit_b16.py vit-b-16.pth