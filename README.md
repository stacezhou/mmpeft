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
修改了 mmengine.hooks.checkpoint_hook.py:255 行    
self.out_dir = runner._log_dir 
解决 nni 并行多组程序时，模型权重保存重复的问题。
但这样会导致不用 nni 时，无法 resume 训练。因为 log_dir 会变。
不用 nni 可以不管这部分。


# convert weight
python tools/convert_weight.py --arch 'ViT-B/32' --save_path data/clip-ViT-B-32.pth

# test
python tools/test.py myconfigs/clip_vit_b32.py data/clip-ViT-B-32.pth \
--cfg-options \
test_dataloader.dataset.data_root=data/imagenet/images/val

# multi-gpu test
bash tools/dist_test.sh myconfigs/clip_vit_b32.py data/clip-ViT-B-32.pth 4 \
--cfg-options \
test_dataloader.dataset.data_root=data/imagenet/images/val

# lora train
python tools/train.py myconfigs/clip_vit_b32.py --work-dir work_dirs/clip_vit_b32 \
--cfg-options \
load_from=data/clip-ViT-B-32.pth \
model.peft.type=LoRA \
model.peft.rank=128 \
model.peft.scale=128 \
model.peft.pattern=visual \
train_dataloader.batch_size=128 \
optim_wrapper.optimizer.lr=0.001 \
default_hooks.checkpoint.interval=2 \
default_hooks.checkpoint.save_last=True \
train_dataloader.dataset.data_root=data/imagenet/images/train \
val_dataloader.dataset.data_root=data/imagenet/images/val \
test_dataloader.dataset.data_root=data/imagenet/images/val

# multi-gpu train
bash tools/dist_train.sh myconfigs/clip_vit_b32.py 4 --work-dir work_dirs/clip_vit_b32_train \
--cfg-options \
load_from=data/clip-ViT-B-32.pth \
model.peft.type=LoRA \
model.peft.rank=128 \
model.peft.scale=128 \
model.peft.pattern=visual \
train_dataloader.batch_size=128 \
optim_wrapper.optimizer.lr=0.001 \
default_hooks.checkpoint.interval=2 \
default_hooks.checkpoint.save_last=True \
train_dataloader.dataset.data_root=data/imagenet/images/train \
val_dataloader.dataset.data_root=data/imagenet/images/val \
test_dataloader.dataset.data_root=data/imagenet/images/val