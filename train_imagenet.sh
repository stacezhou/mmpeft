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
train_dataloader.dataset.data_root=data/imagenet/images/val \
val_dataloader.dataset.data_root=data/imagenet/images/val \
test_dataloader.dataset.data_root=data/imagenet/images/val