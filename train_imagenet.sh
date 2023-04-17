python tools/train.py myconfigs/clip_vit_b32.py \
--work-dir work_dirs/clip_vit_b32 \
--cfg-options \
load_from=data/clip-ViT-B-32.pth \
train_dataloader.batch_size=128 \
optim_wrapper.optimizer.lr=0.001 \
default_hooks.checkpoint.interval=-1 \
default_hooks.checkpoint.save_last=True \
train_dataloader.dataset.data_root=${TRAIN_DIR} \
val_dataloader.dataset.data_root=${VAL_DIR} \
test_dataloader.dataset.data_root=${TEST_DIR} \
model.peft.rank=128 