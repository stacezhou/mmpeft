_base_ = [
    '../datasets/dtd_dataset_224.py',
    '../CLIP/clip-vit-base-p32.py',
    '../schedules/default_runtime.py',
]
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5))
load_from = 'data/clip-ViT-B-32.pth'
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[12, 16, 20], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()