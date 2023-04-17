_base_ = [
    'datasets/imagenet_224.py',
    'CLIP/clip-vit-base-p32.py',
    'schedules/default_runtime.py',
]
# load_from = 'data/clip-ViT-B-32.pth'
model = dict(
    # peft = dict(
    #     type = 'LoRA',
    #     rank = 128,
    #     scale = 128,
    #     # pattern = 'visual|transomfer',
    #     pattern = 'visual', # 只微调视觉分支
    # )
)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[12, 16, 20], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()