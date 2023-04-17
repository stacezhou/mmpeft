_base_ = [
    'datasets/imagenet_224.py',
    'CLIP/clip-vit-base-p32.py',
    'schedules/default_runtime.py',
    'schedules/default_schedule.py'
]
# load_from = 'data/clip-ViT-B-32.pth'
model = dict(
    peft = dict(
        type = 'LoRA',
        rank = 128,
        scale = 128,
        # pattern = 'visual|transomfer',
        pattern = 'visual', # 只微调视觉分支
    )
)