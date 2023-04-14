_base_ = [
    'dtd_dataset.py',
    'CLIP/clip-vit-base-p32.py',
    'debug_schedule.py',
    '_base_/default_runtime.py'
]
load_from = 'data/clip-ViT-B-32.pth'
model = dict(
    peft = dict(
        type = 'LoRA',
        rank = 2,
        scale = 2,
    )
)