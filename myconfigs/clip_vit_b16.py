_base_ = [
    'datasets/dtd_dataset_224.py',
    'CLIP/clip-vit-base-p16.py',
    'schedules/default_runtime.py',
    'schedules/default_schedule.py'
]
load_from = 'data/clip-ViT-B-32.pth'
model = dict(
#     peft = dict(
#         type = 'LoRA',
#         rank = 2,
#         scale = 2,
#         pattern = 'visual|transomfer',
#     )
)