_base_ = [
    'dtd_dataset.py',
    '_base_/schedules/imagenet_bs256.py', '_base_/default_runtime.py'
]

model = dict(
    type='CLIPClassifier',
    embed_dim = 512, 
    image_resolution = 224,
    vision_layers = 12,
    vision_width = 768,
    vision_patch_size = 16,
    context_length = 77,
    vocab_size = 49408,
    transformer_width = 512,
    transformer_heads = 8,
    transformer_layers = 12,
)
custom_hooks = [
    dict(type='ClipSetPrompts')
]