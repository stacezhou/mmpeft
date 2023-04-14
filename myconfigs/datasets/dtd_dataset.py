dataset_type = 'CustomDataset'
batch_size = 128
img_norm_cfg = dict(
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=True
)
classes = ['banded','blotchy','braided','bubbly','bumpy','chequered','cobwebbed','cracked','crosshatched','crystalline','dotted','fibrous','flecked','freckled','frilly','gauzy','grid','grooved','honeycombed','interlaced','knitted','lacelike','lined','marbled','matted','meshed','paisley','perforated','pitted','pleated','polka-dotted','porous','potholed','scaly','smeared','spiralled','sprinkled','stained','stratified','striped','studded','swirly','veined','waffled','woven','wrinkled','zigzagged']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_size=224, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224, interpolation='bicubic', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/dtd/images/train',
        templates=['{} texture.'],
        classes = classes,
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/dtd/images/test',
        templates=['{} texture.'],
        classes = classes,
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
