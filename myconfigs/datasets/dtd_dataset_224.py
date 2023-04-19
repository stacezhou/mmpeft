dataset_type = 'CustomDataset'
batch_size = 128
# train_data_dir = 'data/dtd/images/train'
train_data_dir = 'data/dtd/images/fewshot_train16_A'
test_data_dir = 'data/dtd/images/test'
classes = ['banded','blotchy','braided','bubbly','bumpy','chequered','cobwebbed','cracked','crosshatched','crystalline','dotted','fibrous','flecked','freckled','frilly','gauzy','grid','grooved','honeycombed','interlaced','knitted','lacelike','lined','marbled','matted','meshed','paisley','perforated','pitted','pleated','polka-dotted','porous','potholed','scaly','smeared','spiralled','sprinkled','stained','stratified','striped','studded','swirly','veined','waffled','woven','wrinkled','zigzagged']
templates = ['{} texture.']
input_size = 224

img_norm_cfg = dict(
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=True
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_size=input_size, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=input_size, interpolation='bicubic', backend='pillow'),
    dict(type='CenterCrop', crop_size=input_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=train_data_dir,
        templates=templates,
        classes = classes,
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=test_data_dir,
        templates=templates,
        classes = classes,
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator