dataset_type = 'CustomDataset'

# dataset settings
# data_preprocessor = dict(
#     num_classes=47,
#     # RGB format normalization parameters
#     mean=[125.307, 122.961, 113.8575],
#     std=[51.5865, 50.847, 51.255],
#     # loaded images are already RGB format
#     to_rgb=False)
classes = ['banded','blotchy','braided','bubbly','bumpy','chequered','cobwebbed','cracked','crosshatched','crystalline','dotted','fibrous','flecked','freckled','frilly','gauzy','grid','grooved','honeycombed','interlaced','knitted','lacelike','lined','marbled','matted','meshed','paisley','perforated','pitted','pleated','polka-dotted','porous','potholed','scaly','smeared','spiralled','sprinkled','stained','stratified','striped','studded','swirly','veined','waffled','woven','wrinkled','zigzagged']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_size=224, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/dtd/images/train',
        templates=['a texture of {}.'],
        classes = classes,
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/dtd/images/val',
        templates=['a texture of {}.'],
        classes = classes,
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
