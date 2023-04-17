# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[12, 16, 20], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)