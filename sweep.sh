BATCH_SIZE=64

SCALE=8
LR=1e-3
for RANK in 2 8 32 ; do 
    python tools/train.py myconfigs/clip_vit_b16.py --cfg-options \
        model.peft.rank=$RANK \
        model.peft.scale=$SCALE \
        train_dataloader.dataset.data_root=data/dtd/images/train \
        optim_wrapper.optimizer.lr=$LR \
        train_dataloader.batch_size=$BATCH_SIZE \
        load_from=data/clip-ViT-B-16.pth
done

RANK=8
LR=1e-3
for SCALE in 8 32 ; do
    python tools/train.py myconfigs/clip_vit_b16.py --cfg-options \
        model.peft.rank=$RANK \
        model.peft.scale=$SCALE \
        train_dataloader.dataset.data_root=data/dtd/images/train \
        optim_wrapper.optimizer.lr=$LR \
        train_dataloader.batch_size=$BATCH_SIZE \
        load_from=data/clip-ViT-B-16.pth
done

RANK=8
SCALE=8
for LR in 1e-3 1e-4; do
    python tools/train.py myconfigs/clip_vit_b16.py --cfg-options \
        model.peft.rank=$RANK \
        model.peft.scale=$SCALE \
        train_dataloader.dataset.data_root=data/dtd/images/train \
        optim_wrapper.optimizer.lr=$LR \
        train_dataloader.batch_size=$BATCH_SIZE \
        load_from=data/clip-ViT-B-16.pth
done    