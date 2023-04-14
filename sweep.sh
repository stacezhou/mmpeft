BATCH_SIZE=256
for RANK in 1 2 4 8 16 32 ; do 
for SCALE in 4 8 16 32 ; do
for LR in 1e-2 1e-3 1e-4 1e-5 ; do
    python tools/train.py myconfigs/clip_vit_b32.py --cfg-options \
        model.peft.rank=$RANK \
        model.peft.scale=$SCALE \
        optim_wrapper.optimizer.lr=$LR \
        train_dataloader.batch_size=$BATCH_SIZE \
        --amp
done    
done
done