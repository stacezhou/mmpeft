DATA_ROOT=data/dtd/images
TRAIN_DIR=train
for REPEAT in A B C ; do
for SHOTS in 1 4 16; do
rm -rf ${DATA_ROOT}/fewshot_train${SHOTS}_${REPEAT} 
python tools/make_few_shot_dataset.py --data-dir ${DATA_ROOT}/${TRAIN_DIR} --save-dir ${DATA_ROOT}/fewshot_train${SHOTS}_${REPEAT}  --shots $SHOTS
done
done