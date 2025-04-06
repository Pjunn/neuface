#!/bin/bash
DATASET_NAME="celebv_hq"
DATASET_PATH="$DATA_DIR/CelebV_HQ"
BATCH_SIZE=20
CUDA_VISIBLE_DEVICES="0"

NUM_SAMPLES=`find ${DATASET_PATH}/images -mindepth 1 -maxdepth 1 -type d | wc -l`
echo "Found ${NUM_SAMPLES} samples"

CNT=0
DIR=`find ${DATASET_PATH}/images -mindepth 1 -maxdepth 1 -type d | sort`

for dir in $DIR
do
    ((CNT++))
    echo "Now Processing $dir ( ${CNT} / ${NUM_SAMPLES} )"
    echo "===================================================================================================="
    python neuface_optim.py --cfg configs/neuface_celebv.yml --test_seq_path $dir --batch_size ${BATCH_SIZE} --dataset ${DATASET_NAME} --cuda_device $CUDA_VISIBLE_DEVICES
    echo "DONE $dir ( ${CNT} / ${NUM_SAMPLES} )"
done
