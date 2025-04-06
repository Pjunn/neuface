#!/bin/bash
DATASET_NAME="MEAD"
DATASET_PATH="$DATA_DIR/MEAD"
BATCH_SIZE=140
CUDA_VISIBLE_DEVICES="0"

# Add IDENTITIES as you want.
#IDENTITIES=("W028" "W029" "W033" "W035" "W036" "W037" "W038")
IDENTITIES=("W024")

for identity in ${IDENTITIES[@]}
do
    CNT=0
    NUM_SAMPLES=`find ${DATASET_PATH}/${identity}/images -mindepth 3 -maxdepth 3 -type d | wc -l`
    echo "Looking ${identity}, found ${NUM_SAMPLES} sequences."
    DIR=`find ${DATASET_PATH}/${identity}/images -mindepth 3 -maxdepth 3 -type d | sort`
    for dir in ${DIR[@]}
    do
        ((CNT++))
        echo "Now Processing $dir ( ${CNT} / ${NUM_SAMPLES} )"
        python neuface_optim.py --cfg configs/neuface_mead.yml --test_seq_path $dir --batch_size ${BATCH_SIZE} --dataset ${DATASET_NAME} --cuda_device $CUDA_VISIBLE_DEVICES
        echo "DONE $dir ( ${CNT} / ${NUM_SAMPLES} )"
    done
done
