#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#如果运行的话，更改code_dir目录
CODE_DIR="/home/work/work/modifyAI/textCNN"
MODEL_DIR=$CODE_DIR/model
TRAIN_DATA_DIR=$CODE_DIR/data_set

nohup python3 $CODE_DIR/model.py \
    --is_train=true \
    --num_epochs=200 \
    --save_checkpoints_steps=100 \
    --keep_checkpoint_max=50 \
    --batch_size=64 \
    --positive_data_file=$TRAIN_DATA_DIR/polarity.pos \
    --negative_data_file=$TRAIN_DATA_DIR/polarity.neg \
    --model_dir=$MODEL_DIR > $CODE_DIR/train_log.txt 2>&1 &