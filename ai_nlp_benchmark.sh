#!/bin/bash

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

source ${HOME}/.bashrc

export GPUS=4
export MODEL=bert-large-cased

echo ${MODEL}

for TASK_NAME in qqp mnli qnli
do
    echo ${TASK_NAME}
    accelerate launch --num_processes ${GPUS} glue_finetune.py \
    --model_name_or_path ${MODEL} \
    --task_name $TASK_NAME \
    --max_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --output_dir ${TASK_NAME}_results/
done

