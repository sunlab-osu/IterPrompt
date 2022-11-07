#!/bin/bash
#SBATCH --account=<->
#SBATCH --job-name=<->
#SBATCH --time=<->
#SBATCH --nodes=1 --ntasks-per-node=5
#SBATCH --gpus-per-node=1
#SBATCH --output=<->

echo "Job started..."

cd ..

MODEL_TYPE=seq2seq

MODEL_PATH=facebook/bart-large
DATA_PATH=dataset_2wiki_0.1/RD_Oracle/
OUTPUT_DIR=<save directory>
MAX_SEQ_LENGTH=145
MAX_LENGTH=40
TRAIN_BATCH_SIZE=16

CUDA_VISIBLE_DEVICES=0 python main.py \
--model_type $MODEL_TYPE \
--data_dir $DATA_PATH \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--learning_rate 4e-5 \
--save_step 1000 \
--num_train_epochs 10 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--gradient_accumulation_steps 2 \
--evaluation_metric qa

echo "Job finished!"


