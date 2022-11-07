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

MODEL_PATH=<path to PLM>
DATA_PATH=dataset_lot/IP/
OUTPUT_DIR=<save directory>

MAX_SEQ_LENGTH=64
MAX_LENGTH=64
TRAIN_BATCH_SIZE=16
LEARNING_RATE=4e-3
NUM_EN_PT=80
NUM_DE_PT=80

CUDA_VISIBLE_DEVICES=0 python main.py \
--model_type $MODEL_TYPE \
--data_dir $DATA_PATH \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--save_step 500 \
--num_train_epochs 50 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--gradient_accumulation_steps 8 \
--evaluation_metric qa \
--static_prompt_tuning_mode \
--use_decoder_pt \
--num_encoder_prompt_tokens $NUM_EN_PT \
--num_decoder_prompt_tokens $NUM_DE_PT

echo "Job finished!"


