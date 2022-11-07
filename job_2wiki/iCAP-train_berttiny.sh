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
DATA_PATH=dataset_2wiki_0.1/IP/
OUTPUT_DIR=<save directory>

MAX_SEQ_LENGTH=130
MAX_LENGTH=50
TRAIN_BATCH_SIZE=16
LEARNING_RATE=8e-5
NUM_EN_PT=30
NUM_DE_PT=30

CUDA_VISIBLE_DEVICES=0 python main.py \
--model_type $MODEL_TYPE \
--data_dir $DATA_PATH \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size 32 \
--learning_rate $LEARNING_RATE \
--save_step 6000 \
--num_train_epochs 70 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--gradient_accumulation_steps 2 \
--evaluation_metric qa \
--prompt_tuning_mode \
--use_decoder_pt \
--num_encoder_prompt_tokens $NUM_EN_PT \
--num_decoder_prompt_tokens $NUM_DE_PT \
--ptencoder_name prajjwal1/bert-tiny

echo "Job finished!"


